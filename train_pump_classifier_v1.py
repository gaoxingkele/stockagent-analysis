"""启动子分类器 PoC v1 (V7c 进化).

目标: 训练 LGBM 二分类模型, 预测每只股每日 "5 日启动子" 概率.
启动子定义: 未来 5 日最高涨幅 ≥10% AND 最大回撤 ≤5%

vs r5/r20 回归模型:
  - 回归: 预测平均收益, 不专注 "启动"
  - 二分类: 专门学 "启动 vs 非启动" 模式

输出: output/production/r5_pump_lgbm_v1/
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window, EXCLUDE

PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"
OUT_NAME = "r5_pump_lgbm_v1"

TRAIN_START = "20230101"
TRAIN_END   = "20250930"   # 严格 cut, 同长 OOS 系列
DATA_END    = "20260522"   # 最新

# 启动子定义
PUMP_UP_THRESHOLD = 0.10   # 5 日涨 ≥10%
PUMP_DD_THRESHOLD = 0.05   # 回撤 ≤5%
PUMP_FORWARD = 5


def compute_pump_label(daily_cache_dir: Path, start: str, end: str) -> pd.DataFrame:
    """从 daily cache 算启动子 label (二分类 0/1).

    label = 1 if 未来 5 日 max(high) / next_open - 1 ≥ 0.10
            AND 未来 5 日 min(low) / next_open - 1 ≥ -0.05
    """
    print(f"  从 daily cache 算启动子 label ...", flush=True)
    files = sorted(daily_cache_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low"])
                for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    N = PUMP_FORWARD
    big["max_high_next"] = (big.groupby("ts_code")["high"].apply(
        lambda x: x.rolling(N, min_periods=N).max().shift(-N)).reset_index(level=0, drop=True))
    big["min_low_next"] = (big.groupby("ts_code")["low"].apply(
        lambda x: x.rolling(N, min_periods=N).min().shift(-N)).reset_index(level=0, drop=True))

    big["upside"] = big["max_high_next"] / big["next_open"] - 1
    big["downside"] = big["min_low_next"] / big["next_open"] - 1
    big["is_pump"] = ((big["upside"] >= PUMP_UP_THRESHOLD) &
                       (big["downside"] >= -PUMP_DD_THRESHOLD)).astype(int)
    # 仅保留有 forward label 的 (即未来 N 日数据完整)
    pump = big.dropna(subset=["max_high_next", "min_low_next"])[
        ["ts_code", "trade_date", "is_pump"]
    ]
    print(f"  pump label 行数: {len(pump):,}, 正样本: {pump['is_pump'].sum():,} "
           f"({pump['is_pump'].mean()*100:.2f}%)", flush=True)
    return pump


def precision_at_k_per_day(val_df: pd.DataFrame, top_k: int) -> dict:
    """每日按 pred 取 Top K, 算 precision (Top K 里启动子比例)."""
    rows = []
    for d_, g in val_df.groupby("trade_date"):
        if len(g) < top_k * 2: continue
        top = g.nlargest(top_k, "pred")
        n_pump = top["is_pump"].sum()
        rows.append({"date": d_, "top_k": top_k, "n_pump": n_pump,
                      "precision": n_pump / top_k})
    if not rows:
        return {"top_k": top_k, "precision_avg": 0, "n_days": 0}
    df = pd.DataFrame(rows)
    return {"top_k": top_k, "precision_avg": float(df["precision"].mean()),
             "n_days": len(df)}


def main():
    t0 = time.time()
    print(f"=== 启动子分类器 PoC v1 训练 ===\n", flush=True)
    print(f"  目标: 5 日启动子 (涨 ≥{PUMP_UP_THRESHOLD*100:.0f}% & "
           f"回撤 ≤{PUMP_DD_THRESHOLD*100:.0f}%)", flush=True)
    print(f"  cut: < {TRAIN_END}\n", flush=True)

    out_dir = PROD / OUT_NAME
    if (out_dir / "classifier.txt").exists():
        print(f"[{OUT_NAME}] 已存在, 跳过 (rm 后重训)"); return

    # 1. 加载日线 factor (ST 已源头排除)
    print("[1] load_window 加载日线 factor (235 特征) ...", flush=True)
    df = load_window(TRAIN_START, DATA_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"  加载完成: {len(df):,}", flush=True)

    # 2. merge long_return_features (11 偏置因子)
    if LONG_FEAT_P.exists():
        print("[2] merge long_return_features ...", flush=True)
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        df = df.merge(lf, on=["ts_code", "trade_date"], how="left")
        print(f"  merged, 共 {len(df.columns)} 列", flush=True)

    # 3. 计算启动子 label
    print("\n[3] 启动子 label (从 daily cache) ...", flush=True)
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    pump = compute_pump_label(daily_dir, TRAIN_START, DATA_END)
    df = df.merge(pump, on=["ts_code", "trade_date"], how="inner")
    print(f"  merge label 后: {len(df):,}, 正样本 {df['is_pump'].sum():,} "
           f"({df['is_pump'].mean()*100:.2f}%)", flush=True)

    # 4. industry_id (categorical)
    industries = pd.Categorical(df["industry"].fillna("unknown"))
    df["industry_id"] = industries.codes

    # 5. 特征列
    EXC = set(EXCLUDE) | {"is_pump"}
    feat_cols = [c for c in df.columns
                  if c not in EXC and pd.api.types.is_numeric_dtype(df[c])]
    print(f"\n[4] 特征列: {len(feat_cols)}", flush=True)

    # 6. clip inf/NaN
    for c in feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    # 7. split
    train_df = df[df["trade_date"] < TRAIN_END].copy()
    val_df = df[df["trade_date"] >= TRAIN_END].copy()
    print(f"  train: {len(train_df):,}, val: {len(val_df):,}", flush=True)
    print(f"  train pump 率: {train_df['is_pump'].mean()*100:.2f}%", flush=True)
    print(f"  val pump 率: {val_df['is_pump'].mean()*100:.2f}%", flush=True)

    if len(train_df) > 2_000_000:
        train_df = train_df.sample(n=2_000_000, random_state=42).reset_index(drop=True)
        print(f"  subsample 到 200 万", flush=True)

    X_train = train_df[feat_cols].astype("float32")
    y_train = train_df["is_pump"].astype("int8")
    X_val = val_df[feat_cols].astype("float32")
    y_val = val_df["is_pump"].astype("int8")

    # 8. 训练 LGBM binary classifier
    print(f"\n[5] LGBM 二分类训练 (AUC 主指标) ...", flush=True)
    clf = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=3000, learning_rate=0.03,
        num_leaves=63, min_child_samples=300,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        is_unbalance=True,   # 处理 12% 正样本
        random_state=42, n_jobs=4, verbose=-1,
    )
    clf.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(100, first_metric_only=True),
                          lgb.log_evaluation(100)])

    # 9. 保存
    out_dir.mkdir(exist_ok=True, parents=True)
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.categories,
                                                    range(len(industries.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": "is_pump", "model_type": "binary_classifier",
        "label_def": {"forward_days": PUMP_FORWARD,
                       "upside_threshold": PUMP_UP_THRESHOLD,
                       "drawdown_threshold": PUMP_DD_THRESHOLD},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 10. 评估
    y_pred = clf.predict_proba(X_val)[:, 1]
    auc = clf.best_score_["valid_0"]["auc"]
    print(f"\n[OK] {OUT_NAME} best AUC = {auc:.4f}, best_iter = {clf.best_iteration_}",
           flush=True)

    # precision@K per day
    val_eval = val_df[["ts_code", "trade_date", "is_pump"]].copy()
    val_eval["pred"] = y_pred

    print(f"\n--- 每日 Top K 启动子 precision ---")
    pump_rate = val_eval["is_pump"].mean()
    print(f"  基线 (全市场 pump 率): {pump_rate*100:.2f}%")
    prec_results = []
    for k in [5, 10, 20, 50, 100]:
        r = precision_at_k_per_day(val_eval, k)
        amplify = r["precision_avg"] / pump_rate if pump_rate > 0 else 0
        print(f"  Top {k:3d}: 平均 precision = {r['precision_avg']*100:.2f}% "
               f"(放大 {amplify:.2f}x, n_days={r['n_days']})", flush=True)
        prec_results.append({**r, "amplify": amplify})

    Path(out_dir / "meta.json").write_text(json.dumps({
        "auc": float(auc),
        "best_iter": int(clf.best_iteration_),
        "n_train": len(train_df), "n_val": len(val_df),
        "pump_rate_val": float(pump_rate),
        "precision_at_k": prec_results,
        "version": "pump_v1",
        "train_window": f"< {TRAIN_END}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
