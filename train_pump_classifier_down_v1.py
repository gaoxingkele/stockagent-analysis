"""跌启动子 (负启动子) 分类器 PoC v1.

跌启动子定义:
  未来 5 日最大跌幅 ≥ 10% (min_low / next_open - 1 ≤ -0.10)
  AND 期间最大反弹 ≤ 5% (max_high / next_open - 1 ≤ +0.05)

用途: 实战减仓信号 (持仓股触发跌启动子 → 减仓), 不用于做空
对应正启动子: 未来 5 日涨 ≥10% & 回撤 ≤5%

输出: output/production/r5_pump_down_lgbm_v1/
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
OUT_NAME = "r5_pump_down_lgbm_v1"

TRAIN_START = "20230101"
TRAIN_END   = "20250930"
DATA_END    = "20260522"

# 跌启动子: 5 日跌 ≥10% & 反弹 ≤5%
PUMP_DOWN_THRESHOLD = 0.10
PUMP_REBOUND_THRESHOLD = 0.05
PUMP_FORWARD = 5


def compute_pump_down_label(daily_cache_dir: Path) -> pd.DataFrame:
    print(f"  从 daily cache 算跌启动子 label ...", flush=True)
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
    # 跌启动子: 跌 ≥10% (downside ≤ -0.10) AND 反弹 ≤ 5% (upside ≤ +0.05)
    big["is_pump_down"] = ((big["downside"] <= -PUMP_DOWN_THRESHOLD) &
                             (big["upside"] <= PUMP_REBOUND_THRESHOLD)).astype(int)
    pump = big.dropna(subset=["max_high_next", "min_low_next"])[
        ["ts_code", "trade_date", "is_pump_down"]
    ]
    print(f"  pump_down label 行数: {len(pump):,}, 正样本: {pump['is_pump_down'].sum():,} "
           f"({pump['is_pump_down'].mean()*100:.2f}%)", flush=True)
    return pump


def precision_at_k_per_day(val_df, top_k):
    rows = []
    for d_, g in val_df.groupby("trade_date"):
        if len(g) < top_k * 2: continue
        top = g.nlargest(top_k, "pred")
        rows.append({"top_k": top_k,
                       "n_pump": top["is_pump_down"].sum(),
                       "precision": top["is_pump_down"].sum() / top_k})
    if not rows:
        return {"top_k": top_k, "precision_avg": 0, "n_days": 0}
    df = pd.DataFrame(rows)
    return {"top_k": top_k, "precision_avg": float(df["precision"].mean()),
             "n_days": len(df)}


def main():
    t0 = time.time()
    print(f"=== 跌启动子分类器 PoC v1 ===\n", flush=True)
    print(f"  目标: 5 日跌 ≥{PUMP_DOWN_THRESHOLD*100:.0f}% & 反弹 ≤{PUMP_REBOUND_THRESHOLD*100:.0f}%", flush=True)
    print(f"  cut: < {TRAIN_END}\n", flush=True)

    out_dir = PROD / OUT_NAME
    if (out_dir / "classifier.txt").exists():
        print(f"[{OUT_NAME}] 已存在, 跳过"); return

    print("[1] load_window ...", flush=True)
    df = load_window(TRAIN_START, DATA_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)

    if LONG_FEAT_P.exists():
        print("[2] merge long_return_features ...", flush=True)
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        df = df.merge(lf, on=["ts_code", "trade_date"], how="left")

    print("\n[3] 跌启动子 label ...", flush=True)
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    pump = compute_pump_down_label(daily_dir)
    df = df.merge(pump, on=["ts_code", "trade_date"], how="inner")
    pump_rate = df['is_pump_down'].mean()
    print(f"  merge label 后: {len(df):,}, 正样本 {df['is_pump_down'].sum():,} "
           f"({pump_rate*100:.2f}%)", flush=True)

    industries = pd.Categorical(df["industry"].fillna("unknown"))
    df["industry_id"] = industries.codes
    EXC = set(EXCLUDE) | {"is_pump_down", "is_pump"}  # exclude both labels
    feat_cols = [c for c in df.columns
                  if c not in EXC and pd.api.types.is_numeric_dtype(df[c])]
    print(f"\n[4] 特征列: {len(feat_cols)}", flush=True)

    for c in feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    train_df = df[df["trade_date"] < TRAIN_END].copy()
    val_df = df[df["trade_date"] >= TRAIN_END].copy()
    print(f"  train: {len(train_df):,}, val: {len(val_df):,}", flush=True)
    print(f"  train pump_down 率: {train_df['is_pump_down'].mean()*100:.2f}%", flush=True)
    print(f"  val pump_down 率: {val_df['is_pump_down'].mean()*100:.2f}%", flush=True)

    if len(train_df) > 2_000_000:
        train_df = train_df.sample(n=2_000_000, random_state=42).reset_index(drop=True)
        print(f"  subsample 到 200 万", flush=True)

    X_train = train_df[feat_cols].astype("float32")
    y_train = train_df["is_pump_down"].astype("int8")
    X_val = val_df[feat_cols].astype("float32")
    y_val = val_df["is_pump_down"].astype("int8")

    print(f"\n[5] LGBM 二分类训练 ...", flush=True)
    clf = lgb.LGBMClassifier(
        objective="binary", metric="auc",
        n_estimators=3000, learning_rate=0.03,
        num_leaves=63, min_child_samples=300,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        is_unbalance=True,
        random_state=42, n_jobs=4, verbose=-1,
    )
    clf.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(100, first_metric_only=True),
                          lgb.log_evaluation(100)])

    out_dir.mkdir(exist_ok=True, parents=True)
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.categories,
                                                    range(len(industries.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": "is_pump_down", "model_type": "binary_classifier",
        "label_def": {"forward_days": PUMP_FORWARD,
                       "down_threshold": PUMP_DOWN_THRESHOLD,
                       "rebound_threshold": PUMP_REBOUND_THRESHOLD},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    y_pred = clf.predict_proba(X_val)[:, 1]
    auc = clf.best_score_["valid_0"]["auc"]
    print(f"\n[OK] {OUT_NAME} AUC = {auc:.4f}, best_iter = {clf.best_iteration_}", flush=True)

    val_eval = val_df[["ts_code", "trade_date", "is_pump_down"]].copy()
    val_eval["pred"] = y_pred
    print(f"\n--- 每日 Top K 跌启动子 precision ---")
    print(f"  基线: {pump_rate*100:.2f}%")
    prec_results = []
    for k in [5, 10, 20, 50, 100]:
        r = precision_at_k_per_day(val_eval, k)
        amplify = r["precision_avg"] / pump_rate if pump_rate > 0 else 0
        print(f"  Top {k:3d}: precision = {r['precision_avg']*100:.2f}% "
               f"(放大 {amplify:.2f}x)", flush=True)
        prec_results.append({**r, "amplify": amplify})

    Path(out_dir / "meta.json").write_text(json.dumps({
        "auc": float(auc), "best_iter": int(clf.best_iteration_),
        "n_train": len(train_df), "n_val": len(val_df),
        "pump_down_rate_val": float(val_df["is_pump_down"].mean()),
        "precision_at_k": prec_results,
        "version": "pump_down_v1",
        "train_window": f"< {TRAIN_END}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
