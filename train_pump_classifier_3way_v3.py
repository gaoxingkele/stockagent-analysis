"""启动子三分类 v3 (multiclass softmax).

替代 v1 两个独立二分类 (pump_up + pump_down), 用一个三分类模型:
  label = 0: 中性 (未来 5 日既无大涨也无大跌)
  label = 1: 跌启动子 (跌 ≥10% & 反弹 ≤5%)
  label = 2: 涨启动子 (涨 ≥10% & 回撤 ≤5%)

注: pump_up=1 和 pump_down=1 物理上互斥 (一只股 5 日 forward 不可能既涨10%又跌10%).

softmax 强制 P(0) + P(1) + P(2) = 1, 隐含"涨高时跌必低"互斥 — 用户洞察的体现.

输出: output/production/r5_pump_3way_lgbm_v3/
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
OUT_NAME = "r5_pump_3way_lgbm_v3"

TRAIN_START = "20230101"
TRAIN_END   = "20250930"
DATA_END    = "20260522"

PUMP_UP_THRESHOLD = 0.10
PUMP_DD_THRESHOLD = 0.05
PUMP_FORWARD = 5


def compute_labels_3way(daily_cache_dir: Path) -> pd.DataFrame:
    """三分类 label.
    0: 中性
    1: 跌启动子 (下跌 ≥10% & 反弹 ≤5%)
    2: 涨启动子 (上涨 ≥10% & 回撤 ≤5%)
    """
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

    big["pump_3way"] = 0
    pump_up_mask = (big["upside"] >= PUMP_UP_THRESHOLD) & \
                    (big["downside"] >= -PUMP_DD_THRESHOLD)
    pump_dn_mask = (big["downside"] <= -PUMP_UP_THRESHOLD) & \
                    (big["upside"] <= PUMP_DD_THRESHOLD)
    big.loc[pump_up_mask, "pump_3way"] = 2
    big.loc[pump_dn_mask, "pump_3way"] = 1
    # 互斥检查
    overlap = (pump_up_mask & pump_dn_mask).sum()
    print(f"  涨跌互斥校验: overlap={overlap} (应该 0)", flush=True)

    return big.dropna(subset=["max_high_next"])[
        ["ts_code", "trade_date", "pump_3way"]
    ]


def precision_at_k(val_df, top_k, pred_col, label_target_class):
    """Top K (by pred_col 降序) 内, label == label_target_class 的比例."""
    rows = []
    for d_, g in val_df.groupby("trade_date"):
        if len(g) < top_k * 2: continue
        top = g.nlargest(top_k, pred_col)
        hit = (top["pump_3way"] == label_target_class).sum()
        rows.append({"precision": hit / top_k})
    if not rows: return 0
    return float(pd.DataFrame(rows)["precision"].mean())


def main():
    t0 = time.time()
    print(f"=== 启动子三分类 v3 (multiclass softmax) ===\n", flush=True)
    print(f"  类别: 0=中性, 1=跌启动子, 2=涨启动子", flush=True)
    print(f"  cut: < {TRAIN_END}\n", flush=True)

    out_dir = PROD / OUT_NAME
    if (out_dir / "classifier.txt").exists():
        print(f"[{OUT_NAME}] 已存, 跳过"); return

    print("[1] load_window ...", flush=True)
    df = load_window(TRAIN_START, DATA_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        print("[2] merge long_return_features ...", flush=True)
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        df = df.merge(lf, on=["ts_code", "trade_date"], how="left")

    print("\n[3] 三分类 label ...", flush=True)
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    pump = compute_labels_3way(daily_dir)
    df = df.merge(pump, on=["ts_code", "trade_date"], how="inner")
    cnt = df["pump_3way"].value_counts().sort_index()
    print(f"  类别分布: 中性 {cnt.get(0, 0):,} ({cnt.get(0,0)/len(df)*100:.1f}%), "
           f"跌 {cnt.get(1, 0):,} ({cnt.get(1,0)/len(df)*100:.1f}%), "
           f"涨 {cnt.get(2, 0):,} ({cnt.get(2,0)/len(df)*100:.1f}%)", flush=True)

    industries = pd.Categorical(df["industry"].fillna("unknown"))
    df["industry_id"] = industries.codes
    EXC = set(EXCLUDE) | {"pump_3way", "is_pump_up", "is_pump_down", "is_pump"}
    feat_cols = [c for c in df.columns
                  if c not in EXC and pd.api.types.is_numeric_dtype(df[c])]
    print(f"\n[4] 特征列: {len(feat_cols)}", flush=True)

    for c in feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    train_df = df[df["trade_date"] < TRAIN_END].copy()
    val_df = df[df["trade_date"] >= TRAIN_END].copy()
    print(f"  train: {len(train_df):,}, val: {len(val_df):,}", flush=True)
    print(f"  train 类别: {dict(train_df['pump_3way'].value_counts().sort_index())}", flush=True)
    print(f"  val 类别: {dict(val_df['pump_3way'].value_counts().sort_index())}", flush=True)

    if len(train_df) > 2_000_000:
        train_df = train_df.sample(n=2_000_000, random_state=42).reset_index(drop=True)
        print(f"  subsample 到 200 万", flush=True)

    X_train = train_df[feat_cols].astype("float32")
    y_train = train_df["pump_3way"].astype("int8")
    X_val = val_df[feat_cols].astype("float32")
    y_val = val_df["pump_3way"].astype("int8")

    print(f"\n[5] LGBM multiclass 训练 ...", flush=True)
    clf = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        metric="multi_logloss",
        n_estimators=3000, learning_rate=0.03,
        num_leaves=63, min_child_samples=300,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        class_weight="balanced",   # 处理类别不平衡
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
        "target": "pump_3way", "model_type": "multiclass_3way",
        "class_meaning": {"0": "neutral", "1": "pump_down", "2": "pump_up"},
        "version": "v3_multiclass",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 评估
    y_pred_proba = clf.predict_proba(X_val)   # [n, 3]
    # multi_logloss
    logloss = clf.best_score_["valid_0"]["multi_logloss"]
    print(f"\n[OK] {OUT_NAME} multi_logloss = {logloss:.4f}, best_iter = {clf.best_iteration_}",
           flush=True)

    val_eval = val_df[["ts_code", "trade_date", "pump_3way"]].copy()
    val_eval["P_neutral"] = y_pred_proba[:, 0]
    val_eval["P_down"] = y_pred_proba[:, 1]
    val_eval["P_up"] = y_pred_proba[:, 2]

    # 对比 v1 的 precision@K
    print(f"\n--- v3 vs v1 (二分类) precision@K 对比 ---\n", flush=True)
    print(f"  --- 涨启动子 precision@K (P_up 降序 Top K, label=2 hit rate) ---")
    pump_up_rate = (val_eval["pump_3way"] == 2).mean()
    print(f"  基线 (val 涨率): {pump_up_rate*100:.2f}%", flush=True)
    prec_up = []
    for k in [5, 10, 20, 50, 100]:
        p = precision_at_k(val_eval, k, "P_up", 2)
        amplify = p / pump_up_rate if pump_up_rate > 0 else 0
        print(f"  Top {k:3d}: precision = {p*100:.2f}% (放大 {amplify:.2f}x) | "
               f"v1: ~26%/2.1x", flush=True)
        prec_up.append({"top_k": k, "precision": p, "amplify": amplify, "class": "up"})

    print(f"\n  --- 跌启动子 precision@K (P_down 降序 Top K, label=1 hit rate) ---")
    pump_dn_rate = (val_eval["pump_3way"] == 1).mean()
    print(f"  基线 (val 跌率): {pump_dn_rate*100:.2f}%", flush=True)
    prec_dn = []
    for k in [5, 10, 20, 50, 100]:
        p = precision_at_k(val_eval, k, "P_down", 1)
        amplify = p / pump_dn_rate if pump_dn_rate > 0 else 0
        print(f"  Top {k:3d}: precision = {p*100:.2f}% (放大 {amplify:.2f}x) | "
               f"v1: ~23%/3.5x", flush=True)
        prec_dn.append({"top_k": k, "precision": p, "amplify": amplify, "class": "down"})

    # 互斥验证: P_up 高时 P_down 是否真的低
    high_pup = val_eval[val_eval["P_up"] > val_eval["P_up"].quantile(0.95)]
    high_pdn = val_eval[val_eval["P_down"] > val_eval["P_down"].quantile(0.95)]
    print(f"\n--- 互斥性验证 ---")
    print(f"  P_up Top 5% 股的 P_down 均值: {high_pup['P_down'].mean():.3f} "
           f"(vs 全体均 {val_eval['P_down'].mean():.3f})", flush=True)
    print(f"  P_down Top 5% 股的 P_up 均值: {high_pdn['P_up'].mean():.3f} "
           f"(vs 全体均 {val_eval['P_up'].mean():.3f})", flush=True)

    Path(out_dir / "meta.json").write_text(json.dumps({
        "multi_logloss": float(logloss), "best_iter": int(clf.best_iteration_),
        "n_train": len(train_df), "n_val": len(val_df),
        "pump_up_rate_val": float(pump_up_rate),
        "pump_down_rate_val": float(pump_dn_rate),
        "precision_at_k_up": prec_up,
        "precision_at_k_down": prec_dn,
        "version": "v3_multiclass_3way",
        "train_window": f"< {TRAIN_END}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
