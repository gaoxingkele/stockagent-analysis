"""Phase 2: r1 LambdaRank + 长期收益偏置因子重训.

vs train_long_oos_lambda.py:
  - merge output/long_return_features/features.parquet (11 个新因子)
  - categorical_feature 加 4 个 decile (long_return_252d_decile, long_return_504d_decile,
                                         industry_return_504d_decile, rs_in_decile)
  - 输出 r1_next_open_v3_long_lambda_bias_nost

目标: 让模型显式区分明星股/普通股/落后股, 解决 lambdarank 残余 -2% 月化的"涨过了的
回调被错认为是 r1 信号" 问题.
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROD = ROOT / "output" / "production"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

TRAIN_END_DATE = "20250930"
MAX_TRAIN_DATES = 500
OUT_NAME = "r1_next_open_v3_long_lambda_bias_nost"
LABEL = "r1_next_open"
# Phase 2.1: 精简偏置 - 排除 6 个低 gain 偏置 (前次 gain < 100), 只留 top 5 偏置因子
LOW_GAIN_BIAS_DROP = {
    "long_return_504d", "relative_strength_252d",
    "long_return_252d_decile", "long_return_504d_decile",
    "industry_return_504d_decile", "rs_in_decile",
}
EXCLUDE_COLS = {"ts_code", "trade_time", "trade_date",
                  "r4_1h", "r20_1h", "r40_1h",
                  "r1_next_open", "r4_next_morn", "r8_next_day",
                  } | LOW_GAIN_BIAS_DROP
# Phase 2.1: decile 改 numeric (去 categorical_feature, 解决 NaN warning)
CATEGORICAL_DECILE = []  # 留空 = 全 numeric


def discretize_label_by_date(df: pd.DataFrame, label_col: str, n_bins: int = 10) -> pd.Series:
    ranks = df.groupby("trade_date")[label_col].rank(pct=True, method="first")
    bins_float = (ranks * n_bins).clip(0, n_bins - 1)
    bins_int = np.floor(bins_float.fillna(-1).values).astype("int8")
    bins = pd.Series(bins_int, index=df.index, dtype="Int8")
    return bins.where(bins >= 0)


def main():
    t0 = time.time()
    print(f"\n=== r1 LambdaRank + Bias 重训 (cut at {TRAIN_END_DATE}) ===\n", flush=True)
    print(f"输出: {OUT_NAME}", flush=True)

    out_dir = PROD / OUT_NAME
    if (out_dir / "classifier.txt").exists():
        print(f"[{OUT_NAME}] 已存在, 跳过"); return

    if not LONG_FEAT_P.exists():
        print(f"!! {LONG_FEAT_P} 不存在, 先跑 compute_long_return_features.py")
        return

    # 1. 加载 factors_v3 + EOD bar
    print(f"加载 factors_v3 EOD bar...", flush=True)
    df = pd.read_parquet(F3)
    df["trade_date"] = df["trade_date"].astype(str)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df = df[df["trade_time"].dt.hour == 15].copy()
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last").reset_index(drop=True)
    print(f"  EOD bar: {len(df):,}", flush=True)

    # ST 排除
    basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    basic = pd.read_parquet(basic_p)[["ts_code", "name"]].drop_duplicates("ts_code")
    st_codes = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    before = len(df)
    df = df[~df["ts_code"].isin(st_codes)].reset_index(drop=True)
    print(f"  ST 排除: {before - len(df):,} 行", flush=True)

    # 2. merge 长期收益偏置因子
    print(f"merge long_return_features...", flush=True)
    long_feat = pd.read_parquet(LONG_FEAT_P)
    long_feat["trade_date"] = long_feat["trade_date"].astype(str)
    print(f"  long_feat: {len(long_feat):,} 行, {len(long_feat.columns)} 列", flush=True)
    df = df.merge(long_feat, on=["ts_code", "trade_date"], how="left")
    new_cols = [c for c in long_feat.columns if c not in ("ts_code", "trade_date")]
    print(f"  merged, 新增列: {new_cols}", flush=True)
    for c in new_cols:
        cov = df[c].notna().sum() / len(df) * 100
        print(f"    {c:40s}: {cov:.1f}% 覆盖", flush=True)

    # 3. label 离散化
    df = df.dropna(subset=[LABEL])
    df = df[df[LABEL].abs() <= 20].copy()
    print(f"label 离散化 (decile)...", flush=True)
    df["r1_decile"] = discretize_label_by_date(df, LABEL, n_bins=10)
    df = df.dropna(subset=["r1_decile"]).copy()
    df["r1_decile"] = df["r1_decile"].astype(int)

    # 4. split + subsample
    train_df = df[df["trade_date"] < TRAIN_END_DATE].copy()
    val = df[df["trade_date"] >= TRAIN_END_DATE].copy()
    print(f"  train < {TRAIN_END_DATE}: {len(train_df):,}", flush=True)
    print(f"  val: {len(val):,}", flush=True)

    train_dates = sorted(train_df["trade_date"].unique())
    if len(train_dates) > MAX_TRAIN_DATES:
        rng = np.random.RandomState(42)
        keep_dates = set(rng.choice(train_dates, size=MAX_TRAIN_DATES, replace=False))
        train_df = train_df[train_df["trade_date"].isin(keep_dates)].copy()
        print(f"  subsample → {MAX_TRAIN_DATES} 日, {len(train_df):,} 样本", flush=True)

    # 5. 特征列 (含新偏置因子)
    feat_cols = [c for c in df.columns
                  if c not in EXCLUDE_COLS and c != "r1_decile"
                  and pd.api.types.is_numeric_dtype(df[c])]
    # decile 列已经被 numeric_dtype 涵盖, 但要确认它们在
    for cd in CATEGORICAL_DECILE:
        if cd not in feat_cols and cd in df.columns:
            feat_cols.append(cd)
    print(f"  特征列总数: {len(feat_cols)} (新偏置 {len(new_cols)} 含 {len(CATEGORICAL_DECILE)} categorical)",
           flush=True)

    # decile 类型要变成 int (LightGBM categorical 要求 int)
    for cd in CATEGORICAL_DECILE:
        if cd in train_df.columns:
            train_df[cd] = train_df[cd].astype("Int8").astype("Int16").fillna(-1).astype("int16")
            val[cd] = val[cd].astype("Int8").astype("Int16").fillna(-1).astype("int16")

    # clip 其他 numeric
    for c in feat_cols:
        if c in CATEGORICAL_DECILE: continue
        train_df[c] = train_df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
        val[c] = val[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    # 6. sort by trade_date (group 连续)
    train_df = train_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    val = val.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    group_train = train_df.groupby("trade_date", sort=False).size().values.tolist()
    group_val = val.groupby("trade_date", sort=False).size().values.tolist()
    print(f"  train groups {len(group_train)}, val groups {len(group_val)}", flush=True)

    X_train = train_df[feat_cols].astype("float32")
    y_train = train_df["r1_decile"].astype("int32")
    X_val = val[feat_cols].astype("float32")
    y_val = val["r1_decile"].astype("int32")
    y_val_continuous = val[LABEL].astype("float32")

    # 7. 训练
    print(f"\n训练 LambdaRanker + Bias (NDCG@10)...", flush=True)
    cat_feature_idx = [feat_cols.index(c) for c in CATEGORICAL_DECILE if c in feat_cols]
    print(f"  categorical_feature indices: {cat_feature_idx} → {[feat_cols[i] for i in cat_feature_idx]}",
           flush=True)

    # Phase 2.1: lr 减半 + n_estimators 翻倍, 让 lambdarank 更充分收敛 (Phase 2.0 只跑 46 步)
    clf = lgb.LGBMRanker(
        n_estimators=8000, learning_rate=0.015, num_leaves=63,
        min_child_samples=500, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10, 20],
        label_gain=list(range(10)),
    )
    clf.fit(X_train, y_train, group=group_train,
             eval_set=[(X_val, y_val)], eval_group=[group_val],
             eval_metric="ndcg",
             categorical_feature=cat_feature_idx if cat_feature_idx else "auto",
             callbacks=[lgb.early_stopping(150, first_metric_only=True),
                          lgb.log_evaluation(100)])

    out_dir.mkdir(exist_ok=True, parents=True)
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": LABEL, "model_type": "lambdarank_bias",
        "categorical_feature": CATEGORICAL_DECILE,
        "label_n_bins": 10, "label_discretization": "by_date_decile_first",
        "bias_features": new_cols,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. eval
    y_pred = clf.predict(X_val)
    ic_continuous = stats.pearsonr(y_pred, y_val_continuous)[0]
    rank_ic_continuous = stats.spearmanr(y_pred, y_val_continuous)[0]
    rank_ic_decile = stats.spearmanr(y_pred, y_val)[0]

    val_check = val.copy()
    val_check["pred"] = y_pred
    top10_hit = []
    for d_, g in val_check.groupby("trade_date"):
        if len(g) < 20: continue
        top_pred = g.nlargest(10, "pred")
        top_true = g.nlargest(10, LABEL)
        hit = len(set(top_pred["ts_code"]) & set(top_true["ts_code"]))
        top10_hit.append(hit / 10)
    avg_top10_hit = np.mean(top10_hit) if top10_hit else 0

    p5 = float(np.quantile(y_pred, 0.05))
    p50 = float(np.quantile(y_pred, 0.50))
    p95 = float(np.quantile(y_pred, 0.95))

    # feature importance: 看新偏置因子的排名
    importance = clf.booster_.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, importance), key=lambda x: -x[1])
    bias_in_top = [(name, imp, i+1) for i, (name, imp) in enumerate(feat_imp) if name in new_cols]
    print(f"\n--- 新偏置因子排名 (gain) ---")
    for name, imp, rank in bias_in_top:
        print(f"  #{rank:3d} {name:40s} gain={imp:.0f}", flush=True)
    print(f"\n--- 前 15 因子 ---")
    for name, imp in feat_imp[:15]:
        flag = " ⭐ (bias)" if name in new_cols else ""
        print(f"  {name:40s} gain={imp:.0f}{flag}", flush=True)

    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic_val_continuous": float(ic_continuous),
        "rank_ic_val_continuous": float(rank_ic_continuous),
        "rank_ic_val_decile": float(rank_ic_decile),
        "avg_top10_hit_rate": float(avg_top10_hit),
        "n_train": len(train_df), "n_val": len(val),
        "n_val_dates": len(group_val),
        "n_features": len(feat_cols),
        "version": "long_oos_lambda_bias", "target": LABEL,
        "train_window": f"< {TRAIN_END_DATE}",
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
        "n_train_dates": len(group_train),
        "bias_features": new_cols,
        "bias_importance": [(name, int(imp), int(rank)) for name, imp, rank in bias_in_top[:11]],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] {OUT_NAME}", flush=True)
    print(f"  best_iter={clf.best_iteration_}", flush=True)
    print(f"  IC (vs 连续 r1): {ic_continuous:.4f}", flush=True)
    print(f"  RankIC (vs 连续): {rank_ic_continuous:.4f}", flush=True)
    print(f"  日均 Top10 命中: {avg_top10_hit*100:.1f}% (随机 10%, lambda 24.8%)", flush=True)
    print(f"  锚点: {p5:.3f} / {p50:.3f} / {p95:.3f}", flush=True)
    print(f"  耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
