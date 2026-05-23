"""Phase 1: r1 模型用 LambdaRank objective 重训 (ST 排除前提下).

动机: 旧 nost 模型 IC 0.77 但 Top10 月化 -3.73% — regression objective 优化的是
全样本 MSE 而非 Top N 排序. 改 LambdaRank 直接对齐 NDCG@10 目标.

vs train_long_oos.py:
  - objective: regression → lambdarank
  - label: 连续 r1_next_open → 当日横截面 decile (0-9)
  - 训练数据按 trade_date 排序, group_sizes 传给 LGBM
  - subsample 改为按 trade_date 整组采样 (保 group 完整)
  - metric: ndcg@[5,10,20]

输出: output/production/r1_next_open_v3_long_lambda_nost/
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

TRAIN_END_DATE = "20250930"
MAX_TRAIN_DATES = 500  # subsample by date (而非随机 sample), 保 group 完整
OUT_NAME = "r1_next_open_v3_long_lambda_nost"
LABEL = "r1_next_open"
EXCLUDE_COLS = {"ts_code", "trade_time", "trade_date",
                  "r4_1h", "r20_1h", "r40_1h",
                  "r1_next_open", "r4_next_morn", "r8_next_day"}


def discretize_label_by_date(df: pd.DataFrame, label_col: str,
                                n_bins: int = 10) -> pd.Series:
    """当日横截面按 r1 pct rank 切 n_bins 等分, 输出 0..n_bins-1.

    method='first' 避免并列值合并; 极端值 (NaN) 留 NaN, 训练时过滤.
    """
    ranks = df.groupby("trade_date")[label_col].rank(pct=True, method="first")
    # floor 到整数 (float64 safe cast 不支持 Int8, 必须先取整)
    bins_float = (ranks * n_bins).clip(0, n_bins - 1)
    bins_int = np.floor(bins_float.fillna(-1).values).astype("int8")
    bins = pd.Series(bins_int, index=df.index, dtype="Int8")
    bins = bins.where(bins >= 0)  # -1 → NaN
    return bins


def main():
    t0 = time.time()
    print(f"\n=== r1 LambdaRank 重训 (cut at {TRAIN_END_DATE}) ===\n", flush=True)
    print(f"输出: {OUT_NAME}", flush=True)

    out_dir = PROD / OUT_NAME
    if (out_dir / "classifier.txt").exists():
        print(f"[{OUT_NAME}] 已存在, 跳过 (删除后重训)"); return

    print(f"加载 factors_v3...", flush=True)
    df = pd.read_parquet(F3)
    df["trade_date"] = df["trade_date"].astype(str)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    print(f"  {len(df):,} × {len(df.columns)}", flush=True)

    # 只保 EOD bar (hour=15) - lambdarank query 上限 10000, 全 hour bar 一天 25K 超限
    # 且实战推理就只在 EOD 一次, 训练 EOD-only 更对齐
    df = df[df["trade_time"].dt.hour == 15].copy()
    print(f"  EOD bar 过滤: {len(df):,}", flush=True)
    # 去重 (factors_v3 同股同日可能有多条 bar, 取最后一条)
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last").reset_index(drop=True)
    print(f"  去重后: {len(df):,}", flush=True)

    # ST 源头排除 (与 nost 一致)
    basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    if basic_p.exists():
        basic = pd.read_parquet(basic_p)[["ts_code", "name"]].drop_duplicates("ts_code")
        st_codes = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
        before = len(df)
        df = df[~df["ts_code"].isin(st_codes)].reset_index(drop=True)
        print(f"  ST 排除: {before - len(df):,} 行 ({len(st_codes)} 只 ST)", flush=True)

    df = df.dropna(subset=[LABEL])
    df = df[df[LABEL].abs() <= 20].copy()

    # 离散化 label (按日 decile)
    print(f"label 离散化 (decile, 0-9, by trade_date)...", flush=True)
    df["r1_decile"] = discretize_label_by_date(df, LABEL, n_bins=10)
    df = df.dropna(subset=["r1_decile"]).copy()
    df["r1_decile"] = df["r1_decile"].astype(int)
    print(f"  decile 分布: {df['r1_decile'].value_counts().sort_index().to_dict()}",
           flush=True)

    train_df = df[df["trade_date"] < TRAIN_END_DATE].copy()
    val = df[df["trade_date"] >= TRAIN_END_DATE].copy()
    print(f"  train < {TRAIN_END_DATE}: {len(train_df):,}", flush=True)
    print(f"  val: {len(val):,}", flush=True)

    # subsample by date (保 group 完整)
    train_dates = sorted(train_df["trade_date"].unique())
    if len(train_dates) > MAX_TRAIN_DATES:
        rng = np.random.RandomState(42)
        keep_dates = set(rng.choice(train_dates, size=MAX_TRAIN_DATES, replace=False))
        train_df = train_df[train_df["trade_date"].isin(keep_dates)].copy()
        print(f"  subsample 日期: {len(train_dates)} → {MAX_TRAIN_DATES} 日, "
               f"样本 {len(train_df):,}", flush=True)

    feat_cols = [c for c in df.columns
                  if c not in EXCLUDE_COLS and c != "r1_decile"
                  and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  特征列: {len(feat_cols)}", flush=True)

    # clip inf/NaN
    for c in feat_cols:
        train_df[c] = train_df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
        val[c] = val[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    # 按 trade_date 排序 (group 连续是 LGBMRanker 必须)
    print(f"  排序 train/val by trade_date (group 连续要求)...", flush=True)
    train_df = train_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    val = val.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    group_train = train_df.groupby("trade_date", sort=False).size().values.tolist()
    group_val = val.groupby("trade_date", sort=False).size().values.tolist()
    print(f"  train groups: {len(group_train)} 日, avg 池 {np.mean(group_train):.0f} 股",
           flush=True)
    print(f"  val groups: {len(group_val)} 日, avg 池 {np.mean(group_val):.0f} 股",
           flush=True)

    X_train = train_df[feat_cols].astype("float32")
    y_train = train_df["r1_decile"].astype("int32")
    X_val = val[feat_cols].astype("float32")
    y_val = val["r1_decile"].astype("int32")
    y_val_continuous = val[LABEL].astype("float32")  # 回测时用

    print(f"\n训练 LambdaRanker (NDCG@10 主指标)...", flush=True)
    clf = lgb.LGBMRanker(
        n_estimators=3000, learning_rate=0.03, num_leaves=63,
        min_child_samples=500, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10, 20],
        label_gain=list(range(10)),  # 0..9 线性增益
    )
    clf.fit(X_train, y_train, group=group_train,
             eval_set=[(X_val, y_val)], eval_group=[group_val],
             eval_metric="ndcg",
             callbacks=[lgb.early_stopping(150, first_metric_only=True),
                          lgb.log_evaluation(100)])

    out_dir.mkdir(exist_ok=True, parents=True)
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": LABEL, "model_type": "lambdarank",
        "label_n_bins": 10, "label_discretization": "by_date_decile_first",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 预测 + 评估
    y_pred = clf.predict(X_val)

    # IC 用连续 r1 (回测可比性), RankIC 用 decile
    ic_continuous = stats.pearsonr(y_pred, y_val_continuous)[0]
    rank_ic_continuous = stats.spearmanr(y_pred, y_val_continuous)[0]
    rank_ic_decile = stats.spearmanr(y_pred, y_val)[0]

    # 日内 Top 10 命中率 (NDCG 替代指标)
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

    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic_val_continuous": float(ic_continuous),
        "rank_ic_val_continuous": float(rank_ic_continuous),
        "rank_ic_val_decile": float(rank_ic_decile),
        "avg_top10_hit_rate": float(avg_top10_hit),
        "n_train": len(train_df), "n_val": len(val),
        "n_val_dates": len(group_val),
        "n_features": len(feat_cols),
        "version": "long_oos_lambdarank", "target": LABEL,
        "train_window": f"< {TRAIN_END_DATE}",
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
        "n_train_dates": len(group_train),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] {OUT_NAME}", flush=True)
    print(f"  best_iter={clf.best_iteration_}", flush=True)
    print(f"  IC (vs 连续 r1): {ic_continuous:.4f}", flush=True)
    print(f"  RankIC (vs 连续): {rank_ic_continuous:.4f}", flush=True)
    print(f"  RankIC (vs decile): {rank_ic_decile:.4f}", flush=True)
    print(f"  日均 Top10 命中: {avg_top10_hit*100:.1f}% (随机 10%)", flush=True)
    print(f"  锚点: {p5:.3f} / {p50:.3f} / {p95:.3f}", flush=True)
    print(f"  耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
