"""V17 r5 短期模型 (Sprint 4.1, V12.16).

新增 5 日预测模型, 与现有 r10_v16 / r20_v16 形成三层嵌套.

label: r5 = close[t+5] / open[t+1] - 1 (5 个交易日 forward return)
        从 daily cache 现算 (不依赖 labels parquet)

训练区间: 20230101 → 20260420 (r5 需要 forward 5 日, 末日 0420 + 5 ≈ 0428 有数据)

输出: output/production/r5_v17_all/
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v15_refresh import load_window, spearman_ic, EXCLUDE

ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "output" / "production"

TRAIN_START = "20230101"
TRAIN_END   = "20260420"   # labels: r5 forward 5 日, 0420 + 5 ≈ 0428


def compute_r5_labels(daily_cache_dir: Path, start: str, end: str) -> pd.DataFrame:
    """从 daily cache 计算 r5 forward labels."""
    print(f"  [r5 labels] 加载 daily cache {start} → ...", flush=True)
    files = sorted(daily_cache_dir.glob("*.parquet"))
    # 需要 forward 5 日, 所以截止 end + 10 日左右
    parts = []
    for f in files:
        if f.stem >= start:   # daily 已经按日期保存
            parts.append(pd.read_parquet(f)[["ts_code", "trade_date", "open", "close"]])
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    rows = []
    for ts, g in big.groupby("ts_code"):
        g = g.reset_index(drop=True)
        n = len(g)
        if n < 7: continue
        opens = g["open"].values
        closes = g["close"].values
        dates = g["trade_date"].values
        for idx in range(n - 5):
            td = dates[idx]
            if td > end: break
            entry_open = opens[idx + 1] if idx + 1 < n else None
            close_5 = closes[idx + 5] if idx + 5 < n else None
            if entry_open is None or close_5 is None or entry_open <= 0: continue
            r5 = (close_5 / entry_open - 1) * 100
            rows.append({"ts_code": ts, "trade_date": td, "r5": round(r5, 4)})

    df = pd.DataFrame(rows)
    print(f"  [r5 labels] 完成: {len(df):,} 行, 末日 {df['trade_date'].max()}", flush=True)
    return df


def train_r5_regressor(train_df, val_df, feat_cols, industries,
                       train_subsample=1_800_000):
    name = "r5_v17_all"
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)

    for d in (train_df, val_df):
        d["industry_id"] = pd.Categorical(d["industry"].fillna("unknown"),
                                            categories=industries.categories).codes
    feat_cols = list(feat_cols)
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")

    if len(train_df) > train_subsample:
        train_df = train_df.sample(n=train_subsample, random_state=42).reset_index(drop=True)
    print(f"  [{name}] train={len(train_df):,} val={len(val_df):,} feat={len(feat_cols)}", flush=True)

    X_train = train_df[feat_cols].astype("float32"); y_train = train_df["r5"].astype("float32")
    X_val = val_df[feat_cols].astype("float32"); y_val = val_df["r5"].astype("float32")

    # r5 短期模型, IC 较弱, 用更稳健的参数 (低 lr + 强 early_stop patience)
    clf = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.015, num_leaves=31,
        min_child_samples=500, feature_fraction=0.6,
        bagging_fraction=0.7, bagging_freq=5,
        reg_alpha=0.2, reg_lambda=0.2,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             eval_metric=spearman_ic,
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(200, first_metric_only=True),
                          lgb.log_evaluation(0)])
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.categories,
                                                    range(len(industries.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": "r5", "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    y_pred = clf.predict(X_val)
    ic = stats.pearsonr(y_pred, y_val)[0]
    rank_ic = stats.spearmanr(y_pred, y_val)[0]
    p5 = float(np.quantile(y_pred, 0.05)); p50 = float(np.quantile(y_pred, 0.50)); p95 = float(np.quantile(y_pred, 0.95))
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic_val": float(ic), "rank_ic_val": float(rank_ic),
        "n_train": len(train_df), "n_val": len(val_df),
        "n_features": len(feat_cols),
        "version": "v17",
        "target": "r5",
        "train_window": f"{TRAIN_START}-{TRAIN_END}",
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{name}] val IC={ic:.4f} RankIC={rank_ic:.4f} 锚={p5:.2f}/{p50:.2f}/{p95:.2f}", flush=True)
    return clf


def main():
    t0 = time.time()
    print(f"=== V17 r5 模型重训 ({TRAIN_START} → {TRAIN_END}) ===\n", flush=True)

    # 1. 加载因子矩阵 (复用 V15 的 load_window)
    print("[1/3] 加载因子矩阵 + mfk...", flush=True)
    factor_df = load_window(TRAIN_START, TRAIN_END, with_mfk=True)
    # 清空旧 r/max 标签列, 后面会 merge r5
    drop_cols = [c for c in factor_df.columns
                  if c in ("r5", "r10", "r20", "max_gain_20", "max_dd_20",
                          "max_gain_10", "max_dd_10")]
    if drop_cols:
        factor_df = factor_df.drop(columns=drop_cols)
    print(f"  因子矩阵: {len(factor_df):,} 行 × {len(factor_df.columns)} 列\n", flush=True)

    # 2. 计算 r5 labels + merge
    print("[2/3] 计算 r5 forward labels...", flush=True)
    r5_df = compute_r5_labels(
        ROOT / "output" / "tushare_cache" / "daily",
        TRAIN_START, TRAIN_END,
    )
    df = factor_df.merge(r5_df, on=["ts_code", "trade_date"], how="left")
    df = df.dropna(subset=["r5"])
    print(f"  merge 后: {len(df):,} 行 (有 r5 标签)\n", flush=True)

    # 3. 时序 split + 训练
    print("[3/3] 时序 split + 训练 r5_v17_all...", flush=True)
    df = df.sort_values("trade_date")
    unique_dates = sorted(df["trade_date"].unique())
    cutoff = unique_dates[int(len(unique_dates) * 0.9)]
    train_df = df[df["trade_date"] < cutoff].copy()
    val_df = df[df["trade_date"] >= cutoff].copy()
    print(f"  时序 split: train_end < {cutoff}, val_start ≥ {cutoff}", flush=True)
    print(f"  train n={len(train_df):,}, val n={len(val_df):,}", flush=True)

    industries = pd.Categorical(
        df["industry"].fillna("unknown").astype(str),
        categories=sorted(df["industry"].fillna("unknown").astype(str).unique())
    )
    feat_cols = [c for c in df.columns
                  if c not in EXCLUDE and c != "r5"
                  and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  特征列数: {len(feat_cols)}\n", flush=True)
    train_r5_regressor(train_df, val_df, feat_cols, industries)

    print(f"\n=== 完成, 总耗时 {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
