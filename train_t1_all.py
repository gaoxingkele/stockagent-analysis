"""Sprint 4.6 - 训练 3 个 T+1 真实可执行 label 模型.

- r1_next_open: 次日开盘 (最短 T+1)
- r4_next_morn: 次日 11:30
- r8_next_day:  次日 15:00

输入: output/1h_factors/factors_v3.parquet
输出: output/production/r1_next_open_v3/, r4_next_morn_v3/, r8_next_day_v3/
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
OUT_BASE = ROOT / "output" / "production"

TRAIN_END_DATE = "20260228"
LABELS = [
    # 2026-05-21: ST 排除重训, _nost 保留对照
    ("r1_next_open", "r1_next_open_v3_nost"),
    ("r4_next_morn", "r4_next_morn_v3_nost"),
    ("r8_next_day",  "r8_next_day_v3_nost"),
]


def spearman_ic(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    return "spearman_ic", ic if not np.isnan(ic) else 0.0, True


def train_one(df: pd.DataFrame, feat_cols, label_col, out_name):
    out_dir = OUT_BASE / out_name
    out_dir.mkdir(exist_ok=True, parents=True)

    sub = df.dropna(subset=[label_col])
    n0 = len(sub)
    sub = sub[sub[label_col].abs() <= 20]  # T+1 label 截断 ±20%
    print(f"\n  [{label_col}] {n0:,} → 去除 |label|>20%: {len(sub):,}", flush=True)

    train = sub[sub["trade_date"] < TRAIN_END_DATE]
    val = sub[sub["trade_date"] >= TRAIN_END_DATE]
    print(f"    train={len(train):,}  val={len(val):,}", flush=True)

    if len(train) > 2_500_000:
        train = train.sample(n=2_500_000, random_state=42).reset_index(drop=True)
        print(f"    subsample to 250 万", flush=True)

    X_train = train[feat_cols].astype("float32"); y_train = train[label_col].astype("float32")
    X_val = val[feat_cols].astype("float32"); y_val = val[label_col].astype("float32")

    clf = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.03, num_leaves=63,
        min_child_samples=500, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             eval_metric=spearman_ic,
             callbacks=[lgb.early_stopping(150, first_metric_only=True),
                          lgb.log_evaluation(100)])

    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": label_col, "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    y_pred = clf.predict(X_val)
    ic = stats.pearsonr(y_pred, y_val)[0]
    rank_ic = stats.spearmanr(y_pred, y_val)[0]
    p5 = float(np.quantile(y_pred, 0.05))
    p50 = float(np.quantile(y_pred, 0.50))
    p95 = float(np.quantile(y_pred, 0.95))

    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic_val": float(ic), "rank_ic_val": float(rank_ic),
        "n_train": len(train), "n_val": len(val),
        "n_features": len(feat_cols),
        "version": "v3", "target": label_col,
        "train_window": f"< {TRAIN_END_DATE}",
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [OK][{label_col}] IC={ic:.4f} RankIC={rank_ic:.4f} best_iter={clf.best_iteration_}", flush=True)
    print(f"       锚点 {p5:.3f} / {p50:.3f} / {p95:.3f}", flush=True)

    # 释放内存
    del clf, X_train, y_train, X_val, y_val, train, val, sub
    gc.collect()


def main():
    t0 = time.time()
    print(f"加载 v3 因子...")
    df = pd.read_parquet(SRC)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"  {len(df):,} × {len(df.columns)}", flush=True)

    # ST 源头排除 (2026-05-21 起强制)
    basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    if basic_p.exists():
        basic = pd.read_parquet(basic_p)[["ts_code", "name"]].drop_duplicates("ts_code")
        st_codes = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
        before = len(df)
        df = df[~df["ts_code"].isin(st_codes)].reset_index(drop=True)
        print(f"  ST 排除: {before - len(df):,} 行 ({len(st_codes)} 只 ST)", flush=True)

    EXCLUDE = {"ts_code", "trade_time", "trade_date",
                 "r4_1h", "r20_1h", "r40_1h",
                 "r1_next_open", "r4_next_morn", "r8_next_day"}
    feat_cols = [c for c in df.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  特征列: {len(feat_cols)}", flush=True)

    # 全局 clip 一次 (避免重复)
    for c in feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    print(f"  clip 完成 {time.time()-t0:.0f}s", flush=True)

    for label, name in LABELS:
        out_dir = OUT_BASE / name
        if (out_dir / "classifier.txt").exists() and (out_dir / "meta.json").exists():
            print(f"\n  [{label}] [OK]已存在 {out_dir.name}/, 跳过", flush=True)
            continue
        train_one(df, feat_cols, label, name)

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
