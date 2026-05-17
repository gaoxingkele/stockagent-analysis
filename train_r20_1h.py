"""1H R20 LGBM 模型训练 (用户核心创新完整版).

label: r20_1h (5 交易日 forward, 20 个 1H bar)
特征: 1H 维度 30 个因子 (compute_1h_factors.py 输出)

训练区间: 2024-01-01 → 2026-02-28 (留 0301-0420 为 OOS, 约 60 日 = 240 bar)
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "1h_factors" / "factors.parquet"
OUT_BASE = ROOT / "output" / "production"

TRAIN_END_DATE = "20260228"  # 时序 split (作 date 比较)


def spearman_ic(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    return "spearman_ic", ic if not np.isnan(ic) else 0.0, True


def main():
    t0 = time.time()
    print(f"加载 1H 因子...")
    df = pd.read_parquet(SRC)
    print(f"  {len(df):,} 行 × {len(df.columns)} 列")
    print(f"  时间: {df['trade_time'].min()} → {df['trade_time'].max()}")

    # label = r20_1h (5 日 forward), 去除除权异常 |label| > 50%
    df = df.dropna(subset=["r20_1h"])
    n0 = len(df)
    df = df[df["r20_1h"].abs() <= 50].copy()
    print(f"  有 r20_1h label: {n0:,} → 去除 |label|>50%: {len(df):,} ({(n0-len(df))/n0*100:.2f}% 删)")

    # 时序 split
    df["trade_date"] = df["trade_date"].astype(str)
    train = df[df["trade_date"] < TRAIN_END_DATE].copy()
    val = df[df["trade_date"] >= TRAIN_END_DATE].copy()
    print(f"  train: {len(train):,} (< {TRAIN_END_DATE})")
    print(f"  val:   {len(val):,} (≥ {TRAIN_END_DATE})")

    # 特征列: 排除元数据 + label + 原始价格/量 (跨股不可比)
    EXCLUDE = {"ts_code", "trade_time", "trade_date", "close", "r4_1h", "r20_1h", "r40_1h",
                 "ma5", "ma10", "ma20", "ma60", "vol_ma20"}
    feat_cols = [c for c in df.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  特征列: {len(feat_cols)}")

    # 特征 winsorize (clip 到 [-200, +200], 避免极端值污染 LGBM split)
    for c in feat_cols:
        train[c] = train[c].clip(-200, 200)
        val[c] = val[c].clip(-200, 200)

    # subsample (内存控制)
    if len(train) > 2_000_000:
        train = train.sample(n=2_000_000, random_state=42).reset_index(drop=True)
        print(f"  subsample 训练集到 200 万行")

    X_train = train[feat_cols].astype("float32"); y_train = train["r20_1h"].astype("float32")
    X_val = val[feat_cols].astype("float32"); y_val = val["r20_1h"].astype("float32")

    print(f"\n训练 r20_1h_v1 模型...", flush=True)
    clf = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        min_child_samples=500, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             eval_metric=spearman_ic,
             callbacks=[lgb.early_stopping(100, first_metric_only=True),
                          lgb.log_evaluation(0)])

    out_dir = OUT_BASE / "r20_1h_v1"
    out_dir.mkdir(exist_ok=True)
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": "r20_1h", "model_type": "regressor",
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
        "version": "v1", "target": "r20_1h",
        "train_window": f"< {TRAIN_END_DATE}",
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[r20_1h_v1] IC={ic:.4f} RankIC={rank_ic:.4f} best_iter={clf.best_iteration_}")
    print(f"  锚点: {p5:.2f} / {p50:.2f} / {p95:.2f}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
