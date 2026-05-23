"""1H R20 LGBM v2 训练 - 用 v2 因子 (53 个: 23 v1 + 30 v2 独有).

输入: output/1h_factors/factors_v2.parquet
输出: output/production/r20_1h_v2/
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "1h_factors" / "factors_v2.parquet"
OUT_BASE = ROOT / "output" / "production"

TRAIN_END_DATE = "20260228"


def spearman_ic(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    return "spearman_ic", ic if not np.isnan(ic) else 0.0, True


def main():
    t0 = time.time()
    print(f"加载 1H v2 因子...")
    df = pd.read_parquet(SRC)
    print(f"  {len(df):,} 行 × {len(df.columns)} 列")
    print(f"  时间: {df['trade_time'].min()} → {df['trade_time'].max()}")

    # ST 源头排除 (2026-05-21 起强制)
    basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    if basic_p.exists():
        basic = pd.read_parquet(basic_p)[["ts_code", "name"]].drop_duplicates("ts_code")
        st_codes = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
        before = len(df)
        df = df[~df["ts_code"].isin(st_codes)].reset_index(drop=True)
        print(f"  ST 排除: {before - len(df):,} 行 ({len(st_codes)} 只 ST)", flush=True)

    df = df.dropna(subset=["r20_1h"])
    n0 = len(df)
    df = df[df["r20_1h"].abs() <= 50].copy()
    print(f"  有 r20_1h label: {n0:,} → 去除 |label|>50%: {len(df):,} ({(n0-len(df))/n0*100:.2f}% 删)")

    df["trade_date"] = df["trade_date"].astype(str)
    train = df[df["trade_date"] < TRAIN_END_DATE].copy()
    val = df[df["trade_date"] >= TRAIN_END_DATE].copy()
    print(f"  train: {len(train):,} (< {TRAIN_END_DATE})")
    print(f"  val:   {len(val):,} (≥ {TRAIN_END_DATE})")

    EXCLUDE = {"ts_code", "trade_time", "trade_date", "r4_1h", "r20_1h", "r40_1h"}
    feat_cols = [c for c in df.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  特征列: {len(feat_cols)}")

    # 替换 inf -> nan + clip
    for c in feat_cols:
        train[c] = train[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
        val[c] = val[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    if len(train) > 2_500_000:
        train = train.sample(n=2_500_000, random_state=42).reset_index(drop=True)
        print(f"  subsample 训练集到 250 万行")

    X_train = train[feat_cols].astype("float32"); y_train = train["r20_1h"].astype("float32")
    X_val = val[feat_cols].astype("float32"); y_val = val["r20_1h"].astype("float32")

    print(f"\n训练 r20_1h_v2 模型...", flush=True)
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
                          lgb.log_evaluation(50)])

    out_dir = OUT_BASE / "r20_1h_v2_nost"   # 2026-05-21 ST 排除重训
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
        "version": "v2", "target": "r20_1h",
        "train_window": f"< {TRAIN_END_DATE}",
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[r20_1h_v2] IC={ic:.4f} RankIC={rank_ic:.4f} best_iter={clf.best_iteration_}")
    print(f"  锚点: {p5:.2f} / {p50:.2f} / {p95:.2f}")

    # 特征重要性 top 20
    imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": clf.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False).head(20)
    print(f"\nTop 20 重要特征 (gain):")
    for _, r in imp.iterrows():
        print(f"  {r['feature']:30s} {r['importance']:>10.0f}")
    imp.to_csv(out_dir / "feature_importance_top20.csv", index=False, encoding="utf-8-sig")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
