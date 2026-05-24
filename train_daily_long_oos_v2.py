"""V12 v9 (OOS 扩窗) - 日线 R5/R10/R20 重训, cut at 20260131.

vs train_daily_long_oos.py:
  - TRAIN_END: 20250930 → 20260131 (扩 4 个月)
  - DATA_END: 20260420 → 20260522 (最新数据)
  - OUT 加 _v2 后缀

输出: r5_v17_long_nost_v2, r10_v16_long_nost_v2, r20_v16_long_nost_v2
"""
from __future__ import annotations
import gc, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window, spearman_ic, EXCLUDE

OUT_BASE = ROOT / "output" / "production"

TRAIN_START = "20230101"
TRAIN_END   = "20260131"   # v9: 扩窗 4 个月
DATA_END    = "20260522"   # 用最新 daily


def train_one(name, df, feat_cols, industries, y_col):
    out_dir = OUT_BASE / name
    if (out_dir / "classifier.txt").exists():
        print(f"[{name}] 已存, 跳过"); return
    out_dir.mkdir(exist_ok=True, parents=True)
    df = df[feat_cols + ["industry", y_col, "trade_date"]].dropna(subset=[y_col]).copy()
    df["industry_id"] = pd.Categorical(df["industry"].fillna("unknown"),
                                         categories=industries.categories).codes
    if "industry_id" not in feat_cols: feat_cols = list(feat_cols) + ["industry_id"]

    train_df = df[df["trade_date"] < TRAIN_END]
    val = df[df["trade_date"] >= TRAIN_END]
    print(f"  [{name}] train={len(train_df):,} val={len(val):,} feat={len(feat_cols)}", flush=True)

    if len(train_df) > 1_800_000:
        train_df = train_df.sample(n=1_800_000, random_state=42).reset_index(drop=True)
        print(f"  subsample 训练到 180 万", flush=True)

    X_train = train_df[feat_cols].astype("float32"); y_train = train_df[y_col].astype("float32")
    X_val = val[feat_cols].astype("float32"); y_val = val[y_col].astype("float32")

    clf = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.04, num_leaves=63,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             eval_metric=spearman_ic,
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(80, first_metric_only=True),
                          lgb.log_evaluation(100)])

    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.categories,
                                                    range(len(industries.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": y_col, "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    y_pred = clf.predict(X_val)
    ic = stats.pearsonr(y_pred, y_val)[0]
    rank_ic = stats.spearmanr(y_pred, y_val)[0]
    p5, p50, p95 = np.quantile(y_pred, [0.05, 0.5, 0.95])
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic_val": float(ic), "rank_ic_val": float(rank_ic),
        "n_train": len(train_df), "n_val": len(val),
        "n_features": len(feat_cols),
        "version": "long_oos_v2", "target": y_col,
        "train_window": f"< {TRAIN_END}",
        "anchor_p5": float(p5), "anchor_p50": float(p50), "anchor_p95": float(p95),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {name} IC={ic:.4f} RankIC={rank_ic:.4f} 锚: {p5:.2f}/{p50:.2f}/{p95:.2f}",
           flush=True)
    del clf, X_train, y_train, X_val, y_val
    gc.collect()


def main():
    t0 = time.time()
    print(f"=== 日线 R5/R10/R20 OOS 扩窗重训 v2 (cut at {TRAIN_END}) ===\n")

    print("加载日线 factor + labels...", flush=True)
    df = load_window(TRAIN_START, DATA_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"加载完成: {len(df):,}", flush=True)

    if "r5" not in df.columns:
        print("  r5 column 缺失, 从 daily cache 现算...", flush=True)
        daily_dir = ROOT / "output" / "tushare_cache" / "daily"
        files = sorted(daily_dir.glob("*.parquet"))
        print(f"  加载 daily cache {len(files)} 个文件...", flush=True)
        dailies = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"])
                     for f in files]
        big = pd.concat(dailies, ignore_index=True)
        big["trade_date"] = big["trade_date"].astype(str)
        big = big.sort_values(["ts_code", "trade_date"])
        big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
        big["close_5d"] = big.groupby("ts_code")["close"].shift(-5)
        big["r5"] = (big["close_5d"] / big["next_open"] - 1) * 100
        r5_df = big[["ts_code", "trade_date", "r5"]].dropna()
        df = df.merge(r5_df, on=["ts_code", "trade_date"], how="left")
        print(f"  r5 加成功", flush=True)

    industries = pd.Categorical(df["industry"].fillna("unknown"))
    feat_cols = [c for c in df.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    print(f"基础特征列: {len(feat_cols)}", flush=True)

    train_one("r5_v17_long_nost_v2", df, feat_cols, industries, "r5")
    train_one("r20_v16_long_nost_v2", df, feat_cols, industries, "r20")
    if "r10" in df.columns:
        train_one("r10_v16_long_nost_v2", df, feat_cols, industries, "r10")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
