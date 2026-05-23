"""日线 R5/R10/R20 模型 LambdaRank + 长期收益偏置重训.

vs train_daily_long_oos.py:
  - objective: regression → lambdarank
  - label: 连续 r5/r10/r20 → 当日横截面 decile (0-9)
  - merge output/long_return_features/features.parquet (11 偏置因子)
  - 训练数据按 trade_date 排序, group_sizes 传给 LGBM
  - subsample 改为按 trade_date 整组采样

输出: r5/r10/r20_v*_long_lambda_bias_nost
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
from train_v15_refresh import load_window, EXCLUDE

OUT_BASE = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

TRAIN_START = "20230101"
TRAIN_END   = "20250930"
DATA_END    = "20260420"
MAX_TRAIN_DATES = 500


def discretize_label_by_date(df: pd.DataFrame, label_col: str, n_bins: int = 10) -> pd.Series:
    ranks = df.groupby("trade_date")[label_col].rank(pct=True, method="first")
    bins_float = (ranks * n_bins).clip(0, n_bins - 1)
    bins_int = np.floor(bins_float.fillna(-1).values).astype("int8")
    bins = pd.Series(bins_int, index=df.index, dtype="Int8")
    return bins.where(bins >= 0)


def train_one(name: str, df: pd.DataFrame, feat_cols: list, industries: pd.Categorical,
                y_col: str):
    out_dir = OUT_BASE / name
    if (out_dir / "classifier.txt").exists():
        print(f"[{name}] 已存, 跳过"); return
    out_dir.mkdir(exist_ok=True, parents=True)

    sub = df[feat_cols + ["industry", y_col, "trade_date", "ts_code"]].dropna(subset=[y_col]).copy()
    sub["industry_id"] = pd.Categorical(sub["industry"].fillna("unknown"),
                                          categories=industries.categories).codes
    if "industry_id" not in feat_cols: feat_cols = list(feat_cols) + ["industry_id"]

    # decile label
    print(f"  [{name}] label decile ...", flush=True)
    sub[f"{y_col}_decile"] = discretize_label_by_date(sub, y_col, n_bins=10)
    sub = sub.dropna(subset=[f"{y_col}_decile"]).copy()
    sub[f"{y_col}_decile"] = sub[f"{y_col}_decile"].astype(int)

    train_df = sub[sub["trade_date"] < TRAIN_END].copy()
    val = sub[sub["trade_date"] >= TRAIN_END].copy()
    print(f"  [{name}] train={len(train_df):,} val={len(val):,} feat={len(feat_cols)}", flush=True)

    # subsample by date (保 group)
    train_dates = sorted(train_df["trade_date"].unique())
    if len(train_dates) > MAX_TRAIN_DATES:
        rng = np.random.RandomState(42)
        keep = set(rng.choice(train_dates, size=MAX_TRAIN_DATES, replace=False))
        train_df = train_df[train_df["trade_date"].isin(keep)].copy()
        print(f"  subsample → {MAX_TRAIN_DATES} 日, {len(train_df):,} 样本", flush=True)

    # sort 保 group
    train_df = train_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    val = val.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    group_train = train_df.groupby("trade_date", sort=False).size().values.tolist()
    group_val = val.groupby("trade_date", sort=False).size().values.tolist()
    print(f"  groups train={len(group_train)} val={len(group_val)} "
           f"(avg pool {np.mean(group_train):.0f} / {np.mean(group_val):.0f})", flush=True)

    X_train = train_df[feat_cols].astype("float32")
    y_train = train_df[f"{y_col}_decile"].astype("int32")
    X_val = val[feat_cols].astype("float32")
    y_val = val[f"{y_col}_decile"].astype("int32")
    y_val_cont = val[y_col].astype("float32")

    clf = lgb.LGBMRanker(
        n_estimators=3000, learning_rate=0.03, num_leaves=63,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="lambdarank", metric="ndcg",
        ndcg_eval_at=[5, 10, 20],
        label_gain=list(range(10)),
    )
    clf.fit(X_train, y_train, group=group_train,
             eval_set=[(X_val, y_val)], eval_group=[group_val],
             eval_metric="ndcg",
             callbacks=[lgb.early_stopping(150, first_metric_only=True),
                          lgb.log_evaluation(100)])

    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.categories,
                                                    range(len(industries.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": y_col, "model_type": "lambdarank_bias",
        "label_n_bins": 10,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    y_pred = clf.predict(X_val)
    ic = stats.pearsonr(y_pred, y_val_cont)[0]
    rank_ic = stats.spearmanr(y_pred, y_val_cont)[0]
    p5, p50, p95 = np.quantile(y_pred, [0.05, 0.5, 0.95])

    # feature importance: 偏置因子排名
    feat_imp = clf.booster_.feature_importance(importance_type="gain")
    rank_list = sorted(zip(feat_cols, feat_imp), key=lambda x: -x[1])
    bias_cols = [c for c in feat_cols if c in {
        "long_return_252d", "long_return_504d", "industry_return_252d",
        "industry_return_504d", "relative_strength_252d", "relative_strength_504d",
        "long_return_252d_decile", "long_return_504d_decile",
        "industry_return_504d_decile", "rs_in_decile", "concept_return_504d_mean",
    }]
    bias_in_rank = [(c, int(imp), i+1) for i, (c, imp) in enumerate(rank_list) if c in bias_cols]

    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic_val": float(ic), "rank_ic_val": float(rank_ic),
        "n_train": len(train_df), "n_val": len(val),
        "n_features": len(feat_cols),
        "version": "long_oos_lambda_bias", "target": y_col,
        "train_window": f"< {TRAIN_END}",
        "anchor_p5": float(p5), "anchor_p50": float(p50), "anchor_p95": float(p95),
        "bias_features_rank": bias_in_rank,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [OK] {name} IC={ic:.4f} RankIC={rank_ic:.4f} best_iter={clf.best_iteration_}",
           flush=True)
    print(f"      anchor: {p5:.3f}/{p50:.3f}/{p95:.3f}", flush=True)
    print(f"      bias top: {bias_in_rank[:3]}", flush=True)
    del clf, X_train, y_train, X_val, y_val, sub, train_df, val
    gc.collect()


def main():
    t0 = time.time()
    print(f"=== 日线 R5/R10/R20 LambdaRank + Bias 重训 (cut at {TRAIN_END}) ===\n",
           flush=True)

    print("加载日线 factor + labels (load_window, ST 已排除)...", flush=True)
    df = load_window(TRAIN_START, DATA_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"加载完成: {len(df):,}", flush=True)

    # r5 from daily cache (跟 train_daily_long_oos 一致)
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

    # merge long_return_features
    print(f"\nmerge long_return_features...", flush=True)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        df = df.merge(lf, on=["ts_code", "trade_date"], how="left")
        new_cols = [c for c in lf.columns if c not in ("ts_code", "trade_date")]
        print(f"  merged, 新增 {len(new_cols)} 列", flush=True)
        for c in new_cols[:3]:
            cov = df[c].notna().sum() / len(df) * 100
            print(f"    {c}: {cov:.1f}% 覆盖", flush=True)
    else:
        print(f"!! {LONG_FEAT_P} 不存在, 跳过偏置", flush=True)
        new_cols = []

    industries = pd.Categorical(df["industry"].fillna("unknown"))

    # 特征列 (EXCLUDE + decile 暂留 numeric)
    feat_cols = [c for c in df.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    # decile 转 float (numeric, 不做 categorical)
    for c in new_cols:
        if c.endswith("_decile") and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    print(f"\n基础特征列: {len(feat_cols)}", flush=True)

    # 训三模型
    train_one("r5_v17_long_lambda_bias_nost", df, feat_cols, industries, "r5")
    train_one("r10_v16_long_lambda_bias_nost", df, feat_cols, industries, "r10")
    train_one("r20_v16_long_lambda_bias_nost", df, feat_cols, industries, "r20")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
