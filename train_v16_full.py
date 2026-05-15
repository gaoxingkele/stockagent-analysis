"""V16 全量重训: 用所有可用 labels (到 20260413) 训练.

变更 vs V15:
  - TRAIN 区间扩到 20260413 (V15 是 20260213, 多 60 日训练数据)
  - 用 LightGBM 内部 valid_set (10% random split) 做 early stopping
  - 无外部 OOS, 适应"用全部可用数据生产模型"场景

输出: output/production/{r10_v16_all, r20_v16_all}/
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split

# 复用 V15 的 load_window
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v15_refresh import load_window, spearman_ic, EXCLUDE

ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "output" / "production"

TRAIN_START = "20230101"
TRAIN_END   = "20260413"  # labels 极限末日


def train_regressor_full(name, df, feat_cols, industries, y_col,
                          train_subsample=1_800_000):
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)
    df = df[feat_cols + ["industry", y_col, "trade_date"]].dropna(subset=[y_col]).copy()
    df["industry_id"] = pd.Categorical(df["industry"].fillna("unknown"),
                                         categories=industries.categories).codes
    feat_cols = list(feat_cols)
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")
    if len(df) > train_subsample:
        df = df.sample(n=train_subsample, random_state=42).reset_index(drop=True)

    # 内部 90/10 split for early stopping
    train, val = train_test_split(df, test_size=0.1, random_state=42)
    print(f"  [{name}] train={len(train):,} val={len(val):,} feat={len(feat_cols)}", flush=True)

    X_train = train[feat_cols].astype("float32"); y_train = train[y_col].astype("float32")
    X_val  = val[feat_cols].astype("float32");  y_val  = val[y_col].astype("float32")

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
                          lgb.log_evaluation(0)])
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
    p5 = float(np.quantile(y_pred, 0.05)); p50 = float(np.quantile(y_pred, 0.50)); p95 = float(np.quantile(y_pred, 0.95))
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic_val": float(ic), "rank_ic_val": float(rank_ic),
        "n_train": len(train), "n_val": len(val),
        "n_features": len(feat_cols),
        "version": "v16",
        "train_window": f"{TRAIN_START}-{TRAIN_END}",
        "split": "internal_90/10",
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{name}] val IC={ic:.4f} RankIC={rank_ic:.4f} 锚={p5:.2f}/{p50:.2f}/{p95:.2f}", flush=True)
    del clf, X_train, X_val, y_train, y_val
    gc.collect()


def main():
    t0 = time.time()
    print(f"=== V16 全量重训 ({TRAIN_START} → {TRAIN_END}, 含 V15 全部修复) ===\n", flush=True)

    print("[1/2] 加载训练集 + mfk (TRAIN: 20230101-20260413)...", flush=True)
    train_df = load_window(TRAIN_START, TRAIN_END, with_mfk=True)
    print(f"  训练集: {len(train_df):,} 行\n", flush=True)

    industries = pd.Categorical(
        train_df["industry"].fillna("unknown").astype(str),
        categories=sorted(train_df["industry"].fillna("unknown").astype(str).unique())
    )
    feat_cols = [c for c in train_df.columns
                  if c not in EXCLUDE
                  and pd.api.types.is_numeric_dtype(train_df[c])]
    print(f"特征列数: {len(feat_cols)}\n", flush=True)

    print(f"[2/2] 训练 r10/r20 模型 (内部 90/10 val + early stop)...\n", flush=True)
    train_regressor_full("r10_v16_all", train_df, feat_cols, industries, "r10")
    train_regressor_full("r20_v16_all", train_df, feat_cols, industries, "r20")

    print(f"\n=== 完成, 总耗时 {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
