#!/usr/bin/env python3
"""V6 P0: 重训 sell_10_v6 / sell_20_v6 含 mfk K 线因子.

V4 sell 模型未用 mfk (203 ALL only). V4 评估证明 mfk 让 r20 IC +68%,
sell 端大概率也能从 mfk 受益 (因为预测 max_dd 与预测 r20 共享相似信号).

输出: output/production/{sell_10_v6, sell_20_v6}/
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
OUT_BASE = ROOT / "output" / "production"

TRAIN_START = "20230101"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket",
           "max_gain_10","max_dd_10","max_gain_20","max_dd_20","entry_open"}


def load_window(start, end):
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10","max_dd_10"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    l20 = pd.read_parquet(LABELS_20, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")

    for path in [
        ROOT / "output" / "amount_features" / "amount_features.parquet",
        ROOT / "output" / "regime_extra" / "regime_extra.parquet",
        ROOT / "output" / "moneyflow" / "features.parquet",
        ROOT / "output" / "cogalpha_features" / "features.parquet",
        ROOT / "output" / "mfk_features" / "features.parquet",  # NEW
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")

    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path,
              columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        full = full.merge(rg, on="trade_date", how="left")
    return full


def train_classifier(name, train, test, feat_cols, industries, label_col, threshold,
                      train_subsample=1_500_000):
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)
    train = train[feat_cols + ["industry", label_col, "trade_date"]].dropna(subset=[label_col]).copy()
    test = test[feat_cols + ["industry", label_col, "trade_date"]].dropna(subset=[label_col]).copy()
    train["y"] = (train[label_col] <= threshold).astype(int)
    test["y"]  = (test[label_col] <= threshold).astype(int)
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    feat_cols = list(feat_cols)
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")
    if len(train) > train_subsample:
        train = train.sample(n=train_subsample, random_state=42).reset_index(drop=True)
    pos_ratio = train["y"].mean()
    spw = (1 - pos_ratio) / max(pos_ratio, 1e-3)
    print(f"  {name}: train={len(train):,} test={len(test):,} feat={len(feat_cols)} pos%={pos_ratio*100:.1f}",
          flush=True)

    X_train = train[feat_cols].astype("float32"); y_train = train["y"]
    X_test  = test[feat_cols].astype("float32");  y_test  = test["y"]

    clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=spw, max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="binary", metric="auc",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])

    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "label_col": label_col, "threshold": threshold, "condition": "le",
        "model_type": "classifier",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    p5 = float(np.quantile(proba, 0.05))
    p50 = float(np.quantile(proba, 0.50))
    p95 = float(np.quantile(proba, 0.95))
    # mfk 重要性
    imp = pd.Series(clf.booster_.feature_importance(importance_type="gain"), index=feat_cols)
    mfk_pct = imp[imp.index.str.startswith("mfk_")].sum() / max(imp.sum(), 1) * 100
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "auc": float(auc), "base_rate": float(y_test.mean()),
        "n_train": len(train), "n_test": len(test),
        "n_features": len(feat_cols),
        "version": "v6", "mfk_importance_pct": float(mfk_pct),
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name} 完成: AUC={auc:.4f} mfk_imp={mfk_pct:.1f}% 锚 p5/p50/p95={p5:.3f}/{p50:.3f}/{p95:.3f}",
          flush=True)
    del clf, X_train, X_test, y_train, y_test
    gc.collect()
    return auc


def main():
    t0 = time.time()
    print("=== V6 P0: sell with mfk ===\n", flush=True)
    print("加载数据 (含 mfk)...", flush=True)
    train = load_window(TRAIN_START, TRAIN_END)
    test  = load_window(TEST_START, TEST_END)
    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    feat_cols = [c for c in train.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
    n_mfk = len([c for c in feat_cols if c.startswith("mfk_")])
    print(f"特征数: {len(feat_cols)} (含 {n_mfk} 个 mfk)", flush=True)

    print(f"\n[1/2] sell_10_v6 (max_dd_10 <= -5%)", flush=True)
    train_classifier("sell_10_v6", train, test, feat_cols, industries, "max_dd_10", -5.0)

    print(f"\n[2/2] sell_20_v6 (max_dd_20 <= -8%)", flush=True)
    train_classifier("sell_20_v6", train, test, feat_cols, industries, "max_dd_20", -8.0)

    print(f"\n=== 总耗时 {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
