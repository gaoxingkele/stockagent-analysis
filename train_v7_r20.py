#!/usr/bin/env python3
"""V7 P0: 训练 r20_v7_all (ALL+mfk+pyramid_v2 = 226 feat).

Hypothesis: 加入 6 个 pyramid_v2 因子可能让 r20 IC/RankICIR 进一步提升.
之前 V4 r20_v4_all 用 ALL+17 mfk = 220 feat, IC=0.031, RankICIR=0.81.
V7 加 6 pyramid 后预期 IC ~0.035-0.04, RankICIR 可能突破 0.85.

输出: output/production/r20_v7_all/
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
OUT_DIR = ROOT / "output" / "production" / "r20_v7_all"
OUT_DIR.mkdir(exist_ok=True, parents=True)

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
    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10"])
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
        ROOT / "output" / "mfk_features" / "features.parquet",
        ROOT / "output" / "pyramid_v2" / "features.parquet",  # NEW
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


def spearman_ic(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    return "spearman_ic", ic if not np.isnan(ic) else 0.0, True


def main():
    t0 = time.time()
    print("=== V7 P0: r20_v7_all (ALL+mfk+pyramid_v2) ===\n", flush=True)
    train = load_window(TRAIN_START, TRAIN_END)
    test  = load_window(TEST_START, TEST_END)
    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    feat_cols = [c for c in train.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
    n_mfk = len([c for c in feat_cols if c.startswith("mfk_")])
    n_pyr = len([c for c in feat_cols if c.startswith("pyr_")])
    print(f"特征数: {len(feat_cols)} (mfk {n_mfk}, pyramid_v2 {n_pyr})", flush=True)

    train = train[feat_cols + ["industry", "r20", "trade_date"]].dropna(subset=["r20"]).copy()
    test = test[feat_cols + ["industry", "r20", "trade_date"]].dropna(subset=["r20"]).copy()
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    feat_cols.append("industry_id")
    if len(train) > 1_500_000:
        train = train.sample(n=1_500_000, random_state=42).reset_index(drop=True)
    print(f"  train={len(train):,} test={len(test):,} feat={len(feat_cols)}", flush=True)

    X_train = train[feat_cols].astype("float32"); y_train = train["r20"].astype("float32")
    X_test  = test[feat_cols].astype("float32");  y_test  = test["r20"].astype("float32")

    clf = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             eval_metric=spearman_ic,
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(80, first_metric_only=True),
                          lgb.log_evaluation(0)])

    clf.booster_.save_model(str(OUT_DIR / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(OUT_DIR / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": "r20", "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    y_pred = clf.predict(X_test)
    ic = stats.pearsonr(y_pred, y_test)[0]
    rank_ic = stats.spearmanr(y_pred, y_test)[0]
    monthly = pd.DataFrame({"y": y_test.values, "p": y_pred,
                              "ym": test["trade_date"].astype(str).str[:6].values})
    m_ic = monthly.groupby("ym").apply(
        lambda g: stats.spearmanr(g["p"], g["y"])[0] if len(g) >= 30 else np.nan
    ).dropna()
    rank_ic_ir = m_ic.mean() / m_ic.std() if m_ic.std() > 0 else 0
    p5_pred = float(np.quantile(y_pred, 0.05))
    p50_pred = float(np.quantile(y_pred, 0.50))
    p95_pred = float(np.quantile(y_pred, 0.95))

    imp = pd.Series(clf.booster_.feature_importance(importance_type="gain"), index=feat_cols)
    mfk_pct = imp[imp.index.str.startswith("mfk_")].sum() / max(imp.sum(), 1) * 100
    pyr_pct = imp[imp.index.str.startswith("pyr_")].sum() / max(imp.sum(), 1) * 100

    Path(OUT_DIR / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic": float(ic), "rank_ic": float(rank_ic),
        "rank_ic_ir": float(rank_ic_ir),
        "n_train": len(train), "n_test": len(test),
        "n_features": len(feat_cols),
        "version": "v7", "leakage_fix": True,
        "mfk_importance_pct": float(mfk_pct),
        "pyramid_importance_pct": float(pyr_pct),
        "anchor_p5": p5_pred, "anchor_p50": p50_pred, "anchor_p95": p95_pred,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  r20_v7_all 完成:", flush=True)
    print(f"    IC={ic:.4f} (V4: 0.031)", flush=True)
    print(f"    RankIC={rank_ic:.4f} (V4: 0.072)", flush=True)
    print(f"    RankICIR={rank_ic_ir:.3f} (V4: 0.81)", flush=True)
    print(f"    best_iter={clf.best_iteration_} (V4: 278)", flush=True)
    print(f"    mfk imp={mfk_pct:.1f}%, pyramid imp={pyr_pct:.1f}%", flush=True)
    print(f"    锚点: p5={p5_pred:.3f}, p50={p50_pred:.3f}, p95={p95_pred:.3f}", flush=True)
    print(f"  Top 20 importance:", flush=True)
    print(imp.sort_values(ascending=False).head(20).to_string(), flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
