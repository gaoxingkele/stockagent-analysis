#!/usr/bin/env python3
"""V4 生产模型 (修正标签泄漏 + 因子精选).

变更 vs V3:
  1. EXCLUDE 加 max_gain_20, max_dd_20 (修正泄漏)
  2. r20 模型加 17 个 mfk 因子 (V4 评估证明 IC 最高)
  3. r10 不加 v2/mfk (验证: ALL alone 最佳)
  4. 4 模型全部用一致的 clean EXCLUDE

4 模型:
  r10_v4_all   : ALL (203 features) — IC 0.085 OOS
  r20_v4_all   : ALL + 17 mfk (220 features) — IC 0.057 RankICIR 0.95 OOS
  sell_10_v4   : ALL (203 features) clean
  sell_20_v4   : ALL (203 features) clean

输出: output/production/{r10_v4_all, r20_v4_all, sell_10_v4, sell_20_v4}/
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
OUT_BASE = ROOT / "output" / "production"
OUT_BASE.mkdir(exist_ok=True)

TRAIN_START = "20230101"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

# 修正后的 EXCLUDE: 加上 max_gain_20, max_dd_20 防止泄漏
EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket",
           "max_gain_10","max_dd_10","max_gain_20","max_dd_20","entry_open"}


def load_window(start, end, with_mfk=False):
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
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")

    if with_mfk:
        p = ROOT / "output" / "mfk_features" / "features.parquet"
        if p.exists():
            d = pd.read_parquet(p); d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on=["ts_code","trade_date"], how="left")

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


def train_regressor(name: str, train, test, feat_cols, industries, y_col,
                     train_subsample=1_500_000):
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)

    train = train[feat_cols + ["industry", y_col, "trade_date"]].dropna(subset=[y_col]).copy()
    test = test[feat_cols + ["industry", y_col, "trade_date"]].dropna(subset=[y_col]).copy()
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    feat_cols = list(feat_cols)
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")
    if len(train) > train_subsample:
        train = train.sample(n=train_subsample, random_state=42).reset_index(drop=True)
    print(f"  {name}: train={len(train):,} test={len(test):,} feat={len(feat_cols)}", flush=True)

    X_train = train[feat_cols].astype("float32"); y_train = train[y_col].astype("float32")
    X_test  = test[feat_cols].astype("float32");  y_test  = test[y_col].astype("float32")

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

    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": y_col, "model_type": "regressor",
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
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic": float(ic), "rank_ic": float(rank_ic),
        "rank_ic_ir": float(rank_ic_ir),
        "n_train": len(train), "n_test": len(test),
        "n_features": len(feat_cols),
        "version": "v4",
        "leakage_fix": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name} 完成: IC={ic:.4f} RankIC={rank_ic:.4f} RankICIR={rank_ic_ir:.3f} best_iter={clf.best_iteration_}",
          flush=True)
    # 锚点
    p5_pred = float(np.quantile(y_pred, 0.05))
    p50_pred = float(np.quantile(y_pred, 0.50))
    p95_pred = float(np.quantile(y_pred, 0.95))
    print(f"  {name} OOS 锚点: p5={p5_pred:.3f}, p50={p50_pred:.3f}, p95={p95_pred:.3f}", flush=True)
    del clf, X_train, X_test, y_train, y_test
    gc.collect()
    return ic, rank_ic, rank_ic_ir


def train_classifier(name: str, train, test, feat_cols, industries,
                      label_col, threshold, train_subsample=1_500_000):
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
    print(f"  {name}: train={len(train):,} test={len(test):,} 正例={pos_ratio*100:.1f}%", flush=True)

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
    p5_p = float(np.quantile(proba, 0.05))
    p50_p = float(np.quantile(proba, 0.50))
    p95_p = float(np.quantile(proba, 0.95))
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "auc": float(auc), "base_rate": float(y_test.mean()),
        "n_train": len(train), "n_test": len(test),
        "n_features": len(feat_cols),
        "version": "v4",
        "leakage_fix": True,
        "anchor_p5": p5_p, "anchor_p50": p50_p, "anchor_p95": p95_p,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name} 完成: AUC={auc:.4f} 锚 p5/p50/p95={p5_p:.3f}/{p50_p:.3f}/{p95_p:.3f}", flush=True)
    del clf, X_train, X_test, y_train, y_test
    gc.collect()
    return auc


def main():
    t0 = time.time()
    print("=== V4 生产模型训练 (无泄漏) ===\n", flush=True)

    # ── r10_v4_all (ALL only, 不加 v2/mfk) ──
    print("[1/4] r10_v4_all (ALL only, 203 feat)", flush=True)
    train = load_window(TRAIN_START, TRAIN_END, with_mfk=False)
    test  = load_window(TEST_START, TEST_END, with_mfk=False)
    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    feat_cols = [c for c in train.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
    train_regressor("r10_v4_all", train, test, feat_cols, industries, "r10")

    # ── sell_10_v4 (ALL only, max_dd_10 <= -5%) ──
    print(f"\n[2/4] sell_10_v4 (ALL only, max_dd_10 <= -5%)", flush=True)
    train_classifier("sell_10_v4", train, test, feat_cols, industries, "max_dd_10", -5.0)

    # ── sell_20_v4 (ALL only, max_dd_20 <= -8%) ──
    print(f"\n[3/4] sell_20_v4 (ALL only, max_dd_20 <= -8%)", flush=True)
    train_classifier("sell_20_v4", train, test, feat_cols, industries, "max_dd_20", -8.0)
    del train, test
    gc.collect()

    # ── r20_v4_all (ALL + 17 mfk) ──
    print(f"\n[4/4] r20_v4_all (ALL + 17 mfk, 220 feat)", flush=True)
    train = load_window(TRAIN_START, TRAIN_END, with_mfk=True)
    test  = load_window(TEST_START, TEST_END, with_mfk=True)
    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    feat_cols = [c for c in train.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
    train_regressor("r20_v4_all", train, test, feat_cols, industries, "r20")

    print(f"\n=== 总耗时 {time.time()-t0:.0f}s ===", flush=True)
    print(f"输出: {OUT_BASE}/{{r10_v4_all, r20_v4_all, sell_10_v4, sell_20_v4}}", flush=True)


if __name__ == "__main__":
    main()
