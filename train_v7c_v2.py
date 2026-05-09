#!/usr/bin/env python3
"""V7c 第二代重训: 4 模型 (r10/r20/sell_10/sell_20) 数据扩到 04-03.

核心改动:
  - labels: union(主 + ext_0403)
  - factor_lab: factor_groups + ext + ext2 全合
  - features: 主 + ext_0507 全合 (amount/mf/mfk/pyramid/v7_extras)
  - regime: 已全量到 05-07
  - 训练窗口: 20230101 → 20250630 (扩 2 个月)
  - 测试窗口: 20250701 → 20260403 (扩 2.3 个月新 OOS)

输出: output/production/{r10_v7c_v2, r20_v7c_v2, sell_10_v7c_v2, sell_20_v7c_v2}/
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
PARQUET_EXT = ROOT / "output" / "factor_lab_3y" / "factor_groups_extension"
LABELS_10  = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_10E = ROOT / "output" / "cogalpha_features" / "labels_10d_ext.parquet"
LABELS_20  = ROOT / "output" / "labels" / "max_gain_labels.parquet"
LABELS_20E = ROOT / "output" / "labels" / "max_gain_labels_ext.parquet"
OUT_BASE   = ROOT / "output" / "production"
OUT_BASE.mkdir(exist_ok=True)

TRAIN_START = "20230101"
TRAIN_END   = "20250630"
TEST_START  = "20250701"
TEST_END    = "20260403"

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket",
           "max_gain_10","max_dd_10","max_gain_20","max_dd_20","entry_open",
           "r20_close","is_clean"}


def _read_factor_lab(start, end):
    """factor_groups + factor_groups_extension/* 全合, 过滤窗口."""
    parts = []
    for p in sorted(PARQUET_DIR.glob("group_*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    for p in sorted(PARQUET_EXT.glob("group_*_ext*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full = full.drop_duplicates(subset=["ts_code","trade_date"], keep="last").reset_index(drop=True)
    return full


def _read_labels_10(start, end):
    """labels_10d → r10, r20, max_dd_10 (合法 target)."""
    parts = []
    for p in [LABELS_10, LABELS_10E]:
        if p.exists():
            d = pd.read_parquet(p, columns=["ts_code","trade_date","r10","r20","max_dd_10"])
            d["trade_date"] = d["trade_date"].astype(str)
            d = d[(d["trade_date"] >= start) & (d["trade_date"] <= end)]
            parts.append(d)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["ts_code","trade_date"], keep="last")
    return df


def _read_labels_20(start, end):
    parts = []
    for p in [LABELS_20, LABELS_20E]:
        if p.exists():
            d = pd.read_parquet(p, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
            d["trade_date"] = d["trade_date"].astype(str)
            d = d[(d["trade_date"] >= start) & (d["trade_date"] <= end)]
            parts.append(d)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["ts_code","trade_date"], keep="last")
    return df


def _read_feature(main_path, ext_path):
    parts = []
    for p in [main_path, ext_path]:
        if p is None or str(p) == "" or str(p).endswith(("\\", "/")):
            continue
        if not Path(p).is_file():
            continue
        d = pd.read_parquet(p)
        d["trade_date"] = d["trade_date"].astype(str)
        parts.append(d)
    if not parts: return None
    df = pd.concat(parts, ignore_index=True)
    if "ts_code" in df.columns:
        df = df.drop_duplicates(subset=["ts_code","trade_date"], keep="last")
    else:
        df = df.drop_duplicates(subset=["trade_date"], keep="last")
    return df


def load_window(start, end):
    full = _read_factor_lab(start, end)
    print(f"    factor_lab: {len(full):,} 行", flush=True)

    # 先 drop factor_lab 自带的 forward 列, 让 labels 成为唯一 target 来源 (避免 merge 出 _l10/_l20 后缀残留泄漏)
    forward_cols_in_fl = [c for c in ["r5","r10","r20","r30","r40","dd5","dd10","dd20","dd30","dd40",
                                       "max_gain_10","max_dd_10","max_gain_20","max_dd_20","entry_open"]
                           if c in full.columns]
    if forward_cols_in_fl:
        full = full.drop(columns=forward_cols_in_fl)
        print(f"    drop factor_lab 自带 forward 列: {forward_cols_in_fl}", flush=True)

    l10 = _read_labels_10(start, end)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    print(f"    + labels_10d: r10/r20/max_dd_10", flush=True)

    l20 = _read_labels_20(start, end)
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")
    print(f"    + max_gain_labels: max_gain_20/max_dd_20", flush=True)

    FEATURES = [
        ("amount",        "output/amount_features/amount_features.parquet",
                          "output/amount_features/amount_features_ext_0507.parquet"),
        ("regime_extra",  "output/regime_extra/regime_extra.parquet", None),
        ("moneyflow_v1",  "output/moneyflow/features.parquet",
                          "output/moneyflow/features_ext_0507.parquet"),
        ("cogalpha",      "output/cogalpha_features/features.parquet", None),
        ("mfk",           "output/mfk_features/features.parquet",
                          "output/mfk_features/features_ext_0507.parquet"),
        ("pyramid",       "output/pyramid_v2/features.parquet",
                          "output/pyramid_v2/features_ext_0507.parquet"),
        ("v7_extras",     "output/v7_extras/features.parquet",
                          "output/v7_extras/features_ext_0507.parquet"),
    ]
    for name, mp, ep in FEATURES:
        ep_path = (ROOT / ep) if ep else None
        d = _read_feature(ROOT / mp, ep_path)
        if d is None: continue
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left", suffixes=("",f"_{name}"))
        else:
            full = full.merge(d, on="trade_date", how="left", suffixes=("",f"_{name}"))
        print(f"    + {name}", flush=True)

    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    rg = pd.read_parquet(rg_path,
        columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
    rg["trade_date"] = rg["trade_date"].astype(str)
    rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                              "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
    full = full.merge(rg, on="trade_date", how="left")
    print(f"    + daily_regime", flush=True)
    return full


def spearman_ic(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    return "spearman_ic", ic if not np.isnan(ic) else 0.0, True


def train_regressor(name, train, test, feat_cols, industries, y_col, train_subsample=1_500_000):
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)
    train = train[feat_cols + ["industry", y_col, "trade_date"]].dropna(subset=[y_col]).copy()
    test  = test[feat_cols + ["industry", y_col, "trade_date"]].dropna(subset=[y_col]).copy()
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"]  = pd.Categorical(test["industry"].fillna("unknown"),
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
        reg_alpha=0.1, reg_lambda=0.1, max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             eval_metric=spearman_ic, categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(80, first_metric_only=True), lgb.log_evaluation(0)])
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
    p5 = float(np.quantile(y_pred, 0.05)); p50 = float(np.quantile(y_pred, 0.50))
    p95 = float(np.quantile(y_pred, 0.95))
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic": float(ic), "rank_ic": float(rank_ic), "rank_ic_ir": float(rank_ic_ir),
        "n_train": len(train), "n_test": len(test), "n_features": len(feat_cols),
        "version": "v7c_v2", "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name}: IC={ic:.4f} RankIC={rank_ic:.4f} RankICIR={rank_ic_ir:.3f} "
          f"best_iter={clf.best_iteration_} 锚 p5/p50/p95={p5:.3f}/{p50:.3f}/{p95:.3f}", flush=True)
    del clf, X_train, X_test, y_train, y_test; gc.collect()


def train_classifier(name, train, test, feat_cols, industries, label_col, threshold,
                      train_subsample=1_500_000):
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)
    train = train[feat_cols + ["industry", label_col, "trade_date"]].dropna(subset=[label_col]).copy()
    test  = test[feat_cols + ["industry", label_col, "trade_date"]].dropna(subset=[label_col]).copy()
    train["y"] = (train[label_col] <= threshold).astype(int)
    test["y"]  = (test[label_col] <= threshold).astype(int)
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"]  = pd.Categorical(test["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    feat_cols = list(feat_cols)
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")
    if len(train) > train_subsample:
        train = train.sample(n=train_subsample, random_state=42).reset_index(drop=True)
    pos_ratio = train["y"].mean()
    spw = (1 - pos_ratio) / max(pos_ratio, 1e-3)
    print(f"  {name}: train={len(train):,} test={len(test):,} pos%={pos_ratio*100:.1f}", flush=True)
    X_train = train[feat_cols].astype("float32"); y_train = train["y"]
    X_test  = test[feat_cols].astype("float32");  y_test  = test["y"]

    clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=spw,
        max_bin=127, force_col_wise=True,
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
    p5 = float(np.quantile(proba, 0.05)); p50 = float(np.quantile(proba, 0.50))
    p95 = float(np.quantile(proba, 0.95))
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "auc": float(auc), "base_rate": float(y_test.mean()),
        "n_train": len(train), "n_test": len(test), "n_features": len(feat_cols),
        "version": "v7c_v2", "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name}: AUC={auc:.4f} base={y_test.mean()*100:.1f}% 锚 p5/p50/p95={p5:.3f}/{p50:.3f}/{p95:.3f}",
          flush=True)
    del clf, X_train, X_test, y_train, y_test; gc.collect()


def main():
    t0 = time.time()
    print("=== V7c v2 4 模型重训 ===", flush=True)
    print(f"TRAIN: {TRAIN_START} → {TRAIN_END} | TEST: {TEST_START} → {TEST_END}\n", flush=True)

    print("加载训练集 ...", flush=True)
    train = load_window(TRAIN_START, TRAIN_END)
    print(f"  total: {len(train):,} 行 × {len(train.columns)} 列 ({time.time()-t0:.0f}s)", flush=True)

    print("\n加载测试集 ...", flush=True)
    test  = load_window(TEST_START, TEST_END)
    print(f"  total: {len(test):,} 行 × {len(test.columns)} 列 ({time.time()-t0:.0f}s)", flush=True)

    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    feat_cols = [c for c in train.columns
                  if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
    print(f"\n特征数: {len(feat_cols)}", flush=True)

    # 泄漏自检: feat_cols 不应有任何 forward-looking 列
    import re
    suspect = [f for f in feat_cols if re.search(
        r"(^r\d+$|^dd\d+$|^max_(gain|dd)_|^r20_close|^entry_open|_l10$|_l20$|^r\d+_|^dd\d+_)", f)]
    if suspect:
        raise RuntimeError(f"泄漏自检失败! 可疑列: {suspect}")
    print("  泄漏自检通过 (无 forward-looking 列)", flush=True)

    print(f"\n[1/4] r10_v7c_v2 (target r10)", flush=True)
    train_regressor("r10_v7c_v2", train, test, feat_cols, industries, "r10")

    print(f"\n[2/4] r20_v7c_v2 (target r20)", flush=True)
    train_regressor("r20_v7c_v2", train, test, feat_cols, industries, "r20")

    print(f"\n[3/4] sell_10_v7c_v2 (max_dd_10 <= -5%)", flush=True)
    train_classifier("sell_10_v7c_v2", train, test, feat_cols, industries, "max_dd_10", -5.0)

    print(f"\n[4/4] sell_20_v7c_v2 (max_dd_20 <= -8%)", flush=True)
    train_classifier("sell_20_v7c_v2", train, test, feat_cols, industries, "max_dd_20", -8.0)

    print(f"\n=== 总耗时 {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
