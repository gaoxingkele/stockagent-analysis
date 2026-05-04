#!/usr/bin/env python3
"""训练 4 个生产级 LGBM 模型 (基于 r10/r20 真实标签 + ALL 特征).

模型:
  r10_all   : LGBM 回归预测 r10 (10日收益)
  r20_all   : LGBM 回归预测 r20 (20日收益)
  sell_10   : LGBM 二分类预测 P(max_dd_10 <= -5%) (10日内显著下跌)
  sell_20   : LGBM 二分类预测 P(max_dd_20 <= -8%) (20日内显著下跌)

特征: ALL = 153 原因子 + 6 amount + 5 rextra + 13 moneyflow + 8 cogalpha + industry_id (~196)

训练: 2023-01 → 2025-04
测试: 2025-05 → 2026-01

输出:
  output/production/r10_all/{classifier.txt,feature_meta.json,meta.json}
  output/production/r20_all/...
  output/production/sell_10/...
  output/production/sell_20/...
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"  # 含 max_dd_20
OUT_BASE = ROOT / "output" / "production"
OUT_BASE.mkdir(exist_ok=True)

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

    # 标签
    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10","max_dd_10"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    l20 = pd.read_parquet(LABELS_20, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")

    # ALL features
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
    # regime
    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path,
                              columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        full = full.merge(rg, on="trade_date", how="left")
    return full


def spearman_ic_eval(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    return "spearman_ic", ic if not np.isnan(ic) else 0.0, True


def train_regressor(name: str, train, test, feat_cols: list[str],
                     industries, y_col: str):
    """训练回归模型 (r10/r20)."""
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)

    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")
    train = train.dropna(subset=[y_col])
    test = test.dropna(subset=[y_col])
    print(f"  {name} train={len(train):,} test={len(test):,} feat={len(feat_cols)}")

    X_train = train[feat_cols]; y_train = train[y_col]
    X_test  = test[feat_cols];  y_test  = test[y_col]

    clf = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             eval_metric=spearman_ic_eval,
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(80, first_metric_only=True),
                          lgb.log_evaluation(0)])

    # 立即保存
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": y_col, "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 评估
    y_pred = clf.predict(X_test)
    ic = stats.pearsonr(y_pred, y_test)[0]
    rank_ic = stats.spearmanr(y_pred, y_test)[0]
    # 月度 ICIR
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
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name} 保存完成: IC={ic:.4f} RankIC={rank_ic:.4f} RankICIR={rank_ic_ir:.3f} best_iter={clf.best_iteration_}")
    return clf


def train_classifier(name: str, train, test, feat_cols: list[str],
                      industries, label_col: str, threshold: float, condition: str):
    """训练二分类 (sell_10/sell_20).
    condition: 'le' = label <= threshold (回撤判断, threshold 应为负数)
    """
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)

    train = train.dropna(subset=[label_col]).copy()
    test = test.dropna(subset=[label_col]).copy()
    if condition == "le":
        train["y"] = (train[label_col] <= threshold).astype(int)
        test["y"]  = (test[label_col] <= threshold).astype(int)

    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")
    print(f"  {name} train={len(train):,} test={len(test):,} 正例={train['y'].mean()*100:.1f}%")

    X_train = train[feat_cols]; y_train = train["y"]
    X_test  = test[feat_cols];  y_test  = test["y"]
    pos_ratio = y_train.mean()
    spw = (1 - pos_ratio) / pos_ratio

    clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=spw,
        random_state=42, n_jobs=-1, verbose=-1,
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
        "label_col": label_col, "threshold": threshold, "condition": condition,
        "model_type": "classifier",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "auc": float(auc), "base_rate": float(y_test.mean()),
        "n_train": len(train), "n_test": len(test),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {name} 保存完成: AUC={auc:.4f} base={y_test.mean()*100:.1f}% best_iter={clf.best_iteration_}")
    return clf


def main():
    t0 = time.time()
    print("加载训练 + 测试集...")
    train = load_window(TRAIN_START, TRAIN_END)
    test  = load_window(TEST_START, TEST_END)
    print(f"Train: {len(train):,}, Test: {len(test):,}")

    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    feat_cols = [c for c in train.columns
                 if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
    print(f"特征数: {len(feat_cols)}")

    print("\n=== 训练 r10_all 回归器 ===")
    train_regressor("r10_all", train.copy(), test.copy(), feat_cols.copy(), industries, "r10")

    print("\n=== 训练 r20_all 回归器 ===")
    train_regressor("r20_all", train.copy(), test.copy(), feat_cols.copy(), industries, "r20")

    print("\n=== 训练 sell_10 (max_dd_10 <= -5%) ===")
    train_classifier("sell_10", train.copy(), test.copy(), feat_cols.copy(), industries,
                      "max_dd_10", -5.0, "le")

    print("\n=== 训练 sell_20 (max_dd_20 <= -8%) ===")
    train_classifier("sell_20", train.copy(), test.copy(), feat_cols.copy(), industries,
                      "max_dd_20", -8.0, "le")

    print(f"\n总耗时 {time.time()-t0:.1f}s")
    print(f"输出: {OUT_BASE}")


if __name__ == "__main__":
    main()
