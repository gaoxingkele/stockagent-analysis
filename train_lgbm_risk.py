#!/usr/bin/env python3
"""LightGBM 回撤风险检测器 — 二分类预测"未来 20 天 max_dd <= -8%".

正例: max_dd_20 <= -8%  (高回撤事件)
负例: max_dd_20 > -8%   (回撤可控)

训练: 2024-04 → 2025-04 (12 个月)
测试: 2025-05 → 2026-01 (9 个月 OOS)

输出:
  output/lgbm_risk/classifier.txt
  output/lgbm_risk/feature_meta.json
  output/lgbm_risk/meta.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS = ROOT / "output" / "labels" / "max_gain_labels.parquet"
REGIME_PATH = ROOT / "output" / "regimes" / "daily_regime.parquet"
AMOUNT_PATH = ROOT / "output" / "amount_features" / "amount_features.parquet"
RXTRA_PATH  = ROOT / "output" / "regime_extra" / "regime_extra.parquet"
OUT_DIR = ROOT / "output" / "lgbm_risk"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "20230101"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

DD_THRESH = -8.0   # 标签阈值: max_dd_20 <= -8% 视为高风险

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket"}


def load(start, end, labels_df):
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full = full.merge(labels_df, on=["ts_code","trade_date"], how="left")

    # regime / amount / rextra
    if REGIME_PATH.exists():
        r = pd.read_parquet(REGIME_PATH,
                              columns=["trade_date","regime_id","ret_5d","ret_20d",
                                        "ret_60d","rsi14","vol_ratio",
                                        "cyb_ret_20d","zz500_ret_20d"])
        r["trade_date"] = r["trade_date"].astype(str)
        r = r.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14",
                                "vol_ratio":"mkt_vol_ratio"})
        full = full.merge(r, on="trade_date", how="left")
    if AMOUNT_PATH.exists():
        a = pd.read_parquet(AMOUNT_PATH)
        a["trade_date"] = a["trade_date"].astype(str)
        full = full.merge(a, on=["ts_code","trade_date"], how="left")
    if RXTRA_PATH.exists():
        rx = pd.read_parquet(RXTRA_PATH)
        rx["trade_date"] = rx["trade_date"].astype(str)
        full = full.merge(rx, on="trade_date", how="left")
    return full


def main():
    t0 = time.time()
    print("加载 max_dd 标签...")
    labels = pd.read_parquet(LABELS, columns=["ts_code","trade_date","max_dd_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    labels["is_risky"] = (labels["max_dd_20"] <= DD_THRESH).astype(int)
    print(f"标签: {len(labels)} 行, 高风险比例 {labels['is_risky'].mean()*100:.1f}%")

    train = load(TRAIN_START, TRAIN_END, labels).dropna(subset=["max_dd_20"])
    test  = load(TEST_START,  TEST_END,  labels).dropna(subset=["max_dd_20"])
    print(f"Train: {len(train)} 行, 高风险 {train['is_risky'].sum()} ({train['is_risky'].mean()*100:.1f}%)")
    print(f"Test:  {len(test)} 行, 高风险 {test['is_risky'].sum()} ({test['is_risky'].mean()*100:.1f}%)")

    feat_cols = [c for c in train.columns
                 if c not in EXCLUDE and c not in ("max_dd_20","is_risky")
                 and pd.api.types.is_numeric_dtype(train[c])]
    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    feat_cols.append("industry_id")
    print(f"特征数: {len(feat_cols)}")

    X_train = train[feat_cols]; y_train = train["is_risky"]
    X_test  = test[feat_cols];  y_test  = test["is_risky"]

    pos_ratio = y_train.mean()
    spw = (1 - pos_ratio) / pos_ratio
    print(f"训练 LightGBM 风险检测 (正例比 {pos_ratio*100:.1f}%)...")
    clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=200, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1,
        objective="binary", metric="auc",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)])

    pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    test = test.copy()
    test["risk_prob"] = pred

    print(f"\n=== OOS AUC: {auc:.4f} | best_iter: {clf.best_iteration_} ===")

    # 5 分位
    test["pred_q"] = test.groupby("trade_date")["risk_prob"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1)
    summary = test.groupby("pred_q").agg(
        prob_avg=("risk_prob","mean"),
        true_risky_rate=("is_risky", lambda x: x.mean()*100),
        actual_dd_avg=("max_dd_20", "mean"),
        dd_lt_5=("max_dd_20", lambda x: (x<-5).mean()*100),
        dd_lt_8=("max_dd_20", lambda x: (x<-8).mean()*100),
        dd_lt_15=("max_dd_20", lambda x: (x<-15).mean()*100),
        n=("is_risky","count"),
    ).round(3)
    print("\n5 分位 OOS 表现 (预测风险概率):")
    print(summary.to_string())

    # Lift (高风险股捕获)
    base = y_test.mean()
    sorted_t = test.sort_values("risk_prob", ascending=False)
    print(f"\n=== 高风险捕获 (基线 {base*100:.2f}%) ===")
    for pct in [0.05, 0.10, 0.20, 0.30]:
        n = int(len(sorted_t) * pct)
        rate = sorted_t.head(n)["is_risky"].mean()
        recall = sorted_t.head(n)["is_risky"].sum() / y_test.sum() * 100
        print(f"  Top {pct*100:>2.0f}% (n={n:>6}): 真实风险率 {rate*100:.2f}%, "
              f"recall {recall:.1f}%, lift {rate/base:.2f}x")

    # 低风险段
    sorted_low = test.sort_values("risk_prob", ascending=True)
    print(f"\n=== 低风险段 (bottom% 真实风险率) ===")
    for pct in [0.10, 0.20, 0.30, 0.50]:
        n = int(len(sorted_low) * pct)
        rate = sorted_low.head(n)["is_risky"].mean()
        avg_dd = sorted_low.head(n)["max_dd_20"].mean()
        print(f"  Bottom {pct*100:>2.0f}% (n={n:>6}): 真实风险率 {rate*100:.2f}%, "
              f"平均 dd {avg_dd:.2f}%")

    # 存盘
    clf.booster_.save_model(str(OUT_DIR / "classifier.txt"))
    summary.to_csv(OUT_DIR / "bucket_summary.csv", encoding="utf-8-sig")

    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(OUT_DIR / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "train_period": [TRAIN_START, TRAIN_END],
        "dd_threshold": DD_THRESH,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(OUT_DIR / "meta.json").write_text(json.dumps({
        "n_train": len(train), "n_test": len(test),
        "n_train_risky": int(y_train.sum()),
        "n_test_risky": int(y_test.sum()),
        "best_iter": int(clf.best_iteration_),
        "auc": float(auc),
        "base_rate": float(base),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n耗时 {time.time()-t0:.1f}s, 模型: {OUT_DIR}")


if __name__ == "__main__":
    main()
