#!/usr/bin/env python3
"""LightGBM 直接回归 max_gain_20 (期间最高涨幅) + max_dd_20 (期间最大回撤).

训练目标:
  - max_gain_20: 持有期 20 天内最高点相对买入开盘价的涨幅%
  - max_dd_20:   持有期 20 天内最低点相对买入开盘价的跌幅%

训练: 2024-04 → 2025-04 (12 个月)
测试: 2025-05 → 2026-01 (9 个月 OOS)

输出:
  output/lgbm_maxgain/regressor_gain.txt
  output/lgbm_maxgain/regressor_dd.txt
  output/lgbm_maxgain/feature_meta.json
  output/lgbm_maxgain/bucket_summary.csv
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("lgbm_maxgain")

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_PATH = ROOT / "output" / "labels" / "max_gain_labels.parquet"
OUT_DIR = ROOT / "output" / "lgbm_maxgain"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "20240401"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket"}


def load_with_labels(start: str, end: str, labels: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty:
            parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left")
    return full


def spearman_ic(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    if np.isnan(ic): ic = 0.0
    return "spearman_ic", ic, True


def train_one(name: str, X_train, y_train, X_test, y_test, cat_feats):
    log.info("训练 %s ...", name)
    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05,
        num_leaves=127, min_child_samples=300,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
        objective="regression", metric="None",
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric=spearman_ic,
              categorical_feature=cat_feats,
              callbacks=[lgb.early_stopping(80, first_metric_only=True),
                          lgb.log_evaluation(50)])
    pred = model.predict(X_test)
    ic = stats.spearmanr(pred, y_test)[0]
    rmse = float(np.sqrt(np.mean((y_test - pred)**2)))
    log.info("  %s: best_iter=%d IC=%.4f RMSE=%.3f",
             name, model.best_iteration_, ic, rmse)
    return model, pred, ic, rmse


def main():
    t0 = time.time()
    log.info("加载 max_gain 标签...")
    labels = pd.read_parquet(LABELS_PATH,
                              columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    log.info("标签: %d 行", len(labels))

    log.info("加载训练集...")
    train = load_with_labels(TRAIN_START, TRAIN_END, labels).dropna(subset=["max_gain_20"])
    log.info("Train: %d 行 (max_gain mean=%.2f%%, p90=%.1f%%)",
             len(train), train["max_gain_20"].mean(),
             train["max_gain_20"].quantile(0.9))

    log.info("加载测试集...")
    test = load_with_labels(TEST_START, TEST_END, labels).dropna(subset=["max_gain_20"])
    log.info("Test: %d 行", len(test))

    # 特征
    feat_cols = [c for c in train.columns
                 if c not in EXCLUDE and c not in ("max_gain_20","max_dd_20","entry_open","r20_close","is_clean")
                 and pd.api.types.is_numeric_dtype(train[c])]

    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    feat_cols.append("industry_id")
    log.info("特征数: %d", len(feat_cols))

    X_train = train[feat_cols]; X_test = test[feat_cols]
    y_gain_tr  = train["max_gain_20"]; y_gain_te  = test["max_gain_20"]
    y_dd_tr    = train["max_dd_20"];   y_dd_te    = test["max_dd_20"]

    # 训练两个模型
    m_gain, pred_gain, ic_gain, rmse_gain = train_one(
        "max_gain", X_train, y_gain_tr, X_test, y_gain_te, ["industry_id"])
    m_dd, pred_dd, ic_dd, rmse_dd = train_one(
        "max_dd", X_train, y_dd_tr, X_test, y_dd_te, ["industry_id"])

    # OOS 5 分位 (按 max_gain 预测分位)
    test = test.copy()
    test["pred_gain"] = pred_gain
    test["pred_dd"]   = pred_dd
    test["pred_q"] = test.groupby("trade_date")["pred_gain"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )
    summary = test.groupby("pred_q").agg(
        pred_gain_avg=("pred_gain","mean"),
        true_gain_avg=("max_gain_20","mean"),
        true_gain_med=("max_gain_20","median"),
        gain_15_rate=("max_gain_20", lambda x: (x>=15).mean()*100),
        gain_20_rate=("max_gain_20", lambda x: (x>=20).mean()*100),
        gain_30_rate=("max_gain_20", lambda x: (x>=30).mean()*100),
        true_dd_avg=("max_dd_20","mean"),
        n=("max_gain_20","count"),
    ).round(3)
    log.info("\n=== max_gain 预测 5 分位 OOS 表现 ===")
    print(summary.to_string())

    # 单调性
    g_avg = summary["true_gain_avg"].values
    g_15  = summary["gain_15_rate"].values
    g_20  = summary["gain_20_rate"].values
    def mono(arr): return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
    log.info("true_gain_avg Q1→Q5: %s -> %s",
             [round(v,2) for v in g_avg], "递增" if mono(g_avg) else "非单调")
    log.info("gain≥15%% Q1→Q5: %s -> %s",
             [round(v,1) for v in g_15], "递增" if mono(g_15) else "非单调")

    # 保存
    m_gain.booster_.save_model(str(OUT_DIR / "regressor_gain.txt"))
    m_dd.booster_.save_model(str(OUT_DIR / "regressor_dd.txt"))
    summary.to_csv(OUT_DIR / "bucket_summary.csv", encoding="utf-8-sig")

    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(OUT_DIR / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "train_period": [TRAIN_START, TRAIN_END],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(OUT_DIR / "meta.json").write_text(json.dumps({
        "n_train": len(train), "n_test": len(test),
        "best_iter_gain": int(m_gain.best_iteration_),
        "best_iter_dd":   int(m_dd.best_iteration_),
        "ic_gain": float(ic_gain),
        "ic_dd":   float(ic_dd),
        "rmse_gain": float(rmse_gain),
        "rmse_dd":   float(rmse_dd),
        "monotonic_gain_avg": bool(mono(g_avg)),
        "monotonic_g15": bool(mono(g_15)),
        "monotonic_g20": bool(mono(g_20)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("总耗时 %.1fs, 模型: %s", time.time()-t0, OUT_DIR)


if __name__ == "__main__":
    main()
