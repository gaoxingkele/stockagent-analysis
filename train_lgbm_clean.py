#!/usr/bin/env python3
"""LightGBM "干净上涨" 信号检测器.

正例: clean_trends.parquet 里的 is_clean=True 行 (24.9 万)
       (max_gain_20 >= 20% AND max_dd_20 >= -3%)
负例: parquet 中其余行 (从 1308k 训练 + 920k 测试中剔除正例)

输出:
  output/lgbm_clean/classifier.txt          二分类模型
  output/lgbm_clean/feature_meta.json       特征列表 + 行业映射
  output/lgbm_clean/bucket_summary.csv      OOS 5 分位 (按 clean_prob)
  output/lgbm_clean/meta.json               训练摘要
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
log = logging.getLogger("lgbm_clean")

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
CLEAN_PATH  = ROOT / "output" / "clean_trends" / "clean_trends.parquet"
OUT_DIR     = ROOT / "output" / "lgbm_clean"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "20240401"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

EXCLUDE = {"ts_code", "trade_date", "industry", "name",
           "r5", "r10", "r20", "r30", "r40",
           "dd5", "dd10", "dd20", "dd30", "dd40",
           "mv_bucket", "pe_bucket"}


def load_with_label(start: str, end: str, clean_keys: set) -> pd.DataFrame:
    """加载 parquet 并合并 is_clean 标签."""
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty:
            parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    # Label
    full["_key"] = full["ts_code"] + "|" + full["trade_date"]
    full["is_clean"] = full["_key"].isin(clean_keys).astype(int)
    full = full.drop(columns=["_key"])
    return full


def main():
    t0 = time.time()
    log.info("加载 clean_trends 标签...")
    clean = pd.read_parquet(CLEAN_PATH, columns=["ts_code", "trade_date"])
    clean["trade_date"] = clean["trade_date"].astype(str)
    clean_keys = set(clean["ts_code"] + "|" + clean["trade_date"])
    log.info("正例 keys: %d", len(clean_keys))

    log.info("加载训练集 %s → %s ...", TRAIN_START, TRAIN_END)
    train = load_with_label(TRAIN_START, TRAIN_END, clean_keys)
    train = train.dropna(subset=["r20"])
    pos_train = int(train["is_clean"].sum())
    log.info("Train: %d 行, 正例 %d (%.1f%%)",
             len(train), pos_train, pos_train / len(train) * 100)

    log.info("加载测试集 %s → %s ...", TEST_START, TEST_END)
    test = load_with_label(TEST_START, TEST_END, clean_keys)
    test = test.dropna(subset=["r20"])
    pos_test = int(test["is_clean"].sum())
    log.info("Test: %d 行, 正例 %d (%.1f%%)",
             len(test), pos_test, pos_test / len(test) * 100)

    # 特征
    feat_cols = [c for c in train.columns
                 if c not in EXCLUDE and c != "is_clean"
                 and pd.api.types.is_numeric_dtype(train[c])]

    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    feat_cols.append("industry_id")
    log.info("特征数: %d", len(feat_cols))

    X_train = train[feat_cols]; y_train = train["is_clean"]
    X_test  = test[feat_cols];  y_test  = test["is_clean"]

    # 类别不平衡: 正例约 11%, 用 scale_pos_weight 平衡
    pos_ratio = y_train.mean()
    scale_pos_weight = (1 - pos_ratio) / pos_ratio
    log.info("正例比 %.1f%%, scale_pos_weight=%.2f", pos_ratio * 100, scale_pos_weight)

    log.info("训练 LightGBM 分类 (AUC eval)...")
    classifier = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=300,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        objective="binary",
        metric="auc",   # AUC 比 logloss 更适合不平衡
        verbose=-1,
    )
    classifier.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        categorical_feature=["industry_id"],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)],
    )

    pred_prob = classifier.predict_proba(X_test)[:, 1]

    # AUC + 5 分位
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y_test, pred_prob)
    pr_auc = average_precision_score(y_test, pred_prob)
    log.info("OOS AUC: %.4f | PR-AUC: %.4f", auc, pr_auc)

    # 5 分位: 看不同 prob 段的真实 is_clean 率 + r20 表现
    test = test.copy()
    test["clean_prob"] = pred_prob
    test["pred_q"] = test.groupby("trade_date")["clean_prob"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )
    summary = test.groupby("pred_q").agg(
        prob_avg=("clean_prob", "mean"),
        is_clean_rate=("is_clean", lambda x: x.mean() * 100),
        r20_avg=("r20", "mean"),
        r20_win=("r20", lambda x: (x > 0).mean() * 100),
        r20_p20_win=("r20", lambda x: (x > 20).mean() * 100),  # 实际 r20>20% 比例
        n=("is_clean", "count"),
    ).round(3)
    log.info("\n5 分位 OOS 表现 (按 clean_prob 分位):")
    print(summary.to_string())

    # 单调性
    rates = summary["is_clean_rate"].values
    p20s  = summary["r20_p20_win"].values
    def mono(arr): return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
    log.info("is_clean 实际比例 Q1→Q5: %s -> %s",
             [round(v, 1) for v in rates], "递增" if mono(rates) else "非单调")
    log.info("r20>20%% 实际比例 Q1→Q5: %s -> %s",
             [round(v, 1) for v in p20s], "递增" if mono(p20s) else "非单调")

    # Lift: top 10% 包含多少正例
    test_sorted = test.sort_values("clean_prob", ascending=False)
    top10_n = int(len(test_sorted) * 0.10)
    top10_pos_rate = test_sorted.head(top10_n)["is_clean"].mean()
    base_rate = test["is_clean"].mean()
    lift = top10_pos_rate / base_rate
    log.info("Top 10%% clean_prob 中真实正例比例: %.1f%% (基线 %.1f%%, lift %.2fx)",
             top10_pos_rate * 100, base_rate * 100, lift)

    # 特征重要性
    fi = pd.DataFrame({"feat": feat_cols, "imp": classifier.feature_importances_})
    fi = fi.sort_values("imp", ascending=False).head(20)
    log.info("\nTop 20 特征:")
    print(fi.to_string(index=False))

    # 存盘
    classifier.booster_.save_model(str(OUT_DIR / "classifier.txt"))
    summary.to_csv(OUT_DIR / "bucket_summary.csv", encoding="utf-8-sig")
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(OUT_DIR / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols,
        "industry_map": industry_map,
        "train_period": [TRAIN_START, TRAIN_END],
        "target": "is_clean (max_gain_20>=20%, max_dd_20>=-3%)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    Path(OUT_DIR / "meta.json").write_text(json.dumps({
        "train_period": [TRAIN_START, TRAIN_END],
        "test_period":  [TEST_START, TEST_END],
        "n_train": len(train),
        "n_test": len(test),
        "n_train_pos": pos_train,
        "n_test_pos": pos_test,
        "pos_ratio": float(pos_ratio),
        "best_iter": classifier.best_iteration_,
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "top10_lift": float(lift),
        "monotonic_clean_rate": bool(mono(rates)),
        "monotonic_p20": bool(mono(p20s)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("耗时 %.1fs, 模型: %s", time.time() - t0, OUT_DIR / "classifier.txt")


if __name__ == "__main__":
    main()
