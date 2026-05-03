#!/usr/bin/env python3
"""LightGBM 起涨点检测器 — 二分类预测"明天是不是干净起涨点".

正例: extract_uptrend_starts 产出的 21,825 条干净起涨点
负例: 同期所有非起涨点行 (5x 正例采样, ~110k)

训练: 2023-01 → 2025-04
测试: 2025-05 → 2026-01 (OOS)

输出:
  output/lgbm_uptrend/classifier.txt
  output/lgbm_uptrend/feature_meta.json
  output/lgbm_uptrend/meta.json
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
log = logging.getLogger("lgbm_uptrend")

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
STARTS_PATH = ROOT / "output" / "uptrend_starts" / "starts.parquet"
OUT_DIR = ROOT / "output" / "lgbm_uptrend"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "20230101"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

NEG_RATIO = 5  # 负例: 正例 = 5:1

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket"}


def load_with_label(start: str, end: str, pos_keys: set) -> pd.DataFrame:
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty:
            parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["_key"] = full["ts_code"] + "|" + full["trade_date"]
    full["is_uptrend"] = full["_key"].isin(pos_keys).astype(int)

    # 合并 regime 标签
    regime_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if regime_path.exists():
        regime = pd.read_parquet(regime_path,
                                  columns=["trade_date", "regime_id", "ret_5d",
                                            "ret_20d", "ret_60d", "rsi14",
                                            "vol_ratio", "cyb_ret_20d", "zz500_ret_20d"])
        regime["trade_date"] = regime["trade_date"].astype(str)
        regime = regime.rename(columns={
            "ret_5d": "mkt_ret_5d", "ret_20d": "mkt_ret_20d",
            "ret_60d": "mkt_ret_60d", "rsi14": "mkt_rsi14",
            "vol_ratio": "mkt_vol_ratio",
        })
        full = full.merge(regime, on="trade_date", how="left")

    # 合并 amount features (绝对成交额特征)
    amount_path = ROOT / "output" / "amount_features" / "amount_features.parquet"
    if amount_path.exists():
        amount = pd.read_parquet(amount_path)
        amount["trade_date"] = amount["trade_date"].astype(str)
        full = full.merge(amount, on=["ts_code","trade_date"], how="left")
    return full


def main():
    t0 = time.time()
    log.info("加载起涨点正例标签...")
    starts = pd.read_parquet(STARTS_PATH, columns=["ts_code","trade_date"])
    starts["trade_date"] = starts["trade_date"].astype(str)
    pos_keys = set(starts["ts_code"] + "|" + starts["trade_date"])
    log.info("正例 keys: %d", len(pos_keys))

    log.info("加载训练集 %s → %s ...", TRAIN_START, TRAIN_END)
    train = load_with_label(TRAIN_START, TRAIN_END, pos_keys)
    train = train.dropna(subset=["r20"])  # 需要有 forward return
    pos_train = int(train["is_uptrend"].sum())
    log.info("Train (合并前): %d 行, 正例 %d (%.2f%%)",
             len(train), pos_train, pos_train / len(train) * 100)

    # 负例采样: 5x 正例
    pos_df = train[train["is_uptrend"] == 1]
    neg_df = train[train["is_uptrend"] == 0].sample(
        min(NEG_RATIO * pos_train, train["is_uptrend"].eq(0).sum()),
        random_state=42,
    )
    train_balanced = pd.concat([pos_df, neg_df], ignore_index=True)
    log.info("Train (平衡后): %d 行, %d 正例 (%.1f%%)",
             len(train_balanced), pos_train, pos_train/len(train_balanced)*100)

    log.info("加载测试集 %s → %s ...", TEST_START, TEST_END)
    test = load_with_label(TEST_START, TEST_END, pos_keys)
    test = test.dropna(subset=["r20"])
    pos_test = int(test["is_uptrend"].sum())
    log.info("Test: %d 行 (全量), 正例 %d (%.2f%%)",
             len(test), pos_test, pos_test/len(test)*100)

    # 特征 (剔除 _key, is_uptrend)
    feat_cols = [c for c in train_balanced.columns
                 if c not in EXCLUDE and c not in ("_key","is_uptrend")
                 and pd.api.types.is_numeric_dtype(train_balanced[c])]

    industries = pd.concat([train_balanced["industry"], test["industry"]]).fillna("unknown").astype("category")
    train_balanced["industry_id"] = pd.Categorical(
        train_balanced["industry"].fillna("unknown"),
        categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(
        test["industry"].fillna("unknown"),
        categories=industries.cat.categories).codes
    feat_cols.append("industry_id")
    log.info("特征数: %d", len(feat_cols))

    X_train = train_balanced[feat_cols]; y_train = train_balanced["is_uptrend"]
    X_test  = test[feat_cols];           y_test  = test["is_uptrend"]

    log.info("训练 LightGBM 起涨点检测器 (AUC eval)...")
    pos_ratio = y_train.mean()
    spw = (1 - pos_ratio) / pos_ratio
    classifier = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05,
        num_leaves=127, min_child_samples=200,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=spw,
        random_state=42, n_jobs=-1, verbose=-1,
        objective="binary", metric="auc",
    )
    classifier.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    categorical_feature=["industry_id"],
                    callbacks=[lgb.early_stopping(80), lgb.log_evaluation(50)])

    pred_prob = classifier.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y_test, pred_prob)
    pr_auc = average_precision_score(y_test, pred_prob)
    log.info("OOS AUC: %.4f | PR-AUC: %.4f", auc, pr_auc)

    test = test.copy()
    test["uptrend_prob"] = pred_prob

    # 5 分位
    test["pred_q"] = test.groupby("trade_date")["uptrend_prob"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )
    summary = test.groupby("pred_q").agg(
        prob_avg=("uptrend_prob","mean"),
        true_uptrend_rate=("is_uptrend", lambda x: x.mean()*100),
        r20_avg=("r20","mean"),
        r20_win=("r20", lambda x: (x>0).mean()*100),
        n=("is_uptrend","count"),
    ).round(3)
    log.info("\n5 分位 OOS 表现:")
    print(summary.to_string())

    # Lift: top 10%, 5%, 1%
    test_sorted = test.sort_values("uptrend_prob", ascending=False)
    base_rate = test["is_uptrend"].mean()
    log.info("\n=== Lift 分析 (基线起涨率 %.2f%%) ===", base_rate*100)
    for pct in [0.01, 0.05, 0.10, 0.20]:
        top_n = int(len(test_sorted) * pct)
        top_pos_rate = test_sorted.head(top_n)["is_uptrend"].mean()
        lift = top_pos_rate / base_rate
        log.info("Top %.0f%% (%d 样本): 真实起涨率 %.2f%%, lift %.2fx",
                 pct*100, top_n, top_pos_rate*100, lift)

    # 按 mv 切片看
    test["mv_seg"] = test["total_mv"].apply(lambda v:
        ("20-50亿" if v/1e4 < 50 else "50-100亿" if v/1e4 < 100 else
         "100-300亿" if v/1e4 < 300 else "300-1000亿" if v/1e4 < 1000 else "1000亿+")
        if pd.notna(v) else None)
    log.info("\n=== 按 MV 切片 (高分段 top 10%) ===")
    for mv in test["mv_seg"].dropna().unique():
        sub = test[test["mv_seg"] == mv]
        if len(sub) < 1000: continue
        sub_top = sub.sort_values("uptrend_prob", ascending=False).head(int(len(sub)*0.10))
        rate = sub_top["is_uptrend"].mean() * 100
        base = sub["is_uptrend"].mean() * 100
        log.info("  %s: top10%% 起涨率 %.2f%% (基线 %.2f%%, lift %.2fx)",
                 mv, rate, base, rate/base if base > 0 else 0)

    # Top features
    fi = pd.DataFrame({"feat": feat_cols, "imp": classifier.feature_importances_})
    fi = fi.sort_values("imp", ascending=False).head(20)
    log.info("\nTop 20 特征:")
    print(fi.to_string(index=False))

    # 存盘
    classifier.booster_.save_model(str(OUT_DIR / "classifier.txt"))
    summary.to_csv(OUT_DIR / "bucket_summary.csv", encoding="utf-8-sig")

    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(OUT_DIR / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "train_period": [TRAIN_START, TRAIN_END],
        "neg_ratio": NEG_RATIO,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(OUT_DIR / "meta.json").write_text(json.dumps({
        "n_train": int(len(train_balanced)), "n_test": int(len(test)),
        "n_train_pos": pos_train, "n_test_pos": pos_test,
        "best_iter": int(classifier.best_iteration_),
        "auc": float(auc), "pr_auc": float(pr_auc),
        "base_rate": float(base_rate),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("耗时 %.1fs, 模型: %s", time.time()-t0, OUT_DIR)


if __name__ == "__main__":
    main()
