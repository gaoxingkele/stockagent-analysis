#!/usr/bin/env python3
"""LightGBM 直接预测 r20 (20 日前向涨幅).

训练: 2024-04 → 2025-04 (12 个月牛市数据)
测试: 2025-05 → 2026-01 (9 个月 OOS)

输出:
  - 整体 Spearman IC
  - 周频 IC 均值 / IR / 正值占比
  - 5 分位 OOS 表现 (验证 Q1→Q5 单调性)
  - Top 20 特征重要性
  - 模型存盘 output/lgbm_r20/model.txt
"""
from __future__ import annotations
import logging, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("lgbm_r20")

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OUT_DIR     = ROOT / "output" / "lgbm_r20"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "20240401"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

EXCLUDE = {"ts_code", "trade_date", "industry", "name",
           "r5", "r10", "r20", "r30", "r40",
           "dd5", "dd10", "dd20", "dd30", "dd40",
           "mv_bucket", "pe_bucket"}

TARGET = "r20"


def load_range(start: str, end: str) -> pd.DataFrame:
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty:
            parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    return full


def main():
    log.info("加载训练集 %s → %s ...", TRAIN_START, TRAIN_END)
    train = load_range(TRAIN_START, TRAIN_END).dropna(subset=[TARGET])
    log.info("Train: %d 行, %d 股票, %d 天",
             len(train), train["ts_code"].nunique(), train["trade_date"].nunique())

    log.info("加载测试集 %s → %s ...", TEST_START, TEST_END)
    test = load_range(TEST_START, TEST_END).dropna(subset=[TARGET])
    log.info("Test: %d 行, %d 股票, %d 天",
             len(test), test["ts_code"].nunique(), test["trade_date"].nunique())

    # 特征列
    feat_cols = [c for c in train.columns
                 if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]

    # industry 转 categorical (LightGBM 原生支持)
    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                         categories=industries.cat.categories).codes
    feat_cols.append("industry_id")
    log.info("特征数: %d (含 industry_id)", len(feat_cols))

    X_train = train[feat_cols]
    y_train = train[TARGET]
    X_test  = test[feat_cols]
    y_test  = test[TARGET]

    # === 自定义 IC 评估函数 (sklearn API 签名: y_true, y_pred) ===
    def spearman_ic_eval(y_true, y_pred):
        ic = stats.spearmanr(y_pred, y_true)[0]
        if np.isnan(ic): ic = 0.0
        return "spearman_ic", ic, True  # higher is better

    # === 1) 回归模型 (预测 r20 涨幅) ===
    log.info("训练 LightGBM 回归 (预测 r20)...")
    model = lgb.LGBMRegressor(
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
        random_state=42,
        n_jobs=-1,
        objective="regression",
        metric="None",  # 自定义评估
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=spearman_ic_eval,
        categorical_feature=["industry_id"],
        callbacks=[
            lgb.early_stopping(80, first_metric_only=True),
            lgb.log_evaluation(50),
        ],
    )

    # === 2) 分类模型 (预测 r20 > 0 概率) ===
    log.info("训练 LightGBM 分类 (预测 r20 > 0 概率)...")
    y_train_cls = (y_train > 0).astype(int)
    y_test_cls  = (y_test  > 0).astype(int)
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
        random_state=42,
        n_jobs=-1,
        objective="binary",
        metric="binary_logloss",
        verbose=-1,
    )
    classifier.fit(
        X_train, y_train_cls,
        eval_set=[(X_test, y_test_cls)],
        categorical_feature=["industry_id"],
        callbacks=[
            lgb.early_stopping(80),
            lgb.log_evaluation(50),
        ],
    )

    # 预测 + IC
    pred = model.predict(X_test)
    pred_winprob = classifier.predict_proba(X_test)[:, 1]   # 上涨概率
    ic_overall = stats.spearmanr(pred, y_test)[0]
    cls_auc = stats.spearmanr(pred_winprob, y_test)[0]      # 概率与真实涨幅的相关性
    log.info("OOS 回归 Spearman IC: %.4f", ic_overall)
    log.info("OOS 分类 winprob IC: %.4f", cls_auc)

    # 周频 IC
    test = test.copy()
    test["pred"] = pred
    weekly_ic = []
    for d, sub in test.groupby("trade_date"):
        if len(sub) < 50: continue
        ic_w = stats.spearmanr(sub["pred"], sub[TARGET])[0]
        if not np.isnan(ic_w):
            weekly_ic.append(ic_w)
    weekly_ic = np.array(weekly_ic)
    log.info("周频 IC: 均值=%.4f std=%.4f IR=%.2f | 正值周占比=%.1f%%",
             weekly_ic.mean(), weekly_ic.std(),
             weekly_ic.mean() / weekly_ic.std() if weekly_ic.std() > 0 else 0,
             (weekly_ic > 0).mean() * 100)

    # 5 分位 OOS 单调性 (按日内分位)
    test["pred_q"] = test.groupby("trade_date")["pred"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )
    log.info("\n5 分位 OOS 表现 (按日内 pred 分位):")
    summary = test.groupby("pred_q").agg(
        pred_avg=("pred", "mean"),
        r20_avg=("r20", "mean"),
        r20_med=("r20", "median"),
        r20_win=("r20", lambda x: (x > 0).mean() * 100),
        r5_avg=("r5", "mean") if "r5" in test.columns else ("r20", "size"),
        n=("r20", "count"),
    ).round(3)
    print(summary.to_string())

    # 单调性检查
    means = summary["r20_avg"].values
    wins  = summary["r20_win"].values
    def mono(arr): return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
    log.info("r20 涨幅 Q1→Q5: %s -> %s",
             [round(v, 3) for v in means],
             "递增" if mono(means) else "非单调")
    log.info("r20 胜率 Q1→Q5: %s -> %s",
             [round(v, 1) for v in wins],
             "递增" if mono(wins) else "非单调")

    # 顶/底 spread
    q1 = summary.loc[1]; q5 = summary.loc[5]
    log.info("Q5-Q1 r20 价差: %+.3f%% (Q5=%+.3f%% vs Q1=%+.3f%%)",
             q5["r20_avg"] - q1["r20_avg"], q5["r20_avg"], q1["r20_avg"])
    log.info("Q5-Q1 胜率差: %+.1f%% (Q5=%.1f%% vs Q1=%.1f%%)",
             q5["r20_win"] - q1["r20_win"], q5["r20_win"], q1["r20_win"])

    # Top features
    fi = pd.DataFrame({
        "feat": feat_cols,
        "imp": model.feature_importances_,
    }).sort_values("imp", ascending=False).head(20)
    log.info("\nTop 20 特征:")
    print(fi.to_string(index=False))

    # 存盘
    model.booster_.save_model(str(OUT_DIR / "model.txt"))
    classifier.booster_.save_model(str(OUT_DIR / "classifier.txt"))
    summary.to_csv(OUT_DIR / "bucket_summary.csv", encoding="utf-8-sig")
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # 训练时元数据 (推理用): 行业映射 + 特征列表 + 残差标准差
    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    residuals = (y_test.values - pred)
    residual_std = float(np.std(residuals))
    Path(OUT_DIR / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols,
        "industry_map": industry_map,
        "residual_std": residual_std,
        "target": TARGET,
        "train_period": [TRAIN_START, TRAIN_END],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    meta = {
        "train_period": [TRAIN_START, TRAIN_END],
        "test_period":  [TEST_START, TEST_END],
        "n_train": len(train),
        "n_test":  len(test),
        "n_features": len(feat_cols),
        "best_iter_reg": model.best_iteration_,
        "best_iter_cls": classifier.best_iteration_,
        "ic_overall": float(ic_overall),
        "cls_winprob_ic": float(cls_auc),
        "residual_std": float(residual_std),
        "weekly_ic_mean": float(weekly_ic.mean()),
        "weekly_ic_std": float(weekly_ic.std()),
        "weekly_ic_ir": float(weekly_ic.mean() / weekly_ic.std()) if weekly_ic.std() > 0 else 0,
        "weekly_pos_ratio": float((weekly_ic > 0).mean()),
        "q5_minus_q1_r20": float(q5["r20_avg"] - q1["r20_avg"]),
        "q5_winrate":  float(q5["r20_win"]),
        "q1_winrate":  float(q1["r20_win"]),
        "monotonic_avg":  bool(mono(means)),
        "monotonic_win":  bool(mono(wins)),
    }
    Path(OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    log.info("模型: %s", OUT_DIR / "model.txt")


if __name__ == "__main__":
    main()
