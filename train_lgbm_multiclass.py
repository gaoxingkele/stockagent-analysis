#!/usr/bin/env python3
"""LightGBM 多分类 (5 类标签).

类别:
  0: other     (混合, 不参与决策)
  1: sideways  (震荡)
  2: mild      (温和上涨 15-20%)
  3: strong    (优秀上涨 20-25%)
  4: aggressive (激进上涨 >25%)
  5: fake_pump (假上涨, 避雷)

训练 multi-class LGBM, 输出每类概率.
评分逻辑:
  buy_score = aggressive*2 + strong*1.5 + mild*1 - fake_pump*2 - sideways*0.3
  范围 [-2, +6], 后归一到 0-100

输出:
  output/lgbm_multiclass/classifier.txt
  output/lgbm_multiclass/feature_meta.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_V2 = ROOT / "output" / "labels_v2" / "multiclass_labels.parquet"
OUT_DIR = ROOT / "output" / "lgbm_multiclass"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "20230101"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

CLASS_MAP = {"other":0, "sideways":1, "mild":2, "strong":3, "aggressive":4, "fake_pump":5}
CLASS_NAMES = ["other","sideways","mild","strong","aggressive","fake_pump"]

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket",
           # 标签泄漏列 (来自 labels_v2, 是未来 20 天的真实数据)
           "label","y","max_gain_20","max_dd_20","entry_open",
           "ma5_max_consec_down","early_peak_pct","after_peak_low_pct"}


def load_window(start, end, labels_df):
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full = full.merge(labels_df, on=["ts_code","trade_date"], how="left").dropna(subset=["label"])
    full["y"] = full["label"].map(CLASS_MAP)

    # 各种 feature merge
    for path in [
        ROOT / "output" / "regimes" / "daily_regime.parquet",
        ROOT / "output" / "amount_features" / "amount_features.parquet",
        ROOT / "output" / "regime_extra" / "regime_extra.parquet",
        ROOT / "output" / "moneyflow" / "features.parquet",
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if path.name == "daily_regime.parquet":
            d = d.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                    "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14",
                                    "vol_ratio":"mkt_vol_ratio"})
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")
    return full


def main():
    t0 = time.time()
    print("加载多分类标签...")
    labels = pd.read_parquet(LABELS_V2,
                              columns=["ts_code","trade_date","label","max_gain_20","max_dd_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)

    print("加载训练集 + 测试集...")
    train = load_window(TRAIN_START, TRAIN_END, labels).dropna(subset=["r20"])
    test  = load_window(TEST_START, TEST_END, labels).dropna(subset=["r20"])
    print(f"Train: {len(train):,} 行")
    print(f"Test:  {len(test):,} 行")
    print()
    print("训练集类别分布:")
    print(train["label"].value_counts().to_string())

    # 特征
    feat_cols = [c for c in train.columns
                 if c not in EXCLUDE and c not in ("label","y")
                 and pd.api.types.is_numeric_dtype(train[c])]
    industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    feat_cols.append("industry_id")
    print(f"\n特征数: {len(feat_cols)}")

    X_train = train[feat_cols]; y_train = train["y"].astype(int)
    X_test  = test[feat_cols];  y_test  = test["y"].astype(int)

    print(f"\n训练 LightGBM 多分类 (6 类)...")
    clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=200, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        objective="multiclass", num_class=len(CLASS_NAMES),
        random_state=42, n_jobs=-1, verbose=-1,
        class_weight="balanced",  # 处理不平衡
    )
    clf.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)])

    # 立即保存模型 (避免后续报告失败导致模型丢失)
    print(f"\n[早保存] 写出模型 + meta...")
    clf.booster_.save_model(str(OUT_DIR / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    weights = np.array([0, -0.3, 1.0, 1.5, 2.0, -2.0])
    Path(OUT_DIR / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "class_map": CLASS_MAP, "class_names": CLASS_NAMES,
        "weights": weights.tolist(),
        "buy_score_range": [-2.0, 6.0],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[早保存] 完成: {OUT_DIR}")

    proba = clf.predict_proba(X_test)
    pred = proba.argmax(axis=1)

    try:
        print(f"\n=== OOS 多分类报告 ===")
        print(f"best_iter: {clf.best_iteration_}")
        print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3))
    except Exception as e:
        print(f"分类报告失败: {e}")

    # 各类的 ROC-AUC (one-vs-rest)
    print(f"\n=== 各类 ROC-AUC (one-vs-rest) ===")
    for i, name in enumerate(CLASS_NAMES):
        y_bin = (y_test == i).astype(int)
        if y_bin.sum() < 100: continue
        auc = roc_auc_score(y_bin, proba[:, i])
        print(f"  {name:<12}: AUC = {auc:.4f}, 正例占比 {y_bin.mean()*100:.2f}%")

    # 综合 buy_score
    # 权重: agg=2, strong=1.5, mild=1, fake_pump=-2, sideways=-0.3, other=0
    weights = np.array([0, -0.3, 1.0, 1.5, 2.0, -2.0])  # 对应 0~5 类
    test = test.copy()
    test["buy_score_raw"] = proba @ weights
    # 归一到 0-100 (用 OOS 的实际范围)
    s_min, s_max = test["buy_score_raw"].min(), test["buy_score_raw"].max()
    test["buy_score"] = (test["buy_score_raw"] - s_min) / (s_max - s_min) * 100

    # 5 分位看 max_gain
    print(f"\n=== buy_score 5 分位 vs 真实 max_gain ===")
    test["score_q"] = test.groupby("trade_date")["buy_score"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1)
    summary = test.groupby("score_q").agg(
        score_avg=("buy_score","mean"),
        max_gain_avg=("max_gain_20","mean"),
        max_gain_med=("max_gain_20","median"),
        gain_15=("max_gain_20", lambda x: (x>=15).mean()*100),
        gain_20=("max_gain_20", lambda x: (x>=20).mean()*100),
        gain_30=("max_gain_20", lambda x: (x>=30).mean()*100),
        n=("buy_score","count"),
    ).round(2)
    print(summary.to_string())

    # 按分数实际阈值切分
    print(f"\n=== 按 buy_score 实际阈值切分 ===")
    print(f"{'区间':<14} {'n':>10} {'max_gain 均值':>12} {'≥15%':>7} {'≥20%':>7} {'≥30%':>7}")
    for low, high in [(0, 30), (30, 50), (50, 70), (70, 80), (80, 90), (90, 101)]:
        mask = (test["buy_score"] >= low) & (test["buy_score"] < high)
        sub = test[mask]
        if len(sub) == 0: continue
        avg_g = sub["max_gain_20"].mean()
        g15 = (sub["max_gain_20"]>=15).mean()*100
        g20 = (sub["max_gain_20"]>=20).mean()*100
        g30 = (sub["max_gain_20"]>=30).mean()*100
        print(f"[{low:>3}, {high:>3})  {len(sub):>10,} {avg_g:>11.2f}% {g15:>6.1f}% {g20:>6.1f}% {g30:>6.1f}%")

    # Lift (top X% 真实 aggressive 比例)
    print(f"\n=== Lift 分析 ===")
    base_agg_rate = (y_test == 4).mean()  # aggressive 基线
    sorted_test = test.sort_values("buy_score_raw", ascending=False)
    print(f"基线 aggressive 比例: {base_agg_rate*100:.2f}%")
    for pct in [0.005, 0.01, 0.02, 0.05, 0.10]:
        n = int(len(sorted_test) * pct)
        top = sorted_test.head(n)
        agg_rate = (top["label"] == "aggressive").mean()
        strong_rate = (top["label"] == "strong").mean()
        mild_rate = (top["label"] == "mild").mean()
        positive_rate = ((top["label"] == "aggressive") | (top["label"] == "strong") | (top["label"] == "mild")).mean()
        avg_gain = top["max_gain_20"].mean()
        print(f"  Top {pct*100:>4.1f}% (n={n:>5}): "
              f"aggressive {agg_rate*100:>4.1f}% (lift {agg_rate/base_agg_rate:.1f}x), "
              f"3类合计 {positive_rate*100:>4.1f}%, max_gain 均值 {avg_gain:.2f}%")

    # 保存
    clf.booster_.save_model(str(OUT_DIR / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.cat.categories,
                                                    range(len(industries.cat.categories)))}
    Path(OUT_DIR / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "class_map": CLASS_MAP, "class_names": CLASS_NAMES,
        "weights": weights.tolist(),
        "buy_score_range": [float(s_min), float(s_max)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n模型: {OUT_DIR}, 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
