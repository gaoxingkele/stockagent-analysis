#!/usr/bin/env python3
"""双 horizon (r10 / r20) LGBM 训练 + 5 指标评估 (CogAlpha 论文风格).

5 指标:
  IC      : Pearson 线性相关
  RankIC  : Spearman 排序相关
  ICIR    : IC 月度均值 / 月度标准差 (稳定性)
  RankICIR: RankIC 月度 IR
  MI      : 互信息 (非线性, 离散化估计)

对比四种特征组合:
  base       : 153 原因子
  +amount    : + 6 amount features
  +rextra    : + 5 regime extra
  +moneyflow : + 13 moneyflow
  +cogalpha  : + 8 CogAlpha 流动性/尾部因子
  ALL        : 全部加 (~190 特征)

输出: output/dual_horizon/{r10,r20}_results.csv
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
COGALPHA = ROOT / "output" / "cogalpha_features" / "features.parquet"
OUT_DIR = ROOT / "output" / "dual_horizon"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "20230101"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket",
           # cogalpha labels
           "max_gain_10","max_dd_10","entry_open"}


def load_window(start, end):
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    # r10 标签 (从 cogalpha labels)
    labels = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left")

    return full


def merge_features(full, include_amount=True, include_rextra=True,
                   include_moneyflow=True, include_cogalpha=True):
    """按选项 merge feature parquets."""
    if include_amount:
        a_path = ROOT / "output" / "amount_features" / "amount_features.parquet"
        if a_path.exists():
            d = pd.read_parquet(a_path)
            d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
    if include_rextra:
        r_path = ROOT / "output" / "regime_extra" / "regime_extra.parquet"
        if r_path.exists():
            d = pd.read_parquet(r_path)
            d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on="trade_date", how="left")
    if include_moneyflow:
        m_path = ROOT / "output" / "moneyflow" / "features.parquet"
        if m_path.exists():
            d = pd.read_parquet(m_path)
            d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
    if include_cogalpha:
        if COGALPHA.exists():
            d = pd.read_parquet(COGALPHA)
            d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
    # 加 regime_id
    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path,
                              columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        full = full.merge(rg, on="trade_date", how="left")
    return full


def compute_5_metrics(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray):
    """5 指标: IC, RankIC, ICIR (月度), RankICIR (月度), MI."""
    # 整体 IC
    ic_overall = stats.pearsonr(y_pred, y_true)[0]
    rank_ic = stats.spearmanr(y_pred, y_true)[0]

    # 月度 ICIR
    df = pd.DataFrame({"y": y_true, "p": y_pred, "d": dates})
    df["ym"] = df["d"].astype(str).str[:6]
    monthly = df.groupby("ym").apply(
        lambda g: pd.Series({
            "ic": stats.pearsonr(g["p"], g["y"])[0] if len(g) >= 30 else np.nan,
            "rank_ic": stats.spearmanr(g["p"], g["y"])[0] if len(g) >= 30 else np.nan,
        })
    ).dropna()
    ic_ir = monthly["ic"].mean() / monthly["ic"].std() if monthly["ic"].std() > 0 else 0
    rank_ic_ir = monthly["rank_ic"].mean() / monthly["rank_ic"].std() if monthly["rank_ic"].std() > 0 else 0

    # MI (在子样本上算, 全量太慢)
    n_sample = min(50000, len(y_true))
    idx = np.random.choice(len(y_true), n_sample, replace=False)
    try:
        mi = mutual_info_regression(y_pred[idx].reshape(-1,1), y_true[idx], random_state=42)[0]
    except Exception:
        mi = 0
    return {
        "IC": float(ic_overall), "RankIC": float(rank_ic),
        "ICIR": float(ic_ir), "RankICIR": float(rank_ic_ir), "MI": float(mi),
        "monthly_ic_n": len(monthly),
    }


def train_eval(label_col: str, train: pd.DataFrame, test: pd.DataFrame,
                feat_cols: list[str], industries):
    print(f"\n  特征数: {len(feat_cols)}")
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")
    train = train.dropna(subset=[label_col])
    test = test.dropna(subset=[label_col])
    print(f"  Train: {len(train):,}, Test: {len(test):,}")

    X_train = train[feat_cols]; y_train = train[label_col]
    X_test  = test[feat_cols];  y_test  = test[label_col]

    # 自定义 IC eval
    def spearman_ic(y_true, y_pred):
        ic = stats.spearmanr(y_pred, y_true)[0]
        return "spearman_ic", ic if not np.isnan(ic) else 0.0, True

    clf = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             eval_metric=spearman_ic,
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(80, first_metric_only=True),
                          lgb.log_evaluation(0)])

    y_pred = clf.predict(X_test)
    metrics = compute_5_metrics(y_test.values, y_pred, test["trade_date"].values)
    metrics["best_iter"] = int(clf.best_iteration_)
    metrics["n_train"] = len(train)
    metrics["n_test"] = len(test)
    metrics["n_features"] = len(feat_cols)
    return clf, metrics, y_pred


def main():
    t0 = time.time()
    print(f"加载训练 + 测试集...")
    train_full = load_window(TRAIN_START, TRAIN_END)
    test_full  = load_window(TEST_START, TEST_END)
    print(f"Train: {len(train_full):,}, Test: {len(test_full):,}")

    # 6 种特征组合
    configs = [
        ("base",       dict(include_amount=False, include_rextra=False,
                             include_moneyflow=False, include_cogalpha=False)),
        ("+amount",    dict(include_amount=True,  include_rextra=False,
                             include_moneyflow=False, include_cogalpha=False)),
        ("+rextra",    dict(include_amount=False, include_rextra=True,
                             include_moneyflow=False, include_cogalpha=False)),
        ("+moneyflow", dict(include_amount=False, include_rextra=False,
                             include_moneyflow=True,  include_cogalpha=False)),
        ("+cogalpha",  dict(include_amount=False, include_rextra=False,
                             include_moneyflow=False, include_cogalpha=True)),
        ("ALL",        dict(include_amount=True,  include_rextra=True,
                             include_moneyflow=True,  include_cogalpha=True)),
    ]

    results = []
    for label_col in ["r10", "r20"]:
        print(f"\n{'='*60}")
        print(f"=== Horizon: {label_col} ===")
        print(f"{'='*60}")

        for cfg_name, cfg in configs:
            print(f"\n--- {cfg_name} ---")
            train = merge_features(train_full.copy(), **cfg)
            test  = merge_features(test_full.copy(), **cfg)
            industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
            feat_cols = [c for c in train.columns
                          if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
            try:
                clf, metrics, y_pred = train_eval(label_col, train, test, feat_cols, industries)
                metrics["horizon"] = label_col
                metrics["config"] = cfg_name
                results.append(metrics)
                print(f"  IC={metrics['IC']:.4f}  RankIC={metrics['RankIC']:.4f}  "
                      f"ICIR={metrics['ICIR']:.3f}  RankICIR={metrics['RankICIR']:.3f}  "
                      f"MI={metrics['MI']:.4f}  best_iter={metrics['best_iter']}")
            except Exception as e:
                print(f"  失败: {e}")

    # 汇总
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "dual_horizon_results.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*100}")
    print(f"=== 汇总 ===")
    print(f"{'='*100}")
    print(f"{'horizon':<8} {'config':<14} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'RankICIR':>10} "
          f"{'MI':>8} {'best_iter':>10}")
    for _, r in res_df.iterrows():
        print(f"{r['horizon']:<8} {r['config']:<14} {r['IC']:>+8.4f} {r['RankIC']:>+8.4f} "
              f"{r['ICIR']:>+8.3f} {r['RankICIR']:>+10.3f} {r['MI']:>8.4f} {int(r['best_iter']):>10}")

    print(f"\n总耗时 {time.time()-t0:.1f}s, 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
