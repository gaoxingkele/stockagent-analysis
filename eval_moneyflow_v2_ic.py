#!/usr/bin/env python3
"""V3 IC 评估: ALL+v2 vs ALL (在 dual_horizon 框架上加 moneyflow_v2 配置).

对比 4 配置 × 2 horizon = 8 个模型:
  base+rextra        : 158 特征 (验证 V2 的 IC=+0.069 baseline)
  base+rextra+v2     : 166 特征 (V2 因子单独贡献)
  ALL                : 203 特征 (现有生产配置)
  ALL+v2             : 211 特征 (V3 候选)

只需要看 r10/r20 ALL 是否突破 0.073/0.034 IC 天花板, RankICIR 是否突破 0.49/0.84.

输出: output/moneyflow_v2/ic_eval.csv
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
V2_FEAT = ROOT / "output" / "moneyflow_v2" / "features.parquet"
OUT_DIR = ROOT / "output" / "moneyflow_v2"

TRAIN_START = "20230101"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket",
           "max_gain_10","max_dd_10","entry_open"}


def load_window(start, end, with_amount=True, with_rextra=True,
                  with_moneyflow=True, with_cogalpha=True, with_v2=False):
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    labels = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left")

    if with_amount:
        p = ROOT / "output" / "amount_features" / "amount_features.parquet"
        if p.exists():
            d = pd.read_parquet(p); d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
    if with_rextra:
        p = ROOT / "output" / "regime_extra" / "regime_extra.parquet"
        if p.exists():
            d = pd.read_parquet(p); d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on="trade_date", how="left")
    if with_moneyflow:
        p = ROOT / "output" / "moneyflow" / "features.parquet"
        if p.exists():
            d = pd.read_parquet(p); d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
    if with_cogalpha:
        p = ROOT / "output" / "cogalpha_features" / "features.parquet"
        if p.exists():
            d = pd.read_parquet(p); d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
    if with_v2:
        if V2_FEAT.exists():
            d = pd.read_parquet(V2_FEAT); d["trade_date"] = d["trade_date"].astype(str)
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


def compute_metrics(y_true, y_pred, dates):
    ic = stats.pearsonr(y_pred, y_true)[0]
    rank_ic = stats.spearmanr(y_pred, y_true)[0]
    monthly = pd.DataFrame({"y": y_true, "p": y_pred,
                              "ym": pd.Series(dates).astype(str).str[:6].values})
    m = monthly.groupby("ym").apply(
        lambda g: pd.Series({
            "ic": stats.pearsonr(g["p"], g["y"])[0] if len(g) >= 30 else np.nan,
            "rank_ic": stats.spearmanr(g["p"], g["y"])[0] if len(g) >= 30 else np.nan,
        })
    ).dropna()
    ic_ir = m["ic"].mean() / m["ic"].std() if m["ic"].std() > 0 else 0
    rank_ic_ir = m["rank_ic"].mean() / m["rank_ic"].std() if m["rank_ic"].std() > 0 else 0
    return {"IC": ic, "RankIC": rank_ic, "ICIR": ic_ir, "RankICIR": rank_ic_ir}


def train_eval(label_col, train, test, feat_cols, industries, name):
    train = train.copy(); test = test.copy()
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    if "industry_id" not in feat_cols: feat_cols = feat_cols + ["industry_id"]
    train = train.dropna(subset=[label_col])
    test = test.dropna(subset=[label_col])

    X_train = train[feat_cols]; y_train = train[label_col]
    X_test  = test[feat_cols];  y_test  = test[label_col]

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
    metrics = compute_metrics(y_test.values, y_pred, test["trade_date"].values)
    metrics["best_iter"] = int(clf.best_iteration_)
    metrics["n_features"] = len(feat_cols)
    metrics["config"] = name

    # 单因子 IC: 看 V2 因子在新模型里的重要性
    importance = pd.Series(clf.booster_.feature_importance(importance_type="gain"),
                            index=feat_cols).sort_values(ascending=False)
    v2_cols = [c for c in feat_cols if c.startswith("f") and "_" in c
                and c.split("_")[0] in {"f1","f2","f3","f4","f5","f6","f7","f8"}]
    metrics["v2_top_importance"] = ",".join([f"{c}:{int(importance.get(c, 0))}"
                                              for c in v2_cols[:8]]) if v2_cols else ""

    return metrics


def main():
    if not V2_FEAT.exists():
        print(f"❌ V2 特征文件未找到: {V2_FEAT}")
        print(f"  请先跑 extract_moneyflow_v2.py")
        return

    t0 = time.time()
    print("加载 V2 特征 + 验证完整性...")
    v2_df = pd.read_parquet(V2_FEAT)
    print(f"  V2: {len(v2_df):,} 行 × {v2_df['ts_code'].nunique()} 股")
    print(f"  V2 因子分布:")
    feat_cols = [c for c in v2_df.columns if c.startswith("f")]
    print(v2_df[feat_cols].describe(percentiles=[0.05,0.5,0.95]).round(4).to_string())

    configs = [
        ("ALL",      dict(with_amount=True, with_rextra=True,
                            with_moneyflow=True, with_cogalpha=True, with_v2=False)),
        ("ALL+v2",   dict(with_amount=True, with_rextra=True,
                            with_moneyflow=True, with_cogalpha=True, with_v2=True)),
        ("base+rextra+v2",
                     dict(with_amount=False, with_rextra=True,
                            with_moneyflow=False, with_cogalpha=False, with_v2=True)),
    ]

    results = []
    for cfg_name, cfg in configs:
        print(f"\n=== Config: {cfg_name} ===")
        train = load_window(TRAIN_START, TRAIN_END, **cfg)
        test  = load_window(TEST_START, TEST_END, **cfg)
        industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
        feat_cols = [c for c in train.columns
                      if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
        print(f"  Train: {len(train):,}, Test: {len(test):,}, Features: {len(feat_cols)}")

        for label_col in ["r10", "r20"]:
            print(f"  --- {label_col} ---")
            try:
                metrics = train_eval(label_col, train, test, feat_cols, industries, cfg_name)
                metrics["horizon"] = label_col
                results.append(metrics)
                print(f"    IC={metrics['IC']:+.4f}  RankIC={metrics['RankIC']:+.4f}  "
                      f"RankICIR={metrics['RankICIR']:+.3f}  best_iter={metrics['best_iter']}  feat={metrics['n_features']}")
                if metrics.get("v2_top_importance"):
                    print(f"    V2 importance: {metrics['v2_top_importance']}")
            except Exception as e:
                print(f"    失败: {e}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "ic_eval.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*100}")
    print(f"=== 汇总 (vs ALL baseline: r10 IC=0.0726/RankICIR=0.489, r20 IC=0.0339/RankICIR=0.843) ===")
    print(f"{'='*100}")
    print(f"{'horizon':<8} {'config':<20} {'IC':>+8} {'RankIC':>+8} {'ICIR':>+8} {'RankICIR':>+10} {'best_iter':>10}")
    for _, r in res_df.iterrows():
        print(f"{r['horizon']:<8} {r['config']:<20} {r['IC']:>+8.4f} {r['RankIC']:>+8.4f} "
              f"{r['ICIR']:>+8.3f} {r['RankICIR']:>+10.3f} {int(r['best_iter']):>10}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
