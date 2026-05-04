#!/usr/bin/env python3
"""V4 全面对比: 6 配置 × 2 horizon = 12 模型 IC 对比.

配置:
  ALL              : 现有 203 (V2 baseline)
  ALL+v2           : +8 f1-f8 = 211
  ALL+mfk          : +17 mfk = 220
  ALL+v2+mfk       : 全部 228
  only_v2          : base+rextra+v2 = 184
  only_mfk         : base+rextra+mfk = 193

输出: output/v4_compare/results.csv
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
OUT_DIR = ROOT / "output" / "v4_compare"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_START = "20230101"
TRAIN_END   = "20250430"
TEST_START  = "20250501"
TEST_END    = "20260126"

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket",
           "max_gain_10","max_dd_10","max_gain_20","max_dd_20","entry_open"}


def load_window(start, end, with_full_v1=True, with_v2=False, with_mfk=False):
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10","max_dd_10"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    l20 = pd.read_parquet(LABELS_20, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")

    if with_full_v1:
        for path in [
            ROOT / "output" / "amount_features" / "amount_features.parquet",
            ROOT / "output" / "moneyflow" / "features.parquet",
            ROOT / "output" / "cogalpha_features" / "features.parquet",
        ]:
            if not path.exists(): continue
            d = pd.read_parquet(path)
            if "trade_date" in d.columns:
                d["trade_date"] = d["trade_date"].astype(str)
            if "ts_code" in d.columns:
                full = full.merge(d, on=["ts_code","trade_date"], how="left")

    # 始终加 rextra
    p = ROOT / "output" / "regime_extra" / "regime_extra.parquet"
    if p.exists():
        d = pd.read_parquet(p); d["trade_date"] = d["trade_date"].astype(str)
        full = full.merge(d, on="trade_date", how="left")

    if with_v2:
        p = ROOT / "output" / "moneyflow_v2" / "features.parquet"
        if p.exists():
            d = pd.read_parquet(p); d["trade_date"] = d["trade_date"].astype(str)
            full = full.merge(d, on=["ts_code","trade_date"], how="left")

    if with_mfk:
        p = ROOT / "output" / "mfk_features" / "features.parquet"
        if p.exists():
            d = pd.read_parquet(p); d["trade_date"] = d["trade_date"].astype(str)
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


def train_eval(label_col, train, test, feat_cols, industries, name,
                train_subsample=600_000, test_subsample=300_000):
    # 不 copy, 只取必要列, 直接做转换
    use_cols = feat_cols + ["industry", label_col, "trade_date"]
    use_cols = list(set(use_cols))
    train = train[use_cols].dropna(subset=[label_col]).copy()
    test  = test[use_cols].dropna(subset=[label_col]).copy()
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.cat.categories).codes
    feat_cols_use = list(feat_cols)
    if "industry_id" not in feat_cols_use: feat_cols_use.append("industry_id")
    # 降采样: train 600K, test 300K (保留分时分布)
    if len(train) > train_subsample:
        train = train.sample(n=train_subsample, random_state=42).reset_index(drop=True)
    if len(test) > test_subsample:
        test = test.sample(n=test_subsample, random_state=42).reset_index(drop=True)

    X_train = train[feat_cols_use].astype("float32")
    y_train = train[label_col].astype("float32")
    X_test  = test[feat_cols_use].astype("float32")
    y_test  = test[label_col].astype("float32")
    feat_cols = feat_cols_use

    clf = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        min_child_samples=300, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
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

    # 重要性: v2/mfk 因子贡献
    imp = pd.Series(clf.booster_.feature_importance(importance_type="gain"),
                     index=feat_cols)
    v2_imp = imp[imp.index.str.startswith("f") &
                  imp.index.str.split("_").str[0].isin(["f1","f2","f3","f4","f5","f6","f7","f8"])].sum()
    mfk_imp = imp[imp.index.str.startswith("mfk_")].sum()
    metrics["v2_importance_pct"] = float(v2_imp / imp.sum() * 100) if imp.sum() > 0 else 0
    metrics["mfk_importance_pct"] = float(mfk_imp / imp.sum() * 100) if imp.sum() > 0 else 0

    return clf, metrics, y_pred


def main():
    t0 = time.time()
    configs = [
        ("ALL",          dict(with_full_v1=True, with_v2=False, with_mfk=False)),
        ("ALL+v2",       dict(with_full_v1=True, with_v2=True,  with_mfk=False)),
        ("ALL+mfk",      dict(with_full_v1=True, with_v2=False, with_mfk=True)),
        ("ALL+v2+mfk",   dict(with_full_v1=True, with_v2=True,  with_mfk=True)),
        ("only_v2",      dict(with_full_v1=False, with_v2=True, with_mfk=False)),
        ("only_mfk",     dict(with_full_v1=False, with_v2=False, with_mfk=True)),
    ]

    results = []
    # 增量保存 + 内存释放
    csv_path = OUT_DIR / "results.csv"
    for cfg_name, cfg in configs:
        print(f"\n=== {cfg_name} ===", flush=True)
        try:
            train = load_window(TRAIN_START, TRAIN_END, **cfg)
            test  = load_window(TEST_START, TEST_END, **cfg)
        except Exception as e:
            print(f"  load 失败: {e}", flush=True)
            gc.collect(); continue
        industries = pd.concat([train["industry"], test["industry"]]).fillna("unknown").astype("category")
        feat_cols = [c for c in train.columns
                      if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c])]
        print(f"  Train: {len(train):,}, Test: {len(test):,}, feat: {len(feat_cols)}", flush=True)

        for label in ["r10", "r20"]:
            try:
                clf, m, _ = train_eval(label, train, test, feat_cols, industries, cfg_name)
                m["horizon"] = label
                results.append(m)
                print(f"  {label}: IC={m['IC']:+.4f} RankIC={m['RankIC']:+.4f} "
                      f"RankICIR={m['RankICIR']:+.3f} best_iter={m['best_iter']} "
                      f"v2={m.get('v2_importance_pct',0):.1f}% mfk={m.get('mfk_importance_pct',0):.1f}%",
                      flush=True)
                pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
                del clf
                gc.collect()
            except Exception as e:
                print(f"  {label}: 失败 {e}", flush=True)
                gc.collect()
        del train, test, industries
        gc.collect()

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "results.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*110}")
    print(f"=== 全面对比 (vs CogAlpha SOTA: r10 IC=0.0591/RankICIR=0.435) ===")
    print(f"{'='*110}")
    print(f"{'horizon':<6} {'config':<14} {'IC':>+8} {'RankIC':>+8} {'ICIR':>+8} {'RankICIR':>+10} "
          f"{'best_iter':>10} {'feat':>5} {'v2%':>5} {'mfk%':>6}")
    for _, r in res_df.iterrows():
        print(f"{r['horizon']:<6} {r['config']:<14} {r['IC']:>+8.4f} {r['RankIC']:>+8.4f} "
              f"{r['ICIR']:>+8.3f} {r['RankICIR']:>+10.3f} "
              f"{int(r['best_iter']):>10} {int(r['n_features']):>5} "
              f"{r.get('v2_importance_pct', 0):>5.1f} {r.get('mfk_importance_pct', 0):>6.1f}")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
