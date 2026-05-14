"""V15 重训: 用最新数据 (到 20260413) 重训 r10/r20 模型.

变更 vs V4:
  - 训练区间扩展到 20260213 (留 60 日 OOS 至 20260413)
  - 加载 _ext / _ext2 扩展数据 (factor_lab + labels + 5 个 feature)
  - 只训 r10/r20 (sell 已屏蔽, 不再训)
  - 输出 r10_v15_all / r20_v15_all 不覆盖 V4

输出: output/production/{r10_v15_all, r20_v15_all}/
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "output" / "production"
OUT_BASE.mkdir(exist_ok=True)

TRAIN_START = "20230101"
TRAIN_END   = "20260213"
TEST_START  = "20260214"
TEST_END    = "20260413"

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket",
           "max_gain_10","max_dd_10","max_gain_20","max_dd_20","entry_open","is_clean"}


def _load_concat(main_path: Path, ext_globs: list, columns=None):
    parts = []
    if main_path.exists():
        d = pd.read_parquet(main_path, columns=columns) if columns else pd.read_parquet(main_path)
        parts.append(d)
    for g in ext_globs:
        for f in sorted(ROOT.glob(g)):
            d = pd.read_parquet(f, columns=columns) if columns else pd.read_parquet(f)
            parts.append(d)
    if not parts: return None
    big = pd.concat(parts, ignore_index=True)
    if "trade_date" in big.columns:
        big["trade_date"] = big["trade_date"].astype(str)
    if "ts_code" in big.columns and "trade_date" in big.columns:
        big = big.drop_duplicates(subset=["ts_code","trade_date"], keep="last")
    elif "trade_date" in big.columns:
        big = big.drop_duplicates(subset=["trade_date"], keep="last")
    return big


def load_window(start, end, with_mfk=False):
    print(f"  load factor_lab (主 + ext + ext2)...", flush=True)
    parts = []
    for p in sorted((ROOT / "output/factor_lab_3y/factor_groups").glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    for p in sorted((ROOT / "output/factor_lab_3y/factor_groups_extension").glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["ts_code","trade_date"], keep="last").reset_index(drop=True)
    # factor_lab 主 parquet 含 forward label r5/r10/r20/r30/r40/dd*, 必须删除
    forward_labels = ["r5","r10","r20","r30","r40","dd5","dd10","dd20","dd30","dd40",
                       "max_gain_10","max_dd_10","max_gain_20","max_dd_20","entry_open","is_clean"]
    drops = [c for c in forward_labels if c in full.columns]
    if drops:
        print(f"    删除 factor_lab 内嵌的 forward labels ({len(drops)}): {drops}", flush=True)
        full = full.drop(columns=drops)
    print(f"    factor: {len(full):,} 行 / {full['ts_code'].nunique()} 股", flush=True)

    print(f"  load labels (主 + ext)...", flush=True)
    l10 = _load_concat(
        ROOT / "output/cogalpha_features/labels_10d.parquet",
        ["output/cogalpha_features/labels_10d_ext*.parquet"],
        columns=["ts_code","trade_date","r10","max_dd_10"],
    )
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    l20 = _load_concat(
        ROOT / "output/labels/max_gain_labels.parquet",
        ["output/labels/max_gain_labels_ext*.parquet"],
        columns=["ts_code","trade_date","max_gain_20","max_dd_20","r20_close"],
    )
    full = full.merge(l20.rename(columns={"r20_close":"r20"}), on=["ts_code","trade_date"], how="left")

    print(f"  load features (5 个, 主 + ext_*)...", flush=True)
    feature_dirs = [
        ("amount_features", "amount_features"),
        ("moneyflow", "features"),
        ("cogalpha_features", "features"),
        ("pyramid_v2", "features"),
        ("v7_extras", "features"),
    ]
    if with_mfk:
        feature_dirs.append(("mfk_features", "features"))
    for d, stem in feature_dirs:
        main_p = ROOT / "output" / d / f"{stem}.parquet"
        ext_glob = f"output/{d}/{stem}_ext*.parquet"
        feat = _load_concat(main_p, [ext_glob])
        if feat is None: continue
        # 重命名 feat 内与 full 冲突的列, 避免 pandas 添加 _x/_y 后缀污染 label
        keys = {"ts_code","trade_date"}
        rename_map = {c: f"{d}_{c}" for c in feat.columns
                       if c in full.columns and c not in keys}
        if rename_map:
            feat = feat.rename(columns=rename_map)
        if "ts_code" in feat.columns and "trade_date" in feat.columns:
            full = full.merge(feat, on=["ts_code","trade_date"], how="left")
        elif "trade_date" in feat.columns:
            full = full.merge(feat, on="trade_date", how="left")

    # Sanity: 修复 r20_x/r20_y 这种意外后缀, 保留 label r20
    if "r20" not in full.columns:
        if "r20_x" in full.columns:
            # r20_x 是先 merge 进的 label
            full = full.rename(columns={"r20_x": "r20"})
        if "r20_y" in full.columns:
            full = full.drop(columns=["r20_y"])
    # 强制删除所有 feature 里可能 forward-leaking 的列 (r10/r20/max_gain/max_dd 各种后缀)
    forbidden_substrings = ["_r10", "_r20", "_max_gain", "_max_dd", "r10_close", "r20_close"]
    drops = [c for c in full.columns
             if any(s in c for s in forbidden_substrings) and c not in ("r10","r20")]
    if drops:
        print(f"    drop forbidden cols ({len(drops)}): {drops[:5]}...", flush=True)
        full = full.drop(columns=drops)

    print(f"  load regime + regime_extra...", flush=True)
    rg_path = ROOT / "output/regimes/daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path, columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        rename_map = {c: f"regime_{c}" for c in rg.columns
                       if c in full.columns and c != "trade_date"}
        if rename_map: rg = rg.rename(columns=rename_map)
        full = full.merge(rg, on="trade_date", how="left")
    rgx_path = ROOT / "output/regime_extra/regime_extra.parquet"
    if rgx_path.exists():
        rgx = pd.read_parquet(rgx_path)
        rgx["trade_date"] = rgx["trade_date"].astype(str)
        rename_map = {c: f"rgx_{c}" for c in rgx.columns
                       if c in full.columns and c != "trade_date"}
        if rename_map: rgx = rgx.rename(columns=rename_map)
        full = full.merge(rgx, on="trade_date", how="left")
    return full


def spearman_ic(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    return "spearman_ic", ic if not np.isnan(ic) else 0.0, True


def train_regressor(name, train, test, feat_cols, industries, y_col,
                     train_subsample=1_500_000):
    out_dir = OUT_BASE / name
    out_dir.mkdir(exist_ok=True)
    train = train[feat_cols + ["industry", y_col, "trade_date"]].dropna(subset=[y_col]).copy()
    test = test[feat_cols + ["industry", y_col, "trade_date"]].dropna(subset=[y_col]).copy()
    train["industry_id"] = pd.Categorical(train["industry"].fillna("unknown"),
                                           categories=industries.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                          categories=industries.categories).codes
    feat_cols = list(feat_cols)
    if "industry_id" not in feat_cols: feat_cols.append("industry_id")
    if len(train) > train_subsample:
        train = train.sample(n=train_subsample, random_state=42).reset_index(drop=True)
    print(f"  [{name}] train={len(train):,} test={len(test):,} feat={len(feat_cols)}", flush=True)

    X_train = train[feat_cols].astype("float32"); y_train = train[y_col].astype("float32")
    X_test  = test[feat_cols].astype("float32");  y_test  = test[y_col].astype("float32")

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
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.categories,
                                                    range(len(industries.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": y_col, "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    y_pred = clf.predict(X_test)
    ic = stats.pearsonr(y_pred, y_test)[0]
    rank_ic = stats.spearmanr(y_pred, y_test)[0]
    monthly = pd.DataFrame({"y": y_test.values, "p": y_pred,
                              "ym": test["trade_date"].astype(str).str[:6].values})
    m_ic = monthly.groupby("ym").apply(
        lambda g: stats.spearmanr(g["p"], g["y"])[0] if len(g) >= 30 else np.nan
    ).dropna()
    rank_ic_ir = m_ic.mean() / m_ic.std() if m_ic.std() > 0 else 0
    p5 = float(np.quantile(y_pred, 0.05)); p50 = float(np.quantile(y_pred, 0.50)); p95 = float(np.quantile(y_pred, 0.95))
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic": float(ic), "rank_ic": float(rank_ic),
        "rank_ic_ir": float(rank_ic_ir),
        "n_train": len(train), "n_test": len(test),
        "n_features": len(feat_cols),
        "version": "v15",
        "train_window": f"{TRAIN_START}-{TRAIN_END}",
        "test_window": f"{TEST_START}-{TEST_END}",
        "anchor_p5": p5, "anchor_p50": p50, "anchor_p95": p95,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [{name}] IC={ic:.4f} RankIC={rank_ic:.4f} RankICIR={rank_ic_ir:.3f} 锚={p5:.2f}/{p50:.2f}/{p95:.2f}", flush=True)
    del clf, X_train, X_test, y_train, y_test
    gc.collect()


def main():
    t0 = time.time()
    print("=== V15 重训 (含 V4 全部扩展数据) ===\n", flush=True)
    print(f"训练区间: {TRAIN_START} → {TRAIN_END}", flush=True)
    print(f"OOS 区间: {TEST_START} → {TEST_END} (60 日 OOS)\n", flush=True)

    print("[1/3] 加载训练集 + mfk...", flush=True)
    train_df = load_window(TRAIN_START, TRAIN_END, with_mfk=True)
    print(f"  训练集: {len(train_df):,} 行\n", flush=True)

    print("[2/3] 加载 OOS 集 + mfk...", flush=True)
    test_df = load_window(TEST_START, TEST_END, with_mfk=True)
    print(f"  OOS 集: {len(test_df):,} 行\n", flush=True)

    # industries (从训练集生成)
    all_ind = pd.concat([train_df["industry"], test_df["industry"]]).fillna("unknown").astype(str)
    industries = pd.Categorical(all_ind, categories=sorted(all_ind.unique()))

    # 特征列
    feat_cols = [c for c in train_df.columns
                  if c not in EXCLUDE
                  and pd.api.types.is_numeric_dtype(train_df[c])]
    print(f"特征列数: {len(feat_cols)}", flush=True)

    print(f"\n[3/3] 训练 r10/r20 模型...\n", flush=True)
    train_regressor("r10_v15_all", train_df, test_df, feat_cols, industries, "r10")
    train_regressor("r20_v15_all", train_df, test_df, feat_cols, industries, "r20")

    print(f"\n=== 完成, 总耗时 {time.time()-t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
