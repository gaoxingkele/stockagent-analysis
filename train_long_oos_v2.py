"""V12 v9 (OOS 扩窗) - 1H 模型重训, cut at 20260131.

vs train_long_oos.py:
  - TRAIN_END_DATE: 20250930 → 20260131 (扩 4 个月新数据)
  - OUT_NAME 加 _v2 后缀
  - 留 2026/2-5 (~70 日) 作新 OOS 验证

输出: r20_1h_v2_long_nost_v2, r1_next_open_v3_long_nost_v2
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROD = ROOT / "output" / "production"

TRAIN_END_DATE = "20260131"   # v9: 扩窗 4 个月
F2 = ROOT / "output" / "1h_factors" / "factors_v2.parquet"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"


def spearman_ic(y_true, y_pred):
    ic = stats.spearmanr(y_pred, y_true)[0]
    return "spearman_ic", ic if not np.isnan(ic) else 0.0, True


def train(src, label, out_name, exclude_cols, lr=0.03, leaves=63, patience=150):
    out_dir = PROD / out_name
    if (out_dir / "classifier.txt").exists():
        print(f"[{out_name}] 已存在, 跳过"); return

    print(f"\n=== 训 {out_name} (label={label}, cut at {TRAIN_END_DATE}) ===", flush=True)
    df = pd.read_parquet(src)
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"  {len(df):,} × {len(df.columns)}", flush=True)

    # ST 源头排除
    basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    if basic_p.exists():
        basic = pd.read_parquet(basic_p)[["ts_code", "name"]].drop_duplicates("ts_code")
        st_codes = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
        before = len(df)
        df = df[~df["ts_code"].isin(st_codes)].reset_index(drop=True)
        print(f"  ST 排除: {before - len(df):,} 行", flush=True)

    df = df.dropna(subset=[label])
    df = df[df[label].abs() <= 20].copy()
    train_df = df[df["trade_date"] < TRAIN_END_DATE]
    val = df[df["trade_date"] >= TRAIN_END_DATE]
    print(f"  train < {TRAIN_END_DATE}: {len(train_df):,}, val: {len(val):,}", flush=True)
    val_dates = val["trade_date"].unique()
    print(f"  val 日期数: {len(val_dates)}", flush=True)

    feat_cols = [c for c in df.columns
                  if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  特征列: {len(feat_cols)}", flush=True)

    for c in feat_cols:
        train_df[c] = train_df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
        val[c] = val[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    if len(train_df) > 2_500_000:
        train_df = train_df.sample(n=2_500_000, random_state=42).reset_index(drop=True)
        print(f"  subsample to 250 万", flush=True)

    X_train = train_df[feat_cols].astype("float32"); y_train = train_df[label].astype("float32")
    X_val = val[feat_cols].astype("float32"); y_val = val[label].astype("float32")

    clf = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=lr, num_leaves=leaves,
        min_child_samples=500, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
        objective="regression", metric="None",
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
             eval_metric=spearman_ic,
             callbacks=[lgb.early_stopping(patience, first_metric_only=True),
                          lgb.log_evaluation(200)])

    out_dir.mkdir(exist_ok=True, parents=True)
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "target": label, "model_type": "regressor",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    y_pred = clf.predict(X_val)
    ic = stats.pearsonr(y_pred, y_val)[0]
    rank_ic = stats.spearmanr(y_pred, y_val)[0]
    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "ic_val": float(ic), "rank_ic_val": float(rank_ic),
        "n_train": len(train_df), "n_val": len(val),
        "n_val_dates": len(val_dates),
        "n_features": len(feat_cols),
        "version": "long_oos_v2", "target": label,
        "train_window": f"< {TRAIN_END_DATE}",
        "anchor_p5": float(np.quantile(y_pred, 0.05)),
        "anchor_p50": float(np.quantile(y_pred, 0.50)),
        "anchor_p95": float(np.quantile(y_pred, 0.95)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {out_name} IC={ic:.4f} RankIC={rank_ic:.4f} best_iter={clf.best_iteration_}", flush=True)
    del clf, X_train, y_train, X_val, y_val, train_df, val
    gc.collect()


def main():
    t0 = time.time()
    print(f"=== 1H 长 OOS 重训 v2 (cut at {TRAIN_END_DATE}) ===\n")
    train(F2, "r20_1h", "r20_1h_v2_long_nost_v2",
          exclude_cols={"ts_code","trade_time","trade_date","r4_1h","r20_1h","r40_1h",
                          "ma5","ma10","ma20","ma60","vol_ma20","close"})
    train(F3, "r1_next_open", "r1_next_open_v3_long_nost_v2",
          exclude_cols={"ts_code","trade_time","trade_date",
                          "r4_1h","r20_1h","r40_1h",
                          "r1_next_open","r4_next_morn","r8_next_day"})
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
