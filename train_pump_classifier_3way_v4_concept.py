"""V4: 三分类 + 概念热度因子 (任务 2 收尾, 加入 8 新因子重训).

vs v3b:
  - 加 output/concept_heat/stock_features.parquet (8 概念热度因子)
  - 其他设置同 v3b (multiclass 无 balance)

输出: output/production/r5_pump_3way_lgbm_v4/
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window, EXCLUDE

PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"
CONCEPT_HEAT_P = ROOT / "output" / "concept_heat" / "stock_features.parquet"
OUT_NAME = "r5_pump_3way_lgbm_v4"

TRAIN_START = "20230101"
TRAIN_END   = "20250930"
DATA_END    = "20260522"
PUMP_UP_THRESHOLD = 0.10
PUMP_DD_THRESHOLD = 0.05
PUMP_FORWARD = 5


def compute_labels_3way(daily_cache_dir):
    files = sorted(daily_cache_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low"])
                for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    N = PUMP_FORWARD
    big["max_high_next"] = (big.groupby("ts_code")["high"].apply(
        lambda x: x.rolling(N, min_periods=N).max().shift(-N)).reset_index(level=0, drop=True))
    big["min_low_next"] = (big.groupby("ts_code")["low"].apply(
        lambda x: x.rolling(N, min_periods=N).min().shift(-N)).reset_index(level=0, drop=True))
    big["upside"] = big["max_high_next"] / big["next_open"] - 1
    big["downside"] = big["min_low_next"] / big["next_open"] - 1
    big["pump_3way"] = 0
    big.loc[(big["upside"] >= PUMP_UP_THRESHOLD) & (big["downside"] >= -PUMP_DD_THRESHOLD),
              "pump_3way"] = 2
    big.loc[(big["downside"] <= -PUMP_UP_THRESHOLD) & (big["upside"] <= PUMP_DD_THRESHOLD),
              "pump_3way"] = 1
    return big.dropna(subset=["max_high_next"])[["ts_code", "trade_date", "pump_3way"]]


def precision_at_k(val_df, top_k, pred_col, label_target_class):
    rows = []
    for d_, g in val_df.groupby("trade_date"):
        if len(g) < top_k * 2: continue
        top = g.nlargest(top_k, pred_col)
        hit = (top["pump_3way"] == label_target_class).sum()
        rows.append({"precision": hit / top_k})
    if not rows: return 0
    return float(pd.DataFrame(rows)["precision"].mean())


def main():
    t0 = time.time()
    print(f"=== V4 三分类 + 概念热度因子 ===\n", flush=True)
    out_dir = PROD / OUT_NAME
    if (out_dir / "classifier.txt").exists():
        print(f"[{OUT_NAME}] 已存, 跳过"); return

    df = load_window(TRAIN_START, DATA_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        df = df.merge(lf, on=["ts_code", "trade_date"], how="left")
        print(f"  merged long_return: {len(df):,}", flush=True)

    # 新: merge 概念热度
    if CONCEPT_HEAT_P.exists():
        ch = pd.read_parquet(CONCEPT_HEAT_P)
        ch["trade_date"] = ch["trade_date"].astype(str)
        df = df.merge(ch, on=["ts_code", "trade_date"], how="left")
        new_cols = [c for c in ch.columns if c not in ("ts_code", "trade_date")]
        print(f"  merged 概念热度: {len(new_cols)} 新因子, 列名 {new_cols}", flush=True)
        cov = df[new_cols[0]].notna().mean() * 100
        print(f"  概念因子覆盖率: {cov:.1f}%", flush=True)
    else:
        print(f"!! 概念热度文件不存在", flush=True)
        return

    pump = compute_labels_3way(ROOT / "output/tushare_cache/daily")
    df = df.merge(pump, on=["ts_code", "trade_date"], how="inner")
    cnt = df["pump_3way"].value_counts().sort_index()
    print(f"\n  类别分布: 中性 {cnt.get(0,0):,}, 跌 {cnt.get(1,0):,}, 涨 {cnt.get(2,0):,}",
           flush=True)

    industries = pd.Categorical(df["industry"].fillna("unknown"))
    df["industry_id"] = industries.codes
    EXC = set(EXCLUDE) | {"pump_3way", "is_pump_up", "is_pump_down", "is_pump"}
    feat_cols = [c for c in df.columns
                  if c not in EXC and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  特征列总数: {len(feat_cols)} (v3b 247 + 概念 8)", flush=True)

    for c in feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)

    train_df = df[df["trade_date"] < TRAIN_END].copy()
    val_df = df[df["trade_date"] >= TRAIN_END].copy()
    print(f"  train: {len(train_df):,}, val: {len(val_df):,}", flush=True)

    if len(train_df) > 2_000_000:
        train_df = train_df.sample(n=2_000_000, random_state=42).reset_index(drop=True)
        print(f"  subsample 到 200 万", flush=True)

    X_train = train_df[feat_cols].astype("float32")
    y_train = train_df["pump_3way"].astype("int8")
    X_val = val_df[feat_cols].astype("float32")
    y_val = val_df["pump_3way"].astype("int8")

    print(f"\n训练 multiclass (无 balance) ...", flush=True)
    clf = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, metric="multi_logloss",
        n_estimators=3000, learning_rate=0.03,
        num_leaves=63, min_child_samples=300,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        random_state=42, n_jobs=4, verbose=-1,
    )
    clf.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(100, first_metric_only=True),
                          lgb.log_evaluation(100)])

    out_dir.mkdir(exist_ok=True, parents=True)
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.categories,
                                                    range(len(industries.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": "pump_3way", "model_type": "multiclass_3way_with_concept",
        "version": "v4_concept_heat",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    y_pred_proba = clf.predict_proba(X_val)
    val_eval = val_df[["ts_code", "trade_date", "pump_3way"]].copy()
    val_eval["P_neutral"] = y_pred_proba[:, 0]
    val_eval["P_down"] = y_pred_proba[:, 1]
    val_eval["P_up"] = y_pred_proba[:, 2]

    print(f"\n[OK] {OUT_NAME} best_iter = {clf.best_iteration_}", flush=True)

    print(f"\n--- precision@K (vs v3b) ---", flush=True)
    print(f"\n涨启动子:")
    pump_up_rate = (val_eval["pump_3way"] == 2).mean()
    print(f"  基线: {pump_up_rate*100:.2f}%", flush=True)
    for k in [5, 10, 20, 50]:
        p = precision_at_k(val_eval, k, "P_up", 2)
        amp = p / pump_up_rate
        print(f"  Top {k:3d}: v4 {p*100:.2f}% / {amp:.2f}x  |  v3b ~28%/2.2x", flush=True)

    print(f"\n跌启动子:")
    pump_dn_rate = (val_eval["pump_3way"] == 1).mean()
    print(f"  基线: {pump_dn_rate*100:.2f}%", flush=True)
    for k in [5, 10, 20, 50]:
        p = precision_at_k(val_eval, k, "P_down", 1)
        amp = p / pump_dn_rate
        print(f"  Top {k:3d}: v4 {p*100:.2f}% / {amp:.2f}x  |  v3b ~25%/3.9x", flush=True)

    # 概念因子的重要性
    print(f"\n--- 概念因子 (8 个) 在模型中的重要性 ---", flush=True)
    fi = clf.booster_.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_cols, fi), key=lambda x: -x[1])
    concept_cols = [c for c in feat_cols if c.startswith("stock_cpt_")]
    for c in concept_cols:
        idx = feat_cols.index(c)
        rank = next(i for i, (n, _) in enumerate(feat_imp) if n == c) + 1
        print(f"  {c:30s} rank #{rank:3d}/{len(feat_cols)}, gain={fi[idx]:.0f}", flush=True)

    Path(out_dir / "meta.json").write_text(json.dumps({
        "best_iter": int(clf.best_iteration_),
        "version": "v4_concept_heat",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
