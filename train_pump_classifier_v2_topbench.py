"""启动子分类器 v2: 加龙虎榜因子重训.

vs v1:
  - 加 12 个龙虎榜因子 (top_list_features/features.parquet)
  - 非上榜日因子用 0 (let model learn 0 = 非上榜)

目标: AUC 0.69 → 0.72+? precision@10 26% → 30%+?
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window, EXCLUDE

PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"
TOP_LIST_FEAT_P = ROOT / "output" / "top_list_features" / "features.parquet"

TRAIN_START = "20230101"
TRAIN_END   = "20250930"
DATA_END    = "20260522"

PUMP_UP_THRESHOLD = 0.10
PUMP_DD_THRESHOLD = 0.05
PUMP_FORWARD = 5

# 输出两个模型 (up + down) v2
MODELS = [
    {"name": "r5_pump_lgbm_v2", "label_col": "is_pump_up",
      "label_logic": lambda up, dn: ((up >= PUMP_UP_THRESHOLD) & (dn >= -PUMP_DD_THRESHOLD)).astype(int)},
    {"name": "r5_pump_down_lgbm_v2", "label_col": "is_pump_down",
      "label_logic": lambda up, dn: ((dn <= -PUMP_UP_THRESHOLD) & (up <= PUMP_DD_THRESHOLD)).astype(int)},
]


def compute_labels(daily_cache_dir: Path) -> pd.DataFrame:
    """从 daily cache 算 pump_up 和 pump_down label."""
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
    big["is_pump_up"] = ((big["upside"] >= PUMP_UP_THRESHOLD) &
                           (big["downside"] >= -PUMP_DD_THRESHOLD)).astype(int)
    big["is_pump_down"] = ((big["downside"] <= -PUMP_UP_THRESHOLD) &
                             (big["upside"] <= PUMP_DD_THRESHOLD)).astype(int)
    return big.dropna(subset=["max_high_next"])[
        ["ts_code", "trade_date", "is_pump_up", "is_pump_down"]
    ]


def precision_at_k(val_df, top_k, label_col):
    rows = []
    for d_, g in val_df.groupby("trade_date"):
        if len(g) < top_k * 2: continue
        top = g.nlargest(top_k, "pred")
        rows.append({"precision": top[label_col].sum() / top_k})
    if not rows: return 0
    return float(pd.DataFrame(rows)["precision"].mean())


def train_one(name, label_col, df, feat_cols, industries):
    out_dir = PROD / name
    if (out_dir / "classifier.txt").exists():
        print(f"[{name}] 已存, 跳过"); return

    print(f"\n=== 训 {name} (label={label_col}) ===", flush=True)
    train_df = df[df["trade_date"] < TRAIN_END].copy()
    val_df = df[df["trade_date"] >= TRAIN_END].copy()
    print(f"  train={len(train_df):,}, val={len(val_df):,}", flush=True)
    print(f"  train pump 率: {train_df[label_col].mean()*100:.2f}%, "
           f"val: {val_df[label_col].mean()*100:.2f}%", flush=True)

    if len(train_df) > 2_000_000:
        train_df = train_df.sample(n=2_000_000, random_state=42).reset_index(drop=True)

    X_train = train_df[feat_cols].astype("float32")
    y_train = train_df[label_col].astype("int8")
    X_val = val_df[feat_cols].astype("float32")
    y_val = val_df[label_col].astype("int8")

    clf = lgb.LGBMClassifier(
        objective="binary", metric="auc",
        n_estimators=3000, learning_rate=0.03,
        num_leaves=63, min_child_samples=300,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        max_bin=127, force_col_wise=True,
        is_unbalance=True,
        random_state=42, n_jobs=4, verbose=-1,
    )
    clf.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(100, first_metric_only=True),
                          lgb.log_evaluation(200)])

    out_dir.mkdir(exist_ok=True, parents=True)
    clf.booster_.save_model(str(out_dir / "classifier.txt"))
    industry_map = {str(s): int(i) for s, i in zip(industries.categories,
                                                    range(len(industries.categories)))}
    Path(out_dir / "feature_meta.json").write_text(json.dumps({
        "feature_cols": feat_cols, "industry_map": industry_map,
        "target": label_col, "model_type": "binary_classifier",
        "version": "v2_with_top_list",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    y_pred = clf.predict_proba(X_val)[:, 1]
    auc = clf.best_score_["valid_0"]["auc"]
    print(f"\n[OK] {name} AUC = {auc:.4f}, best_iter = {clf.best_iteration_}", flush=True)

    val_eval = val_df[["ts_code", "trade_date", label_col]].copy()
    val_eval["pred"] = y_pred
    pump_rate = val_eval[label_col].mean()
    print(f"  基线 (val pump 率): {pump_rate*100:.2f}%")
    prec_results = []
    for k in [5, 10, 20, 50]:
        p = precision_at_k(val_eval, k, label_col)
        amplify = p / pump_rate if pump_rate > 0 else 0
        print(f"  Top {k:3d}: precision = {p*100:.2f}% (放大 {amplify:.2f}x)", flush=True)
        prec_results.append({"top_k": k, "precision": p, "amplify": amplify})

    Path(out_dir / "meta.json").write_text(json.dumps({
        "auc": float(auc), "best_iter": int(clf.best_iteration_),
        "n_train": len(train_df), "n_val": len(val_df),
        "pump_rate_val": float(pump_rate),
        "precision_at_k": prec_results,
        "version": "v2_with_top_list",
        "train_window": f"< {TRAIN_END}",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    del clf, X_train, y_train, X_val, y_val, train_df, val_df
    gc.collect()


def main():
    t0 = time.time()
    print(f"=== 启动子分类器 v2 (加龙虎榜因子) ===\n", flush=True)

    print("[1] load_window ...", flush=True)
    df = load_window(TRAIN_START, DATA_END, with_mfk=True)
    df["trade_date"] = df["trade_date"].astype(str)

    if LONG_FEAT_P.exists():
        print("[2] merge long_return_features ...", flush=True)
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        df = df.merge(lf, on=["ts_code", "trade_date"], how="left")

    # 关键: merge top_list_features, 非上榜日填 0
    if TOP_LIST_FEAT_P.exists():
        print("[3] merge top_list_features ...", flush=True)
        tlf = pd.read_parquet(TOP_LIST_FEAT_P)
        tlf["trade_date"] = tlf["trade_date"].astype(str)
        new_cols = [c for c in tlf.columns if c not in ("ts_code", "trade_date")]
        df = df.merge(tlf, on=["ts_code", "trade_date"], how="left")
        # 非上榜日填 0 (days_since_last_tl 例外填 999)
        for c in new_cols:
            if c == "days_since_last_tl":
                df[c] = df[c].fillna(999)
            else:
                df[c] = df[c].fillna(0)
        print(f"  新增 {len(new_cols)} 个龙虎榜因子: {new_cols[:3]} ...", flush=True)
        # 上榜率 (非 0)
        in_list_rate = (df["tl_in_list"] > 0).mean() * 100
        print(f"  上榜日占总样本比例: {in_list_rate:.2f}%", flush=True)
    else:
        print("!! top_list_features 不存在, 先跑 compute_top_list_features.py", flush=True)
        return

    # labels
    print("\n[4] pump_up + pump_down labels ...", flush=True)
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    labels = compute_labels(daily_dir)
    df = df.merge(labels, on=["ts_code", "trade_date"], how="inner")
    print(f"  合并后: {len(df):,}, pump_up {df['is_pump_up'].mean()*100:.2f}%, "
           f"pump_down {df['is_pump_down'].mean()*100:.2f}%", flush=True)

    industries = pd.Categorical(df["industry"].fillna("unknown"))
    df["industry_id"] = industries.codes

    EXC = set(EXCLUDE) | {"is_pump_up", "is_pump_down", "is_pump"}
    feat_cols = [c for c in df.columns
                  if c not in EXC and pd.api.types.is_numeric_dtype(df[c])]
    print(f"\n  特征列总数 (v1: 247, v2 多 12): {len(feat_cols)}", flush=True)

    for c in feat_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).clip(-1e10, 1e10)

    for m in MODELS:
        train_one(m["name"], m["label_col"], df, feat_cols, industries)

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
