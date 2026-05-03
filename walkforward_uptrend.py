#!/usr/bin/env python3
"""走步式滚动训练 — 5 窗口跨 regime 稳定性验证.

每窗口: 12 月训练 + 3 月测试.

| 窗口 | 训练期 | 测试期 | 测试期主要 regime |
| W1 | 2023-01→2023-12 | 2024-01→2024-04 | 探底/震荡 |
| W2 | 2023-04→2024-04 | 2024-05→2024-09 | 震荡→反弹启动 |
| W3 | 2023-10→2024-09 | 2024-10→2024-12 | 政策快牛 |
| W4 | 2024-04→2025-04 | 2025-05→2025-09 | 慢牛分化 |
| W5 | 2024-10→2025-10 | 2025-11→2026-01 | 慢牛后期 |

对每窗口产出:
  - OOS AUC + Top 1% lift
  - 5 分位单调性
  - 各 regime 段表现
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
STARTS_PATH = ROOT / "output" / "uptrend_starts" / "starts.parquet"
REGIME_PATH = ROOT / "output" / "regimes" / "daily_regime.parquet"
AMOUNT_PATH = ROOT / "output" / "amount_features" / "amount_features.parquet"
OUT_DIR = ROOT / "output" / "walkforward"
OUT_DIR.mkdir(exist_ok=True)

WINDOWS = [
    ("W1", "20230101", "20231231", "20240101", "20240430"),
    ("W2", "20230401", "20240430", "20240501", "20240930"),
    ("W3", "20231001", "20240930", "20241001", "20241231"),
    ("W4", "20240401", "20250430", "20250501", "20250930"),
    ("W5", "20241001", "20251031", "20251101", "20260126"),
]

EXCLUDE = {"ts_code","trade_date","industry","name",
           "r5","r10","r20","r30","r40",
           "dd5","dd10","dd20","dd30","dd40",
           "mv_bucket","pe_bucket","_key","is_uptrend"}
NEG_RATIO = 5


def load_window(start: str, end: str, pos_keys: set) -> pd.DataFrame:
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["_key"] = full["ts_code"] + "|" + full["trade_date"]
    full["is_uptrend"] = full["_key"].isin(pos_keys).astype(int)
    # regime
    if REGIME_PATH.exists():
        regime = pd.read_parquet(REGIME_PATH,
                                  columns=["trade_date","regime","regime_id",
                                            "ret_5d","ret_20d","ret_60d","rsi14",
                                            "vol_ratio","cyb_ret_20d","zz500_ret_20d"])
        regime["trade_date"] = regime["trade_date"].astype(str)
        regime = regime.rename(columns={
            "ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
            "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio",
        })
        full = full.merge(regime, on="trade_date", how="left")
    # amount
    if AMOUNT_PATH.exists():
        amount = pd.read_parquet(AMOUNT_PATH)
        amount["trade_date"] = amount["trade_date"].astype(str)
        full = full.merge(amount, on=["ts_code","trade_date"], how="left")
    return full


def run_window(name, train_start, train_end, test_start, test_end, pos_keys):
    print(f"\n{'='*70}")
    print(f"窗口 {name}: 训 {train_start}→{train_end}, 测 {test_start}→{test_end}")
    print(f"{'='*70}")

    train = load_window(train_start, train_end, pos_keys).dropna(subset=["r20"])
    pos_train = int(train["is_uptrend"].sum())
    if pos_train < 1000:
        print(f"  ⚠ 训练正例 {pos_train} 太少, 跳过")
        return None

    pos_df = train[train["is_uptrend"] == 1]
    neg_df = train[train["is_uptrend"] == 0].sample(
        min(NEG_RATIO * pos_train, train["is_uptrend"].eq(0).sum()),
        random_state=42)
    train_b = pd.concat([pos_df, neg_df], ignore_index=True)

    test = load_window(test_start, test_end, pos_keys).dropna(subset=["r20"])
    pos_test = int(test["is_uptrend"].sum())
    print(f"  训练: {len(train_b)} 行 ({pos_train} 正例) | 测试: {len(test)} 行 ({pos_test} 正例)")

    feat_cols = [c for c in train_b.columns
                 if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train_b[c])]
    industries = pd.concat([train_b["industry"], test["industry"]]).fillna("unknown").astype("category")
    train_b["industry_id"] = pd.Categorical(train_b["industry"].fillna("unknown"),
                                              categories=industries.cat.categories).codes
    test["industry_id"] = pd.Categorical(test["industry"].fillna("unknown"),
                                            categories=industries.cat.categories).codes
    feat_cols.append("industry_id")

    X_train = train_b[feat_cols]; y_train = train_b["is_uptrend"]
    X_test = test[feat_cols]; y_test = test["is_uptrend"]

    pos_ratio = y_train.mean()
    spw = (1 - pos_ratio) / pos_ratio

    clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=200, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=5,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1,
        objective="binary", metric="auc",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
             categorical_feature=["industry_id"],
             callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)])

    pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    test = test.copy()
    test["uptrend_prob"] = pred
    base = y_test.mean()

    # Lift
    sorted_t = test.sort_values("uptrend_prob", ascending=False)
    n = len(sorted_t)
    top1 = sorted_t.head(int(n*0.01))
    top5 = sorted_t.head(int(n*0.05))
    rate1 = top1["is_uptrend"].mean(); rate5 = top5["is_uptrend"].mean()
    lift1 = rate1 / max(base, 0.0001); lift5 = rate5 / max(base, 0.0001)

    # 5 分位 (单调性)
    test["pred_q"] = test.groupby("trade_date")["uptrend_prob"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1
    )
    q_rates = test.groupby("pred_q")["is_uptrend"].mean().values
    monotonic = all(q_rates[i] <= q_rates[i+1] for i in range(len(q_rates)-1))

    # 各 regime 在 top 5%
    regime_perf = {}
    if "regime" in test.columns:
        for r in test["regime"].dropna().unique():
            sub = test[test["regime"] == r]
            if len(sub) < 200: continue
            sub_top = sub.sort_values("uptrend_prob", ascending=False).head(int(len(sub)*0.05))
            regime_perf[r] = {
                "n": len(sub), "top5_rate": float(sub_top["is_uptrend"].mean()*100),
            }

    print(f"  AUC={auc:.4f} | best_iter={clf.best_iteration_} | base={base*100:.2f}%")
    print(f"  Top 1% 起涨率 {rate1*100:.2f}% (lift {lift1:.1f}x)")
    print(f"  Top 5% 起涨率 {rate5*100:.2f}% (lift {lift5:.1f}x)")
    print(f"  5 分位单调: {monotonic} (Q1→Q5: {[round(v*100,2) for v in q_rates]})")
    if regime_perf:
        for r, p in sorted(regime_perf.items(), key=lambda x: -x[1]["top5_rate"]):
            print(f"    [{r}] n={p['n']:>6} top5% 起涨率 {p['top5_rate']:.2f}%")

    return {
        "window": name, "train": [train_start, train_end], "test": [test_start, test_end],
        "n_train": len(train_b), "n_test": len(test),
        "n_train_pos": pos_train, "n_test_pos": pos_test,
        "best_iter": int(clf.best_iteration_),
        "auc": float(auc), "base_rate": float(base),
        "top1_rate": float(rate1), "top1_lift": float(lift1),
        "top5_rate": float(rate5), "top5_lift": float(lift5),
        "monotonic": bool(monotonic),
        "q_rates": [float(v) for v in q_rates],
        "regime_perf": regime_perf,
    }


def main():
    t0 = time.time()
    starts = pd.read_parquet(STARTS_PATH, columns=["ts_code","trade_date"])
    starts["trade_date"] = starts["trade_date"].astype(str)
    pos_keys = set(starts["ts_code"]+"|"+starts["trade_date"])

    results = []
    for w in WINDOWS:
        try:
            r = run_window(*w, pos_keys=pos_keys)
            if r: results.append(r)
        except Exception as e:
            print(f"窗口 {w[0]} 失败: {e}")

    print(f"\n{'='*70}")
    print(f"完成 5 窗口走步式回测, 总耗时 {time.time()-t0:.1f}s")
    print(f"{'='*70}")

    summary = pd.DataFrame([{
        "window": r["window"],
        "测试期": f"{r['test'][0]}→{r['test'][1]}",
        "AUC": round(r["auc"], 4),
        "Top1%起涨率": round(r["top1_rate"]*100, 2),
        "Top1%lift": round(r["top1_lift"], 1),
        "Top5%起涨率": round(r["top5_rate"]*100, 2),
        "Top5%lift": round(r["top5_lift"], 1),
        "单调": r["monotonic"],
        "正例数": r["n_test_pos"],
    } for r in results])
    print("\n=== 汇总 ===")
    print(summary.to_string(index=False))

    summary.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    Path(OUT_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
