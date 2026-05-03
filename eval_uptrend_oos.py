#!/usr/bin/env python3
"""OOS 全量评估 — 起涨点检测器 + 综合 entry_score 表现.

用 batch 预测 (秒级), 评估:
  1. uptrend_prob 5 分位 vs 真实起涨点命中率
  2. uptrend_prob 各档位 vs 真实 max_gain / max_dd
  3. entry_score 各档位 vs 真实表现
  4. regime × MV 切片 (新模型加了 regime feature)
  5. 真实大涨股 (max_gain >= 30%) 在 top X% 中的捕获率

测试期: 2025-05 → 2026-01
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
LABELS = "output/labels/max_gain_labels.parquet"
STARTS = "output/uptrend_starts/starts.parquet"
REGIMES = "output/regimes/daily_regime.parquet"
ETF    = "output/etf_analysis/stock_to_etfs.json"
OUT_DIR = Path("output/eval_uptrend")
OUT_DIR.mkdir(exist_ok=True)

START = "20250501"
END   = "20260126"


def bucket_mv(v):
    if pd.isna(v): return None
    bil = v / 1e4
    if bil < 50: return "20-50亿"
    if bil < 100: return "50-100亿"
    if bil < 300: return "100-300亿"
    if bil < 1000: return "300-1000亿"
    return "1000亿+"


def main():
    t0 = time.time()
    print(f"加载 OOS {START} → {END}...")
    parts = []
    for p in sorted(Path("output/factor_lab_3y/factor_groups").glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    # 标签
    labels = pd.read_parquet(LABELS, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left")
    full = full.dropna(subset=["max_gain_20"])

    # 起涨点真实标签
    starts = pd.read_parquet(STARTS, columns=["ts_code","trade_date"])
    starts["trade_date"] = starts["trade_date"].astype(str)
    pos_keys = set(starts["ts_code"]+"|"+starts["trade_date"])
    full["_key"] = full["ts_code"]+"|"+full["trade_date"]
    full["is_uptrend"] = full["_key"].isin(pos_keys).astype(int)

    # 合并 amount features
    amount_path = Path("output/amount_features/amount_features.parquet")
    if amount_path.exists():
        amount = pd.read_parquet(amount_path)
        amount["trade_date"] = amount["trade_date"].astype(str)
        full = full.merge(amount, on=["ts_code","trade_date"], how="left")

    # 合并 regime extra
    rextra_path = Path("output/regime_extra/regime_extra.parquet")
    if rextra_path.exists():
        rextra = pd.read_parquet(rextra_path)
        rextra["trade_date"] = rextra["trade_date"].astype(str)
        full = full.merge(rextra, on="trade_date", how="left")

    # 合并 moneyflow features
    mf_path = Path("output/moneyflow/features.parquet")
    if mf_path.exists():
        mf = pd.read_parquet(mf_path)
        mf["trade_date"] = mf["trade_date"].astype(str)
        full = full.merge(mf, on=["ts_code","trade_date"], how="left")

    # regime
    if Path(REGIMES).exists():
        regime = pd.read_parquet(REGIMES,
                                  columns=["trade_date","regime","regime_id",
                                            "ret_5d","ret_20d","ret_60d","rsi14",
                                            "vol_ratio","cyb_ret_20d","zz500_ret_20d"])
        regime["trade_date"] = regime["trade_date"].astype(str)
        regime = regime.rename(columns={
            "ret_5d": "mkt_ret_5d", "ret_20d": "mkt_ret_20d",
            "ret_60d": "mkt_ret_60d", "rsi14": "mkt_rsi14",
            "vol_ratio": "mkt_vol_ratio",
        })
        full = full.merge(regime, on="trade_date", how="left")

    print(f"OOS 样本: {len(full)}, 真实起涨 {full['is_uptrend'].sum()} ({full['is_uptrend'].mean()*100:.2f}%)")

    # 加载 uptrend 模型
    booster = lgb.Booster(model_file="output/lgbm_uptrend/classifier.txt")
    meta = json.loads(Path("output/lgbm_uptrend/feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    indmap = meta["industry_map"]

    # 准备特征
    full["industry_id"] = full["industry"].fillna("unknown").map(
        lambda x: indmap.get(str(x), -1))
    for c in feat_cols:
        if c not in full.columns: full[c] = np.nan
    X = full[feat_cols].astype(float)

    print("批量预测...")
    full["uptrend_prob"] = booster.predict(X)
    print(f"预测完成 ({time.time()-t0:.1f}s)")

    # 1. Lift 分析
    base = full["is_uptrend"].mean()
    print(f"\n=== Lift (基线起涨率 {base*100:.3f}%) ===")
    sorted_df = full.sort_values("uptrend_prob", ascending=False)
    rows = []
    for pct in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
        n = int(len(sorted_df) * pct)
        sub = sorted_df.head(n)
        rate = sub["is_uptrend"].mean() * 100
        avg_gain = sub["max_gain_20"].mean()
        avg_dd = sub["max_dd_20"].mean()
        gain_15 = (sub["max_gain_20"] >= 15).mean() * 100
        gain_30 = (sub["max_gain_20"] >= 30).mean() * 100
        rows.append({
            "top_pct": f"{pct*100:.1f}%", "n": n,
            "起涨率": round(rate, 2), "lift": round(rate / max(base*100, 0.01), 2),
            "max_gain_avg": round(avg_gain, 2), "max_dd_avg": round(avg_dd, 2),
            "gain_15%": round(gain_15, 1), "gain_30%": round(gain_30, 1),
        })
    lift_df = pd.DataFrame(rows)
    print(lift_df.to_string(index=False))

    # 2. 真实大涨股 (max_gain >= 30%) 在 top X% 中的捕获率
    big_up = full[full["max_gain_20"] >= 30]
    print(f"\n=== 大涨股捕获率 (max_gain >= 30%, N={len(big_up)}) ===")
    for pct in [0.01, 0.02, 0.05, 0.10, 0.20]:
        n = int(len(sorted_df) * pct)
        captured = sorted_df.head(n)["max_gain_20"].ge(30).sum()
        recall = captured / len(big_up) * 100 if len(big_up) > 0 else 0
        precision = captured / n * 100 if n > 0 else 0
        print(f"  Top {pct*100:.0f}% (n={n:>6}): 捕获 {captured} 个 (recall {recall:.1f}%, precision {precision:.2f}%)")

    # 3. uptrend_prob 5 分位
    print("\n=== uptrend_prob 5 分位 ===")
    full["pred_q"] = full.groupby("trade_date")["uptrend_prob"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") + 1)
    summary = full.groupby("pred_q").agg(
        prob_avg=("uptrend_prob", "mean"),
        true_uptrend_rate=("is_uptrend", lambda x: x.mean()*100),
        max_gain_avg=("max_gain_20", "mean"),
        max_dd_avg=("max_dd_20", "mean"),
        gain_15=("max_gain_20", lambda x: (x>=15).mean()*100),
        gain_20=("max_gain_20", lambda x: (x>=20).mean()*100),
        gain_30=("max_gain_20", lambda x: (x>=30).mean()*100),
        n=("is_uptrend", "count"),
    ).round(3)
    print(summary.to_string())

    # 4. 按 regime 分桶 (新功能)
    if "regime" in full.columns:
        print("\n=== Top 5% 按 regime 切片 ===")
        top5 = sorted_df.head(int(len(sorted_df) * 0.05))
        rg = top5.groupby("regime", observed=True).agg(
            n=("is_uptrend", "count"),
            真实起涨率=("is_uptrend", lambda x: x.mean()*100),
            max_gain_avg=("max_gain_20", "mean"),
            gain_15=("max_gain_20", lambda x: (x>=15).mean()*100),
            gain_20=("max_gain_20", lambda x: (x>=20).mean()*100),
        ).round(2)
        rg = rg.sort_values("真实起涨率", ascending=False)
        print(rg.to_string())

    # 5. MV × ETF top 5%
    full["mv_seg"] = full["total_mv"].apply(bucket_mv)
    etf_holders = set(json.loads(Path(ETF).read_text(encoding="utf-8")).keys())
    full["etf_held"] = full["ts_code"].isin(etf_holders)
    print("\n=== Top 5% 按 MV × ETF 切片 ===")
    top5 = sorted_df.head(int(len(sorted_df) * 0.05))
    full_top5 = full.loc[top5.index]
    cuts = full_top5.groupby(["mv_seg", "etf_held"], observed=True).agg(
        n=("is_uptrend", "count"),
        起涨率=("is_uptrend", lambda x: x.mean()*100),
        max_gain=("max_gain_20", "mean"),
        gain_15=("max_gain_20", lambda x: (x>=15).mean()*100),
        gain_30=("max_gain_20", lambda x: (x>=30).mean()*100),
    ).round(2)
    print(cuts.to_string())

    # 存盘
    lift_df.to_csv(OUT_DIR / "lift.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "5_quantile.csv", encoding="utf-8-sig")
    if "regime" in full.columns:
        rg.to_csv(OUT_DIR / "by_regime.csv", encoding="utf-8-sig")
    cuts.to_csv(OUT_DIR / "by_mv_etf.csv", encoding="utf-8-sig")

    print(f"\n总耗时 {time.time()-t0:.1f}s, 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
