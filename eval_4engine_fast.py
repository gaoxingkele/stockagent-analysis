#!/usr/bin/env python3
"""快速 OOS 评估 — 批量预测 (10x 快于逐行).

策略:
  1. 加载 OOS 数据 (220k 样本)
  2. 一次性 batch 预测 max_gain / max_dd (秒级)
  3. 应用仓位逻辑 (向量化)
  4. merge 真实 max_gain/dd 标签
  5. 按仓位档汇总

用 sparse 占位 (=70 中位数) 跳过 sparse 计算 — 仓位主要由 max_gain 驱动.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest_18m import FACTOR_DIR, FACTOR_COLS_EXCLUDE

START = "20250501"
END   = "20260126"
LABELS = "output/labels/max_gain_labels.parquet"
ETF    = "output/etf_analysis/stock_to_etfs.json"
OUT_DIR = Path("output/eval_4engine")
OUT_DIR.mkdir(exist_ok=True)


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
    for p in sorted(FACTOR_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    # 抽样: 每月前 5 天
    full["_ym"] = full["trade_date"].str[:6]
    sampled = full.groupby("_ym")["trade_date"].apply(
        lambda x: sorted(x.unique())[:5]).explode().tolist()
    full = full[full["trade_date"].isin(sampled)].copy()

    # 合并 max_gain 标签
    labels = pd.read_parquet(LABELS, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left")
    full = full.dropna(subset=["max_gain_20"])
    print(f"样本: {len(full)} ({time.time()-t0:.1f}s)")

    # 加载 max_gain 模型
    booster_g = lgb.Booster(model_file="output/lgbm_maxgain/regressor_gain.txt")
    booster_d = lgb.Booster(model_file="output/lgbm_maxgain/regressor_dd.txt")
    meta = json.loads(Path("output/lgbm_maxgain/feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta["industry_map"]

    # 准备特征矩阵
    full["industry_id"] = full["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), industry_map.get("unknown", -1)))
    # 缺失列填 NaN
    for c in feat_cols:
        if c not in full.columns:
            full[c] = np.nan
    X = full[feat_cols].astype(float)

    print(f"批量预测 max_gain + max_dd...")
    full["pred_gain"] = booster_g.predict(X)
    full["pred_dd"]   = booster_d.predict(X)
    full["gain_dd_ratio"] = full["pred_gain"] / np.abs(full["pred_dd"]).clip(lower=0.5)
    print(f"预测完成 ({time.time()-t0:.1f}s)")

    # 应用仓位逻辑 (向量化)
    etf_holders = set(json.loads(Path(ETF).read_text(encoding="utf-8")).keys())
    full["mv_seg"] = full["total_mv"].apply(bucket_mv)
    full["etf_held"] = full["ts_code"].isin(etf_holders)

    # 仓位档
    pg = full["pred_gain"]
    pos = pd.Series(5.0, index=full.index)
    pos = np.where(pg >= 9, 10, pos)
    pos = np.where(pg >= 11, 20, pos)
    pos = np.where(pg >= 13, 40, pos)
    pos = np.where(pg >= 15, 60, pos)
    pos = np.where(pg >= 17, 80, pos)
    pos = pd.Series(pos, index=full.index)

    # 域限: 1000亿+ 非 ETF → 0%
    pos = pos.where(~((full["mv_seg"]=="1000亿+") & (~full["etf_held"])), 0.0)
    # 大盘段非 ETF → 减半
    big_no_etf = full["mv_seg"].isin(["300-1000亿","1000亿+"]) & (~full["etf_held"])
    pos = pos.where(~big_no_etf, pos * 0.5)
    # gain/dd 风险比 < 1.5 → 仓位 *0.7
    low_ratio = full["gain_dd_ratio"] < 1.5
    pos = pos.where(~low_ratio, pos * 0.7)
    # 避雷: pred_gain < 8% → 0
    pos = pos.where(pg >= 8, 0.0)

    full["position_pct"] = pos.round(1)
    print(f"仓位计算完成 ({time.time()-t0:.1f}s)")

    # === 汇总 ===
    full["pos_bucket"] = pd.cut(full["position_pct"],
                                 bins=[-1, 5, 15, 30, 50, 70, 100],
                                 labels=["0-5%","5-15%","15-30%","30-50%","50-70%","70-100%"])
    print("\n" + "=" * 78)
    print("【新四引擎】仓位档 vs 真实 max_gain (OOS, 抽样)")
    print("=" * 78)
    summary = full.groupby("pos_bucket", observed=True).agg(
        n=("max_gain_20","count"),
        pred_gain_avg=("pred_gain","mean"),
        true_gain_avg=("max_gain_20","mean"),
        true_gain_med=("max_gain_20","median"),
        gain_15=("max_gain_20", lambda x: (x>=15).mean()*100),
        gain_20=("max_gain_20", lambda x: (x>=20).mean()*100),
        gain_30=("max_gain_20", lambda x: (x>=30).mean()*100),
        true_dd_avg=("max_dd_20","mean"),
        dd_lt_8=("max_dd_20", lambda x: (x<-8).mean()*100),
        dd_lt_15=("max_dd_20", lambda x: (x<-15).mean()*100),
    ).round(2)
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "pos_bucket_new.csv", encoding="utf-8-sig")

    # MV × ETF 高仓位档
    print("\n=== 强买档 (>=70%) MV × ETF 切片 ===")
    strong = full[full["position_pct"] >= 70]
    if len(strong) > 0:
        c = strong.groupby(["mv_seg","etf_held"], observed=True).agg(
            n=("max_gain_20","count"),
            true_gain=("max_gain_20","mean"),
            gain_15=("max_gain_20", lambda x: (x>=15).mean()*100),
            gain_20=("max_gain_20", lambda x: (x>=20).mean()*100),
            gain_30=("max_gain_20", lambda x: (x>=30).mean()*100),
            dd_avg=("max_dd_20","mean"),
        ).round(2)
        print(c.to_string())
        c.to_csv(OUT_DIR / "strong_mv_etf_new.csv", encoding="utf-8-sig")

    # 旧 vs 新 对比
    print("\n=== 旧 (clean 主导) vs 新 (max_gain 主导) 对比 ===")
    print("旧高仓位 60-100% 档: max_gain=11.10%, gain_15=22.2%, gain_20=13.9%")
    new_60plus = full[full["position_pct"] >= 60]
    if len(new_60plus) > 0:
        print(f"新高仓位 60-100% 档 (n={len(new_60plus)}): "
              f"max_gain={new_60plus['max_gain_20'].mean():.2f}% "
              f"gain_15={(new_60plus['max_gain_20']>=15).mean()*100:.1f}% "
              f"gain_20={(new_60plus['max_gain_20']>=20).mean()*100:.1f}%")

    print(f"\n总耗时: {time.time()-t0:.1f}s")
    full.to_parquet(OUT_DIR / "scored_fast.parquet", index=False)


if __name__ == "__main__":
    main()
