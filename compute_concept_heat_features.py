"""概念热度因子计算 (任务 2 + 任务 3 共享数据).

对每个 (概念, 交易日), 算 5 个热度指标 (全部向后看, 防 forward leak):

A 概念历史强度 (过去 N 日):
  1. cpt_ret_5d:   概念成员过去 5 日累计 return 均值
  2. cpt_ret_20d:  过去 20 日累计 return 均值
  3. cpt_vol_ratio_5d:  过去 5 日成交量 / MA20 均值
  4. cpt_winners_5d:  过去 5 日 ret > 0 的成员占比
  5. cpt_strong_5d:  过去 5 日 ret > 3% 的成员占比 (强势股比例)

B 个股层面 (反向 merge):
  对每股 t 日, 取其关联的所有概念的均值 (跨概念聚合)
  输出列: stock_cpt_ret_5d, stock_cpt_ret_20d, ... (前缀 stock_cpt_)

输出:
  output/concept_heat/concept_daily.parquet     (concept_name, trade_date) 级
  output/concept_heat/stock_features.parquet    (ts_code, trade_date) 级新因子
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "concept_heat"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print(f"\n=== 概念热度因子计算 ===\n", flush=True)

    # 1. 加载 daily cache 算个股 N 日历史 return
    print(f"[1] 加载 daily cache 算个股历史 return ...", flush=True)
    daily_dir = ROOT / "output/tushare_cache/daily"
    files = sorted(daily_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low",
                                            "close", "vol"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 防 leak: 用 close[t-1] / close[t-N-1] - 1
    big["close_prev"] = big.groupby("ts_code")["close"].shift(1)
    big["close_5d_ago"] = big.groupby("ts_code")["close"].shift(6)
    big["close_20d_ago"] = big.groupby("ts_code")["close"].shift(21)
    big["stock_ret_5d"] = (big["close_prev"] / big["close_5d_ago"] - 1) * 100
    big["stock_ret_20d"] = (big["close_prev"] / big["close_20d_ago"] - 1) * 100
    big["vol_ma20_lag"] = big.groupby("ts_code")["vol"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=20).mean())
    big["vol_ratio_5d"] = (big.groupby("ts_code")["vol"].shift(1) /
                              big["vol_ma20_lag"]).clip(0, 20)
    print(f"  daily: {len(big):,}, 日数 {big['trade_date'].nunique()}", flush=True)

    # 2. 加载 concept_merged
    cm = pd.read_parquet(ROOT / "output/concept_local/concept_merged.parquet")
    cm = cm[["stock_code", "concept_name"]].drop_duplicates()
    print(f"  concept_merged: {len(cm):,}, {cm['concept_name'].nunique()} 概念", flush=True)

    # 3. 计算 concept_daily 因子
    print(f"\n[2] 概念每日热度因子 (groupby concept × trade_date) ...", flush=True)
    # merge concept ↔ daily
    big_slim = big[["ts_code", "trade_date", "stock_ret_5d", "stock_ret_20d",
                     "vol_ratio_5d"]].dropna()
    joined = cm.merge(big_slim, left_on="stock_code", right_on="ts_code", how="inner")
    print(f"  joined: {len(joined):,}", flush=True)

    concept_daily = joined.groupby(["concept_name", "trade_date"]).agg(
        cpt_ret_5d=("stock_ret_5d", "mean"),
        cpt_ret_20d=("stock_ret_20d", "mean"),
        cpt_vol_ratio_5d=("vol_ratio_5d", "mean"),
        cpt_winners_5d=("stock_ret_5d", lambda x: (x > 0).mean()),
        cpt_strong_5d=("stock_ret_5d", lambda x: (x > 3).mean()),
        cpt_n_members=("ts_code", "count"),
    ).reset_index()
    print(f"  concept_daily: {len(concept_daily):,} 行 ({concept_daily['concept_name'].nunique()} 概念 × {concept_daily['trade_date'].nunique()} 日)",
           flush=True)

    out_p1 = OUT / "concept_daily.parquet"
    concept_daily.to_parquet(out_p1, index=False)
    print(f"  输出 concept_daily: {out_p1}", flush=True)

    # 4. 反向 merge: 给每股 (ts_code, trade_date) 算"所属概念的热度均值"
    print(f"\n[3] 反向 merge 给个股加热度因子 ...", flush=True)
    # 用 concept_merged + concept_daily merge, 然后 groupby (stock_code, trade_date) 取均值
    big_concept = cm.merge(concept_daily, on="concept_name", how="left")
    print(f"  big_concept (含 NaN): {len(big_concept):,}", flush=True)

    # 按 (stock_code, trade_date) 聚合 (每股关联多概念, 取其热度均值)
    stock_heat = big_concept.dropna(subset=["trade_date"]).groupby(
        ["stock_code", "trade_date"]).agg(
            stock_cpt_ret_5d=("cpt_ret_5d", "mean"),
            stock_cpt_ret_20d=("cpt_ret_20d", "mean"),
            stock_cpt_vol_ratio_5d=("cpt_vol_ratio_5d", "mean"),
            stock_cpt_winners_5d=("cpt_winners_5d", "mean"),
            stock_cpt_strong_5d=("cpt_strong_5d", "mean"),
            stock_cpt_max_strong=("cpt_strong_5d", "max"),    # 最强概念
            stock_cpt_top_ret_5d=("cpt_ret_5d", "max"),       # 最强概念的 5 日 ret
            stock_cpt_n_concepts=("cpt_ret_5d", "count"),
    ).reset_index()
    stock_heat = stock_heat.rename(columns={"stock_code": "ts_code"})
    print(f"  stock_heat: {len(stock_heat):,} (ts_code, trade_date)", flush=True)

    out_p2 = OUT / "stock_features.parquet"
    stock_heat.to_parquet(out_p2, index=False)
    print(f"  输出 stock_features: {out_p2}", flush=True)

    # 简要统计
    print(f"\n--- 统计 ---", flush=True)
    print(f"  stock_features 列: {stock_heat.columns.tolist()}", flush=True)
    print(f"  数据覆盖率 (非 NaN):", flush=True)
    for c in stock_heat.columns:
        if c in ("ts_code", "trade_date"): continue
        n = stock_heat[c].notna().sum()
        print(f"    {c:30s}: 均 {stock_heat[c].mean():+.3f}, std {stock_heat[c].std():.3f}",
               flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
