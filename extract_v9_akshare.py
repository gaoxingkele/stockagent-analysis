#!/usr/bin/env python3
"""V9 AKShare 8 指标采集 (轻量, 不参与 V7c 训练).

8 指标:
  全市场快照 (stock_comment_em - 5179 股):
    ef_score (综合得分 0-100)
    ef_focus (关注指数 0-100)
    ef_inst_pct (机构参与度 0-1)
    ef_main_cost_dev (主力成本偏离 %)
    ef_rank_chg (排名变化)
  雪球热度:
    xq_follow_pct (关注度全市场分位)
    xq_tweet_pct (讨论度全市场分位)
  附加:
    ef_desire_now (当前参与意愿, 单股拉)

输出: output/v9_data/akshare_snapshot_{date}.parquet (~5K 股)
"""
from __future__ import annotations
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import akshare as ak

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "v9_data"
OUT_DIR.mkdir(exist_ok=True)


def fetch_em_snapshot():
    """东财全市场综合评论."""
    print("  拉 stock_comment_em (东财综合)...", flush=True)
    t0 = time.time()
    df = ak.stock_comment_em()
    print(f"    {len(df)} 行, {time.time()-t0:.0f}s", flush=True)
    # 标准化字段名
    df = df.rename(columns={
        "代码": "ts_code_raw",
        "综合得分": "ef_score",
        "关注指数": "ef_focus",
        "机构参与度": "ef_inst_pct",  # 已是 0-100, 转换为 0-1
        "主力成本": "ef_main_cost",
        "上升": "ef_rank_chg",
        "最新价": "close_now",
        "涨跌幅": "pct_chg",
        "市盈率": "pe_ttm_now",
        "换手率": "turnover_now",
        "目前排名": "ef_rank_now",
        "交易日": "snap_date",
    })
    df["ef_inst_pct"] = df["ef_inst_pct"] / 100  # 0-1
    # 偏离度
    df["ef_main_cost_dev"] = (df["close_now"] - df["ef_main_cost"]) / df["ef_main_cost"] * 100
    # ts_code 标准化 (000001 → 000001.SZ etc)
    def _ts(c):
        c = str(c).strip()
        if c.startswith(("8","4","9")): return f"{c}.BJ"
        if c.startswith(("5","6")): return f"{c}.SH"
        return f"{c}.SZ"
    df["ts_code"] = df["ts_code_raw"].apply(_ts)
    keep = ["ts_code","snap_date","ef_score","ef_focus","ef_inst_pct",
             "ef_main_cost_dev","ef_rank_chg","ef_rank_now","close_now",
             "pe_ttm_now","turnover_now"]
    return df[keep]


def fetch_xq_hot():
    """雪球关注热度."""
    print("  拉 stock_hot_follow_xq (雪球关注度)...", flush=True)
    t0 = time.time()
    df_f = ak.stock_hot_follow_xq()
    print(f"    {len(df_f)} 行, {time.time()-t0:.0f}s", flush=True)
    df_f = df_f.rename(columns={"股票代码": "ts_code_raw", "关注": "xq_follow_count"})
    # 标准化代码 (SH600519 → 600519.SH)
    def _ts2(c):
        c = str(c).strip()
        if c.startswith("SH"): return f"{c[2:]}.SH"
        if c.startswith("SZ"): return f"{c[2:]}.SZ"
        if c.startswith("BJ"): return f"{c[2:]}.BJ"
        return c
    df_f["ts_code"] = df_f["ts_code_raw"].apply(_ts2)
    df_f["xq_follow_pct"] = df_f["xq_follow_count"].rank(pct=True)

    print("  拉 stock_hot_tweet_xq (雪球讨论度)...", flush=True)
    t0 = time.time()
    df_t = ak.stock_hot_tweet_xq()
    print(f"    {len(df_t)} 行, {time.time()-t0:.0f}s", flush=True)
    df_t = df_t.rename(columns={"股票代码": "ts_code_raw", "关注": "xq_tweet_count"})
    df_t["ts_code"] = df_t["ts_code_raw"].apply(_ts2)
    df_t["xq_tweet_pct"] = df_t["xq_tweet_count"].rank(pct=True)

    df = df_f[["ts_code","xq_follow_count","xq_follow_pct"]].merge(
        df_t[["ts_code","xq_tweet_count","xq_tweet_pct"]],
        on="ts_code", how="outer"
    )
    return df


def main():
    today = datetime.now().strftime("%Y%m%d")
    print(f"=== V9 AKShare 8 指标采集 ({today}) ===\n", flush=True)
    t0 = time.time()

    em = fetch_em_snapshot()
    xq = fetch_xq_hot()

    df = em.merge(xq, on="ts_code", how="left")
    out_path = OUT_DIR / f"akshare_snapshot_{today}.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\n=== 输出 {out_path} ({len(df)} 行) ===", flush=True)
    print("\n=== 8 指标分布 ===", flush=True)
    cols = ["ef_score","ef_focus","ef_inst_pct","ef_main_cost_dev",
            "ef_rank_chg","xq_follow_pct","xq_tweet_pct"]
    print(df[cols].describe(percentiles=[0.05,0.5,0.95]).round(3).to_string(), flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
