"""拉龙虎榜历史数据 (Tushare top_list + top_inst).

Tushare 接口:
  top_list: 龙虎榜每日明细 (个股入榜原因 + 买卖额)
  top_inst: 龙虎榜机构明细 (营业部/机构买卖)

时间范围: 2023-01 至今 (跟 daily 同步)

输出:
  output/tushare_cache/top_list.parquet
  output/tushare_cache/top_inst.parquet
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
import tushare as ts

TOKEN = os.getenv("TUSHARE_TOKEN")
if not TOKEN:
    print("!! TUSHARE_TOKEN 缺失"); sys.exit(2)
pro = ts.pro_api(TOKEN)

CACHE = ROOT / "output" / "tushare_cache"
CACHE.mkdir(parents=True, exist_ok=True)

START = "20230101"
END = "20260525"


def fetch_top_list():
    out_p = CACHE / "top_list.parquet"
    if out_p.exists():
        df = pd.read_parquet(out_p)
        print(f"[top_list] 缓存: {len(df):,} 行 {df['trade_date'].nunique()} 日", flush=True)
        return df

    # 获取交易日历
    cal = pro.trade_cal(exchange="SSE", start_date=START, end_date=END, is_open="1")
    dates = sorted(cal["cal_date"].tolist())
    print(f"[top_list] 拉 {len(dates)} 个交易日 ...", flush=True)

    parts = []
    for i, d in enumerate(dates):
        try:
            df = pro.top_list(trade_date=d)
            if df is None or df.empty: continue
            df["trade_date"] = df["trade_date"].astype(str)
            parts.append(df)
            if (i + 1) % 50 == 0:
                print(f"  进度 {i+1}/{len(dates)}", flush=True)
            time.sleep(0.4)
        except Exception as e:
            print(f"  {d} 失败: {e}", flush=True)
            time.sleep(2)

    if not parts:
        print("!! 全部失败 (可能权限不够)", flush=True); return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_p, index=False)
    print(f"  → {len(df):,} 行, 列: {list(df.columns)}", flush=True)
    return df


def fetch_top_inst():
    out_p = CACHE / "top_inst.parquet"
    if out_p.exists():
        df = pd.read_parquet(out_p)
        print(f"[top_inst] 缓存: {len(df):,} 行", flush=True)
        return df

    # top_list 里的日期都有 top_inst
    top_list_p = CACHE / "top_list.parquet"
    if not top_list_p.exists():
        print("!! 先拉 top_list", flush=True); return pd.DataFrame()
    top_list = pd.read_parquet(top_list_p)
    dates = sorted(top_list["trade_date"].unique())
    print(f"[top_inst] 拉 {len(dates)} 个上榜日 ...", flush=True)

    parts = []
    for i, d in enumerate(dates):
        try:
            df = pro.top_inst(trade_date=d)
            if df is None or df.empty: continue
            df["trade_date"] = df["trade_date"].astype(str)
            parts.append(df)
            if (i + 1) % 50 == 0:
                print(f"  进度 {i+1}/{len(dates)}", flush=True)
            time.sleep(0.4)
        except Exception as e:
            print(f"  {d} 失败: {e}", flush=True)
            time.sleep(2)

    if not parts:
        print("!! 全部失败", flush=True); return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out_p, index=False)
    print(f"  → {len(df):,} 行, 列: {list(df.columns)}", flush=True)
    return df


def main():
    t0 = time.time()
    print(f"\n=== 拉龙虎榜历史数据 {START}-{END} ===\n", flush=True)
    df_list = fetch_top_list()
    if df_list.empty:
        print("top_list 拉取失败, 退出"); return
    df_inst = fetch_top_inst()

    # 简要统计
    if not df_list.empty:
        print(f"\n--- top_list 统计 ---", flush=True)
        print(f"  唯一股票数: {df_list['ts_code'].nunique():,}", flush=True)
        print(f"  唯一上榜日: {df_list['trade_date'].nunique():,}", flush=True)
        print(f"  日均上榜数: {len(df_list) / df_list['trade_date'].nunique():.1f} 条", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
