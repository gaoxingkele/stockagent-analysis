# -*- coding: utf-8 -*-
"""从已下载的 bulk 缓存派生: stock_basic + 单股 moneyflow cache。

1. stock_basic → output/tushare_cache/stock_basic.parquet (ts_code,name,industry,...)
2. output/tushare_cache/moneyflow/*.parquet (逐日全市场) → pivot 成
   output/moneyflow/cache/{ts_code}.parquet (单股, 列=RAW_FIELDS, 按 trade_date 排序)
   —— 省掉原脚本 ~5000 次 per-stock API 调用。
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
import tushare as ts
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.moneyflow.extractor import RAW_FIELDS

MF_BULK = ROOT / "output/tushare_cache/moneyflow"
MF_CACHE = ROOT / "output/moneyflow/cache"
MF_CACHE.mkdir(parents=True, exist_ok=True)
BASIC = ROOT / "output/tushare_cache/stock_basic.parquet"


def log(s):
    sys.stdout.buffer.write((f"[{time.strftime('%H:%M:%S')}] {s}\n").encode("utf-8"))
    sys.stdout.flush()


def build_stock_basic():
    # 上市 + 退市 + 暂停 全都要 (ST 过滤/名称匹配需覆盖全市场)
    parts = []
    for st in ("L", "D", "P"):
        try:
            df = pro.stock_basic(list_status=st,
                                 fields="ts_code,symbol,name,area,industry,market,list_date,list_status")
            if df is not None and len(df):
                parts.append(df)
            time.sleep(0.3)
        except Exception as e:
            log(f"  stock_basic({st}) err {str(e)[:50]}")
    basic = pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")
    basic.to_parquet(BASIC, index=False)
    log(f"stock_basic 写出 {len(basic)} 行 -> {BASIC.name}")


def build_moneyflow_cache():
    files = sorted(MF_BULK.glob("*.parquet"))
    log(f"读取 {len(files)} 个 bulk moneyflow 文件 ...")
    big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    # 只保留 RAW_FIELDS 里存在的列
    keep = [c for c in RAW_FIELDS if c in big.columns]
    log(f"bulk 合并 {len(big):,} 行, 保留列 {keep}")
    n = 0
    t0 = time.time()
    groups = big.groupby("ts_code")
    total = len(groups)
    for ts_code, g in groups:
        g = g[keep].sort_values("trade_date").reset_index(drop=True)
        g.to_parquet(MF_CACHE / f"{ts_code}.parquet", index=False)
        n += 1
        if n % 500 == 0 or n == total:
            log(f"  {n}/{total} 单股缓存写出, 用时 {time.time()-t0:.0f}s")
    log(f"moneyflow 单股缓存完成: {n} 只")


if __name__ == "__main__":
    build_stock_basic()
    build_moneyflow_cache()
