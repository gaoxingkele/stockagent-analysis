#!/usr/bin/env python
"""Bring the per-day Tushare caches up to the latest open trading day (first step of the daily run).

Per-day folders under output/tushare_cache/ (one parquet per trade date; missing days are fetched):
  daily, daily_basic, moneyflow, stk_limit, sw_daily (Shenwan indices), etf_daily (fund_daily)
Per-index files under output/tushare_cache/index_daily/ for the broad indices the market context uses
(the whole series is re-fetched from --start, so a file always ends at the latest day).

Usage: python scripts/update_daily_caches.py [--end YYYYMMDD]
Token from the project .env (TUSHARE_TOKEN). Market data only.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "output/tushare_cache"
BROAD = ["000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH", "000688.SH", "000016.SH"]


def call(fn, retries: int = 4, **kw) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            return fn(**kw)
        except Exception as error:  # noqa: BLE001
            text = str(error)
            wait = 61 if ("每分钟" in text or "limit" in text.lower()) else 5 * (attempt + 1)
            print(f"  retry in {wait}s: {text[:100]}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up on {kw}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    ap.add_argument("--index-start", default="20230101")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    last = max(p.stem for p in (CACHE / "daily").glob("*.parquet"))
    cal = call(pro.trade_cal, exchange="SSE", start_date=last, end_date=a.end, is_open="1")
    days = sorted(cal.cal_date.astype(str))
    apis = {"daily": pro.daily, "daily_basic": pro.daily_basic, "moneyflow": pro.moneyflow,
            "stk_limit": pro.stk_limit, "sw_daily": pro.sw_daily, "etf_daily": pro.fund_daily}
    for name, api in apis.items():
        folder = CACHE / name
        folder.mkdir(parents=True, exist_ok=True)
        for d in days:
            path = folder / f"{d}.parquet"
            if path.exists():
                continue
            df = call(api, trade_date=d)
            if df is None or df.empty:
                print(f"  {name} {d}: no rows yet", flush=True)
                continue
            df.to_parquet(path, index=False)
            print(f"  {name} {d}: {len(df)} rows", flush=True)
            time.sleep(0.3)
    folder = CACHE / "index_daily"
    for code in BROAD:
        df = call(pro.index_daily, ts_code=code, start_date=a.index_start, end_date=a.end)
        if df is not None and len(df):
            df.sort_values("trade_date").to_parquet(folder / f"{code}.parquet", index=False)
            print(f"  index {code}: through {df.trade_date.max()}", flush=True)
        time.sleep(0.3)
    print(f"caches up to {max(p.stem for p in (CACHE / 'daily').glob('*.parquet'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
