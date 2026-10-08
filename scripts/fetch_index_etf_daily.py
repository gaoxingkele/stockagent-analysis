#!/usr/bin/env python
"""Cache index, sector-index and ETF daily bars from Tushare, so market context can be built from real indices.

What it writes under output/tushare_cache/ (resumable: existing files are skipped):
  index_daily/<ts_code>.parquet   broad market indices and every CSI sector/theme index, full history from --start
  sw_daily/<YYYYMMDD>.parquet     Shenwan (2021) industry indices, all levels, one file per trading day
  etf_daily/<YYYYMMDD>.parquet    every exchange-traded fund (fund_daily), one file per trading day
  fund_basic_E.parquet, index_basic.parquet, sw_index_classify.parquet   name tables

Token from the project .env (TUSHARE_TOKEN). Rate-limit aware: sleeps between calls and backs off a
minute on a per-minute limit error. Market data only; nothing here scores a list.
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
BROAD = ["000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000905.SH", "000852.SH", "932000.CSI", "000688.SH",
         "000016.SH", "000906.SH", "000985.CSI", "399005.SZ", "399303.SZ", "399101.SZ", "399102.SZ", "000010.SH"]


def call(fn, retries: int = 4, **kw) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            return fn(**kw)
        except Exception as error:  # noqa: BLE001
            text = str(error)
            wait = 61 if ("每分钟" in text or "limit" in text.lower()) else 5 * (attempt + 1)
            print(f"  retry in {wait}s: {text[:100]}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up on {getattr(fn, '__name__', fn)} {kw}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--skip", nargs="*", default=[], choices=["index", "sw", "etf"])
    a = ap.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    days = [p.stem for p in sorted((CACHE / "daily").glob("*.parquet")) if a.start <= p.stem <= a.end]
    print(f"trading days in range from the stock cache: {len(days)} ({days[0]}..{days[-1]})", flush=True)

    # name tables
    basic = call(pro.index_basic, market="CSI")
    for market in ("SSE", "SZSE"):
        basic = pd.concat([basic, call(pro.index_basic, market=market)], ignore_index=True)
    basic.to_parquet(CACHE / "index_basic.parquet", index=False)
    sw = pd.concat([call(pro.index_classify, level=level, src="SW2021") for level in ("L1", "L2", "L3")], ignore_index=True)
    sw.to_parquet(CACHE / "sw_index_classify.parquet", index=False)
    funds = call(pro.fund_basic, market="E", status="L")
    funds.to_parquet(CACHE / "fund_basic_E.parquet", index=False)
    print(f"name tables: {len(basic)} indices, {len(sw)} SW industries, {len(funds)} listed funds", flush=True)

    if "index" not in a.skip:
        folder = CACHE / "index_daily"
        folder.mkdir(parents=True, exist_ok=True)
        sector = basic[(basic.market == "CSI") & basic.category.fillna("").str.contains("行业|主题|风格|策略")]
        codes = BROAD + sorted(set(sector.ts_code) - set(BROAD))
        print(f"index_daily: {len(codes)} indices ({len(BROAD)} broad + {len(codes) - len(BROAD)} CSI sector/theme/style)", flush=True)
        for n, code in enumerate(codes, 1):
            path = folder / f"{code}.parquet"
            if path.exists():
                continue
            frame = call(pro.index_daily, ts_code=code, start_date=a.start, end_date=a.end)
            if len(frame):
                frame.sort_values("trade_date").to_parquet(path, index=False)
            if n % 50 == 0:
                print(f"  index_daily {n}/{len(codes)}", flush=True)
            time.sleep(a.sleep)

    for name, api, skip in (("sw_daily", pro.sw_daily, "sw"), ("etf_daily", pro.fund_daily, "etf")):
        if skip in a.skip:
            continue
        folder = CACHE / name
        folder.mkdir(parents=True, exist_ok=True)
        todo = [d for d in days if not (folder / f"{d}.parquet").exists()]
        print(f"{name}: {len(todo)} days to fetch", flush=True)
        for n, day in enumerate(todo, 1):
            frame = call(api, trade_date=day)
            if len(frame):
                frame.to_parquet(folder / f"{day}.parquet", index=False)
            if n % 100 == 0:
                print(f"  {name} {n}/{len(todo)}", flush=True)
            time.sleep(a.sleep)
    for name in ("index_daily", "sw_daily", "etf_daily"):
        files = list((CACHE / name).glob("*.parquet"))
        print(f"{name}: {len(files)} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
