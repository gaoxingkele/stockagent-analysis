#!/usr/bin/env python
"""Backfill per-trading-day Tushare endpoints into output/tushare_cache/<api>/<YYYYMMDD>.parquet.

Idempotent and resumable (existing day files are skipped), rate-limit aware.
Token from the project .env (TUSHARE_TOKEN, never committed).

    python scripts/fetch_tushare_by_date.py daily_basic --start 20240101 --end 20260930
    python scripts/fetch_tushare_by_date.py daily stk_limit --start 20150101 --end 20231231
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "output/tushare_cache"
FIELDS = {
    "daily_basic": "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps_ttm,"
                   "dv_ttm,total_share,float_share,free_share,total_mv,circ_mv",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("apis", nargs="+", help="per-date endpoints, e.g. daily daily_basic stk_limit margin_detail")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sleep", type=float, default=0.35)
    a = ap.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    days = pro.trade_cal(exchange="SSE", start_date=a.start, end_date=a.end, is_open="1")["cal_date"].astype(str)
    days = sorted(days.tolist())
    for api in a.apis:
        out = CACHE / api
        out.mkdir(parents=True, exist_ok=True)
        todo = [d for d in days if not (out / f"{d}.parquet").exists()]
        print(f"{api}: {len(days)} trading days, {len(todo)} to fetch", flush=True)
        t0, ok, empty = time.time(), 0, 0
        for i, d in enumerate(todo, 1):
            for attempt in range(5):
                try:
                    kw = {"trade_date": d}
                    if api in FIELDS:
                        kw["fields"] = FIELDS[api]
                    pages, offset = [], 0
                    while True:     # Tushare caps one query at 6000 rows: page until a short page
                        page = getattr(pro, api)(offset=offset, limit=6000, **kw)
                        pages.append(page)
                        if page is None or len(page) < 6000:
                            break
                        offset += 6000
                    df = pd.concat([x for x in pages if x is not None], ignore_index=True)
                    break
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    wait = 30 if ("每分钟" in msg or "频率" in msg or "RATE" in msg.upper()) else 3 * (attempt + 1)
                    print(f"  {api} {d} retry {attempt + 1}: {msg[:60]} (sleep {wait}s)", flush=True)
                    time.sleep(wait)
            else:
                print(f"  {api} {d} FAILED", flush=True)
                continue
            if df is None or df.empty:
                empty += 1
            else:
                df.to_parquet(out / f"{d}.parquet", index=False)
                ok += 1
            if i % 100 == 0:
                print(f"  {api} [{i}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
            time.sleep(a.sleep)
        print(f"{api}: fetched {ok}, empty {empty}, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
