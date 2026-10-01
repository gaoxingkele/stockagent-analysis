#!/usr/bin/env python
"""Fetch Tushare report_rc (analyst EPS forecasts) under its tight rate limit.

This token allows report_rc 1 call/minute and 10 calls/hour, so instead of one call
per day the data is pulled by month windows (start_date/end_date) with offset paging,
one call every 6.5 minutes. Resumable: each (month, page) is one file.

    python scripts/fetch_report_rc_slow.py --start 202407 --end 202609
Output: output/tushare_cache/events/report_rc_month/<YYYYMM>_p<k>.parquet
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
OUT = ROOT / "output/tushare_cache/events/report_rc_month"
PAGE, GAP = 6000, 390   # seconds between calls: 10/hour with margin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    OUT.mkdir(parents=True, exist_ok=True)
    months = pd.period_range(a.start[:4] + "-" + a.start[4:], a.end[:4] + "-" + a.end[4:], freq="M")
    for m in months:
        s, e = m.start_time.strftime("%Y%m%d"), m.end_time.strftime("%Y%m%d")
        k = 0
        while True:
            f = OUT / f"{m.strftime('%Y%m')}_p{k}.parquet"
            if f.exists():
                if len(pd.read_parquet(f)) < PAGE:
                    break
                k += 1
                continue
            try:
                df = pro.report_rc(start_date=s, end_date=e, offset=k * PAGE, limit=PAGE)
            except Exception as exc:  # noqa: BLE001
                print(f"{m} p{k}: {str(exc)[:80]} -> wait", flush=True)
                time.sleep(GAP)
                continue
            df.to_parquet(f, index=False)
            print(f"{m} p{k}: {len(df)} rows", flush=True)
            time.sleep(GAP)
            if len(df) < PAGE:
                break
            k += 1
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
