#!/usr/bin/env python
"""Backfill event-type Tushare data for the risk gates and analyst revisions.

One parquet per query key under output/tushare_cache/events/<api>/ (resumable).
Point-in-time use: every consumer must filter on the announcement date
(ann_date / report_date) <= signal date, never on the event date alone.

  share_float      by float_date (calendar days)  -> unlock calendar incl. old announcements
  stk_holdertrade  by ann_date   (calendar days)  -> holder increases / reductions
  forecast         by ann_date   (calendar days)  -> earnings pre-announcements
  report_rc        by report_date(calendar days)  -> analyst EPS forecasts and ratings
  disclosure_date  by end_date   (report periods) -> scheduled / actual report dates

    python scripts/fetch_tushare_events.py --start 20240701 --end 20260930
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
OUT = ROOT / "output/tushare_cache/events"
PERIODS = ["20240630", "20240930", "20241231", "20250331", "20250630", "20250930", "20251231",
           "20260331", "20260630", "20260930"]


PAGE = 6000   # Tushare caps a single query at 6000 rows; page with offset when a page is full


def call(pro, api: str, **kw) -> pd.DataFrame | None:
    pages, offset = [], 0
    while True:
        df = call_once(pro, api, offset=offset, limit=PAGE, **kw)
        if df is None:
            return None if not pages else pd.concat(pages, ignore_index=True)
        pages.append(df)
        if len(df) < PAGE:
            return pd.concat(pages, ignore_index=True)
        offset += PAGE
        time.sleep(0.3)


def call_once(pro, api: str, **kw) -> pd.DataFrame | None:
    for attempt in range(5):
        try:
            return getattr(pro, api)(**kw)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            wait = 30 if ("每分钟" in msg or "频率" in msg or "RATE" in msg.upper()) else 3 * (attempt + 1)
            print(f"  {api} {kw} retry {attempt + 1}: {msg[:60]} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    return None


def run(pro, api: str, key: str, values: list[str], sleep: float, extra_end: str | None = None) -> None:
    out = OUT / api
    out.mkdir(parents=True, exist_ok=True)
    # a stored file with exactly a full page was truncated by the row cap: fetch it again
    for v in values:
        f = out / f"{v}.parquet"
        if f.exists() and len(pd.read_parquet(f, columns=None)) == PAGE:
            f.unlink()
    todo = [v for v in values if not (out / f"{v}.parquet").exists()]
    print(f"{api}: {len(values)} keys, {len(todo)} to fetch", flush=True)
    n = 0
    for i, v in enumerate(todo, 1):
        df = call(pro, api, **{key: v})
        if df is not None:
            df.to_parquet(out / f"{v}.parquet", index=False)   # empty days are stored too (marks them done)
            n += len(df)
        if i % 100 == 0:
            print(f"  {api} [{i}/{len(todo)}] rows so far {n:,}", flush=True)
        time.sleep(sleep)
    print(f"{api}: done, {n:,} rows", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--float-end", default="20261231", help="unlock calendar horizon")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    days = pd.date_range(a.start, a.end).strftime("%Y%m%d").tolist()
    float_days = pd.date_range(a.start, a.float_end).strftime("%Y%m%d").tolist()
    jobs = [("disclosure_date", "end_date", PERIODS), ("share_float", "float_date", float_days),
            ("forecast", "ann_date", days), ("stk_holdertrade", "ann_date", days),
            ("report_rc", "report_date", days)]
    for api, key, values in jobs:
        if a.only and api not in a.only:
            continue
        run(pro, api, key, values, a.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
