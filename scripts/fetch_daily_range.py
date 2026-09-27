"""Fill missing trading days in output/tushare_cache/daily for a date range.

Requires TUSHARE_TOKEN in the project .env (never committed). Existing files are
skipped, so the script is idempotent and safe to run as a daily backfill.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import tushare as ts

CACHE = ROOT / "output" / "tushare_cache" / "daily"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill missing daily quotes")
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env", override=False)
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("TUSHARE_TOKEN is not set in .env", flush=True)
        return 2
    pro = ts.pro_api(token)
    calendar = pro.trade_cal(exchange="SSE", start_date=args.start, end_date=args.end,
                             is_open="1")
    dates = [str(day) for day in calendar["cal_date"].tolist()]
    fetched, skipped = 0, 0
    for day in dates:
        path = CACHE / f"{day}.parquet"
        if path.exists():
            skipped += 1
            continue
        frame = pro.daily(trade_date=day)
        if frame is None or frame.empty:
            print(f"{day}: empty response", flush=True)
            continue
        frame.to_parquet(path, index=False)
        fetched += 1
        print(f"{day}: {len(frame)} stocks", flush=True)
        time.sleep(args.sleep)
    print(f"done: fetched={fetched} skipped={skipped} of {len(dates)} trading days",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
