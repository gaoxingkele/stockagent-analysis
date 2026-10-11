#!/usr/bin/env python
"""Per-day Tushare caches for crowding events: margin_detail (financing balance per stock) and
top_list (dragon-tiger list: daily abnormal-move disclosures with net buying).

  python scripts/fetch_margin_toplist.py --start 20240701 --end 20260805

Writes output/tushare_cache/events/margin_detail/<date>.parquet and .../top_list/<date>.parquet;
existing days are skipped. Token from the project .env (TUSHARE_TOKEN). Market data only.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from update_daily_caches import call  # noqa: E402

EVENTS = ROOT / "output/tushare_cache/events"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--apis", default="margin_detail,top_list")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    days = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet") if a.start <= p.stem <= a.end)
    apis = {"margin_detail": pro.margin_detail, "top_list": pro.top_list}
    for name in a.apis.split(","):
        api = apis[name]
        folder = EVENTS / name
        folder.mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(days, 1):
            path = folder / f"{d}.parquet"
            if path.exists():
                continue
            df = call(api, trade_date=d)
            (df if df is not None else pd.DataFrame()).to_parquet(path, index=False)
            if i % 50 == 0:
                print(f"{name} {i}/{len(days)} {d}: {0 if df is None else len(df)} rows", flush=True)
        print(f"{name} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
