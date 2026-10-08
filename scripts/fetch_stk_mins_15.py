#!/usr/bin/env python
"""Cache 15-minute bars (Tushare stk_mins) for the stocks that ever enter the S20 stage1 top 100.

30- and 60-minute bars are built from these (scripts/intraday_bars.py), so only 15 minutes is fetched.
Range 2025-02-01 .. 2026-08-05: the five sessions before the first S20 signal day (2025-03-03) through
the end of the confirmation window. Nothing from the reserved window (2026-08-06 on) is fetched.

Output: output/tushare_cache/min15/<ts_code>.parquet  (trade_time, open, high, low, close, vol, amount)
Resumable (finished stocks are skipped); backs off a minute on a rate-limit error.
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
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "output/tushare_cache/min15"
CHUNKS = [("2025-02-01 09:00:00", "2025-07-31 15:00:00"), ("2025-08-01 09:00:00", "2026-01-31 15:00:00"),
          ("2026-02-01 09:00:00", "2026-08-05 15:00:00")]


def codes() -> list[str]:
    from stockagent_analysis.s20_pure import PureConfig, select  # noqa: F401
    fr = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet", columns=["ts_code", "trade_date", "stage1_probability"])
    fr["r"] = fr.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    return sorted(fr.loc[fr.r <= 100, "ts_code"].unique())


def call(pro, **kw) -> pd.DataFrame:
    for attempt in range(5):
        try:
            return pro.stk_mins(**kw)
        except Exception as error:  # noqa: BLE001
            text = str(error)
            wait = 61 if ("每分钟" in text or "limit" in text.lower() or "频" in text) else 5 * (attempt + 1)
            print(f"  retry in {wait}s: {text[:100]}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up {kw}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--reverse", action="store_true", help="walk the list from the end (run a second process alongside)")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [c for c in codes() if not (OUT / f"{c}.parquet").exists()]
    if a.reverse:
        todo = todo[::-1]
    print(f"stocks to fetch: {len(todo)}", flush=True)
    t0 = time.time()
    for n, code in enumerate(todo, 1):
        if (OUT / f"{code}.parquet").exists():      # the other process got here first
            continue
        parts = []
        for s, e in CHUNKS:
            df = call(pro, ts_code=code, freq="15min", start_date=s, end_date=e)
            if df is not None and len(df):
                parts.append(df)
            time.sleep(a.sleep)
        if parts:
            df = pd.concat(parts, ignore_index=True).drop_duplicates("trade_time").sort_values("trade_time")
            df[["trade_time", "open", "high", "low", "close", "vol", "amount"]].to_parquet(OUT / f"{code}.parquet", index=False)
        else:
            pd.DataFrame(columns=["trade_time", "open", "high", "low", "close", "vol", "amount"]).to_parquet(OUT / f"{code}.parquet", index=False)
        if n % 50 == 0:
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(todo)}  {rate * 60:.0f} stocks/min, eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)
    print(f"done: {len(list(OUT.glob('*.parquet')))} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
