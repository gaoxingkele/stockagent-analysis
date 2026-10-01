#!/usr/bin/env python
"""Per-stock Eastmoney research-report history (AKShare), the analyst-revision source.

Tushare report_rc is capped at 10 calls/day for this token, so the history comes from
Eastmoney: every report with its date, rating and EPS forecasts for the next fiscal
years. Universe = every stock that was ever in the stage1 daily Top150 in the
evaluation window (or --codes). Resumable: one parquet per stock.

    python scripts/fetch_research_reports_em.py
Output: output/news/research_reports_em/<ts_code>.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd

pd.set_option("future.infer_string", False)
import akshare as ak  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/news/research_reports_em"


def universe(top: int = 150) -> list[str]:
    s = pd.read_parquet(ROOT / "output/experiments/s20_20r_stage1_forward_outcomes/scored_with_outcomes.parquet",
                        columns=["ts_code", "trade_date", "stage1_score"])
    s["r"] = s.groupby("trade_date").stage1_score.rank(ascending=False)
    codes = set(s[s.r <= top].ts_code)
    raw = ROOT / "output/experiments/s20_pure_v1_shadow/daily_lists_raw.csv"
    if raw.exists():
        codes |= set(pd.read_csv(raw, dtype={"ts_code": str}).ts_code)
    return sorted(codes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    codes = a.codes or universe()
    todo = [c for c in codes if not (OUT / f"{c}.parquet").exists()]
    print(f"{len(codes)} stocks, {len(todo)} to fetch", flush=True)
    for i, c in enumerate(todo, 1):
        for attempt in range(3):
            try:
                df = ak.stock_research_report_em(symbol=c.split(".")[0])
                df = df.astype(str)
                df["ts_code"] = c
                df["fetched_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                df.to_parquet(OUT / f"{c}.parquet", index=False)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    print(f"  {c} failed: {str(exc)[:80]}", flush=True)
                    pd.DataFrame({"ts_code": [c]}).to_parquet(OUT / f"{c}.parquet", index=False)  # mark as tried
                time.sleep(3)
        if i % 50 == 0:
            print(f"  [{i}/{len(todo)}]", flush=True)
        time.sleep(0.5)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
