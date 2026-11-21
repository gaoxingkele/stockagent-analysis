#!/usr/bin/env python
"""Rebuild the R20 training labels (entry_open, max_gain_20, max_dd_20, r20_close, is_clean) from the tushare daily cache.

  python scripts/rebuild_r20_labels_from_cache.py --check      # fidelity against output/labels/max_gain_labels.parquet (2024-01-02..2026-01-26)
  python scripts/rebuild_r20_labels_from_cache.py --extend     # write output/labels/max_gain_labels_ext_cache.parquet for 2026-01-27 .. last mature day

Same definition as extract_all_labels.py (which read raw TDX day files; D:/tdx is not on this machine): entry = next session open, window = the next
20 sessions' high / low, r20_close = close of session +20 over entry, all on RAW (unadjusted) prices, in percent.  The cache is raw prices as well.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "output/tushare_cache/daily"
LABELS = ROOT / "output/labels/max_gain_labels.parquet"
OUT = ROOT / "output/labels/max_gain_labels_ext_cache.parquet"
LOOK = 20


def load(start: str, end: str) -> pd.DataFrame:
    fs = [p for p in sorted(DAILY.glob("*.parquet")) if start <= p.stem <= end]
    d = pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True)
    d["trade_date"] = d.trade_date.astype(str)
    return d[["ts_code", "trade_date", "open", "high", "low", "close"]].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def labels(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for code, g in d.groupby("ts_code", sort=False):
        o, h, l, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        n = len(g)
        if n <= LOOK + 1:
            continue
        m = n - LOOK - 1                                    # signal rows 0..n-LOOK-2 have a full window and day +20
        hs = np.lib.stride_tricks.sliding_window_view(h, LOOK)[1:m + 1].max(axis=1)
        ls = np.lib.stride_tricks.sliding_window_view(l, LOOK)[1:m + 1].min(axis=1)
        eo = o[1:m + 1]
        ce = c[LOOK:LOOK + m]
        ok = eo > 0
        mg, md = (hs / eo - 1) * 100, (ls / eo - 1) * 100
        out.append(pd.DataFrame({"ts_code": code, "trade_date": g.trade_date.to_numpy()[:m], "entry_open": eo, "max_gain_20": mg.round(4),
                                 "max_dd_20": md.round(4), "r20_close": ((ce / eo - 1) * 100).round(4), "is_clean": (mg >= 20.0) & (md >= -3.0)})[ok])
    return pd.concat(out, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--extend", action="store_true")
    a = ap.parse_args()
    if a.check:
        old = pd.read_parquet(LABELS)
        old["trade_date"] = old.trade_date.astype(str)
        new = labels(load("20250101", "20260320"))
        new = new[new.trade_date.between("20250102", "20260126")]
        m = old.merge(new, on=["ts_code", "trade_date"], suffixes=("_old", "_new"))
        print(f"overlap rows {len(m):,} of old {len(old[old.trade_date.between('20250102', '20260126')]):,}; new {len(new):,}")
        for k in ("entry_open", "max_gain_20", "max_dd_20", "r20_close"):
            dlt = (m[k + "_new"] - m[k + "_old"]).abs()
            print(f"{k:12s} corr {m[k + '_old'].corr(m[k + '_new']):.5f}; |diff| median {dlt.median():.4f}, 99% {dlt.quantile(.99):.4f}; share within 0.1: {(dlt < 0.1).mean():.4f}")
        print(f"is_clean agreement {(m.is_clean_old == m.is_clean_new).mean():.4f}")
    if a.extend:
        new = labels(load("20260101", "20261029"))
        new = new[new.trade_date >= "20260127"]
        new.to_parquet(OUT, index=False)
        print(f"wrote {OUT}: {len(new):,} rows, {new.trade_date.min()}..{new.trade_date.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
