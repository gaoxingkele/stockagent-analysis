#!/usr/bin/env python
"""Parameterised first-passage panel for the "pure up / pure down / chop" study.

Instead of freezing one (+U, -D) pair into a label, store for every
(ts_code, signal date) the first session on which each up/down barrier is
touched. Any three-state label (pure up / pure down / chop) for any U, D and
horizon H <= 20 can then be derived without rescanning prices.

Entry = next session open (T+1). Session k=1 is the entry day itself.
A value of 0 means "not touched within 20 sessions".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "output/tushare_cache/daily"
OUT = ROOT / "output/experiments/s20_pure_20260928"
UPS = (5, 8, 10, 15, 20, 25, 30)
DOWNS = (3, 5, 8, 10, 12, 15, 20)
H = 20


def _wide(frames: list[pd.DataFrame], col: str, dates, codes) -> np.ndarray:
    df = pd.concat([f[["trade_date", "ts_code", col]] for f in frames])
    return (
        df.pivot(index="trade_date", columns="ts_code", values=col)
        .reindex(index=dates, columns=codes)
        .to_numpy(dtype=np.float64)
    )


def main() -> int:
    files = sorted(DAILY.glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    for f in frames:
        f["trade_date"] = f["trade_date"].astype(str)
    dates = sorted({d for f in frames for d in f["trade_date"].unique()})
    codes = sorted({c for f in frames for c in f["ts_code"].unique() if c.endswith((".SH", ".SZ"))})
    o, h, l, c, pc = (_wide(frames, k, dates, codes) for k in ("open", "high", "low", "close", "pre_close"))
    n_dates, n_codes = o.shape
    print(f"panel {n_dates} dates x {n_codes} codes", flush=True)

    # board limit for the entry-day limit-up check (ST ignored; ST excluded downstream)
    limit = np.array([0.20 if cd.startswith(("300", "301", "688", "689")) else 0.10 for cd in codes])

    rows = []
    for t in range(n_dates - 1):
        e = t + 1
        end = min(e + H, n_dates)
        if end - e < H:  # horizon not complete
            break
        entry = o[e]
        valid = np.isfinite(entry) & (entry > 0)
        hi = h[e:end] / entry - 1.0   # (H, n_codes)
        lo = l[e:end] / entry - 1.0
        cl = c[e:end] / entry - 1.0
        rec = {
            "ts_code": np.array(codes),
            "trade_date": dates[t],
            "entry_limit_up": (o[e] >= pc[e] * (1 + limit) - 0.011).astype(np.int8),
            "ret5": cl[4], "ret10": cl[9], "ret20": cl[H - 1],
            "max_gain20": np.nanmax(hi, axis=0), "max_dd20": np.nanmin(lo, axis=0),
            "n_missing": np.isnan(cl).sum(axis=0).astype(np.int8),
        }
        for u in UPS:
            hit = hi >= u / 100
            rec[f"up{u}_day"] = np.where(hit.any(0), hit.argmax(0) + 1, 0).astype(np.int8)
        for d in DOWNS:
            hit = lo <= -d / 100
            rec[f"dn{d}_day"] = np.where(hit.any(0), hit.argmax(0) + 1, 0).astype(np.int8)
        df = pd.DataFrame(rec)
        rows.append(df[valid & (df["n_missing"].to_numpy() <= 2)])
    panel = pd.concat(rows, ignore_index=True)
    for col in ("ret5", "ret10", "ret20", "max_gain20", "max_dd20"):
        panel[col] = (panel[col] * 100).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "path_panel.parquet", index=False)
    print(f"rows {len(panel):,}  signal dates {panel.trade_date.min()}..{panel.trade_date.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
