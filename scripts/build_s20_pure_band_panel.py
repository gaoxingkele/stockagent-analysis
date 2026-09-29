#!/usr/bin/env python
"""Band exit panel: take-profit is a range [a, b], not a point.

For every (ts_code, signal date), entry = next open, horizon 20 sessions:
  leg 1  crash line -D touched before +a      -> whole position exits at -D ("bad")
  leg 2  +a touched first                     -> half exits at +a, stop moves to entry;
         the other half exits at +b if touched before price falls back to entry,
         at 0 if it falls back first, else at the day-20 close
  none   neither touched                      -> exit at the day-20 close
Stored per (a, b, D): outcome code and band return (before cost), plus the
20-session max drawdown for the "never crash" check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "output/tushare_cache/daily"
OUT = ROOT / "output/experiments/s20_pure_20260928"
AS = (3, 5, 8)
BS = (10, 15, 20)
DS = (8, 10, 12, 15)
H = 20


def _wide(frames, col, dates, codes):
    df = pd.concat([f[["trade_date", "ts_code", col]] for f in frames])
    return df.pivot(index="trade_date", columns="ts_code", values=col).reindex(index=dates, columns=codes).to_numpy(float)


def first(mask: np.ndarray) -> np.ndarray:
    return np.where(mask.any(0), mask.argmax(0) + 1, 0)


def main() -> int:
    frames = [pd.read_parquet(f) for f in sorted(DAILY.glob("*.parquet"))]
    for f in frames:
        f["trade_date"] = f["trade_date"].astype(str)
    dates = sorted({d for f in frames for d in f.trade_date.unique()})
    codes = sorted({c for f in frames for c in f.ts_code.unique() if c.endswith((".SH", ".SZ"))})
    o, h, l, c = (_wide(frames, k, dates, codes) for k in ("open", "high", "low", "close"))
    rows = []
    for t in range(len(dates) - H - 1 + 1):
        e = t + 1
        if e + H > len(dates):
            break
        entry = o[e]
        valid = np.isfinite(entry) & (entry > 0)
        hi, lo, cl = h[e:e + H] / entry - 1, l[e:e + H] / entry - 1, c[e:e + H] / entry - 1
        ok = valid & (np.isnan(cl).sum(0) <= 2)
        close20 = cl[H - 1]
        rec = {"ts_code": np.array(codes)[ok], "trade_date": dates[t],
               "maxdd20": (np.nanmin(lo, 0) * 100)[ok].astype(np.float32),
               "ret20": (close20 * 100)[ok].astype(np.float32)}
        days = np.arange(1, H + 1)[:, None]
        for a in AS:
            ua = first(hi >= a / 100)
            for D in DS:
                dd = first(lo <= -D / 100)
                bad = (dd > 0) & ((ua == 0) | (dd <= ua))        # same-day -> assume crash first
                win = (ua > 0) & ~bad
                for b in BS:
                    # second half after the +a session: +b vs back-to-entry, both from the next session
                    after = days > ua[None, :]
                    ub = first((hi >= b / 100) & after)
                    be = first((lo <= 0) & after)
                    ub_on_a_day = (ua > 0) & (hi[np.clip(ua - 1, 0, H - 1), np.arange(len(ua))] >= b / 100)
                    second = np.where(ub_on_a_day, b,
                                      np.where((ub > 0) & ((be == 0) | (ub < be)), b,
                                               np.where(be > 0, 0.0, close20 * 100)))
                    ret = np.where(bad, -D, np.where(win, 0.5 * a + 0.5 * second, close20 * 100))
                    code = np.where(bad, 2, np.where(win, 1, np.where(close20 > 0, 3, 0)))
                    rec[f"ret_a{a}_b{b}_d{D}"] = ret[ok].astype(np.float32)
                    if b == BS[0]:
                        rec[f"cls_a{a}_d{D}"] = code[ok].astype(np.int8)  # 1 win, 3 small win, 0 small loss, 2 bad
        rows.append(pd.DataFrame(rec))
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(OUT / "band_panel.parquet", index=False)
    print(f"rows {len(panel):,}  signal dates {panel.trade_date.min()}..{panel.trade_date.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
