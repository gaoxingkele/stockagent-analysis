#!/usr/bin/env python
"""Per (stock, signal day) labels and exit variants for the 2026-10-11 alternative-design trial.

  python scripts/build_s20_alt_paths.py

Same conventions as build_s20_pure_path_panel.py: raw daily bars, entry = next session open, session 1
is the entry day, 20 sessions, a same-session double touch is booked as the adverse side.
Columns:
  natr14, rv20          signal-day NATR(14) (fraction) and the std of the last 20 daily returns (%)
  lab_atr               1 = +3*ATR before -2*ATR within 20 sessions, 0 = the reverse, -1 = neither
  lab_exc, exc20        industry-excess path (stock close / entry over equal-weight industry index since
                        the signal close): 1 = +10% before -10%, 0 = the reverse, -1 = neither; exc20 at day 20 (%)
  ret_base              U15D10 (+15/-10/20d), cost 0.3 (equals s20_pure_history.outcomes().ret_v1)
  ret_x1                time stop: no touch by day 10 -> exit at the day-10 close
  ret_x2                half off at +10%, the other half as U15D10
  ret_x3                after +10% the stop moves to entry (0%), from the next session on
  ret20, up15_day, dn10_day, horizon_end_date
Signal days 2024-01-02..2026-08-05 (later signal days are the reserved window and are not built).
Output: output/experiments/s20_alt_designs_20261011/paths.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/s20_alt_designs_20261011"
H, COST, LAST_SIGNAL = 20, 0.3, "20260805"


def first(hit: np.ndarray) -> np.ndarray:
    return np.where(hit.any(0), hit.argmax(0) + 1, 0)


def race(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    return np.where((up > 0) & ((dn == 0) | (up < dn)), 1, np.where(dn > 0, 0, -1))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")) if "20231101" <= f.stem <= "20260915"]
    df = pd.concat([pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close"]) for f in files])
    df = df[df.ts_code.str.endswith((".SH", ".SZ"))]
    df["trade_date"] = df.trade_date.astype(str)
    dates = sorted(df.trade_date.unique())
    codes = sorted(df.ts_code.unique())
    W = {k: df.pivot(index="trade_date", columns="ts_code", values=k).reindex(index=dates, columns=codes).to_numpy(np.float64)
         for k in ("open", "high", "low", "close", "pre_close")}
    o, h, l, c, pc = W["open"], W["high"], W["low"], W["close"], W["pre_close"]
    del df
    print(f"panel {len(dates)} dates x {len(codes)} codes", flush=True)

    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    natr = pd.DataFrame(tr).rolling(14, min_periods=10).mean().to_numpy() / c
    r = c / pc - 1
    rv20 = pd.DataFrame(r).rolling(20, min_periods=15).std().to_numpy() * 100

    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "industry"]).drop_duplicates("ts_code")
    ind = pd.Series(codes).map(basic.set_index("ts_code").industry).fillna("NA").to_numpy()
    groups = {g: np.where(ind == g)[0] for g in np.unique(ind)}
    ind_ret = np.zeros_like(r)
    for g, idx in groups.items():
        m = np.nanmean(np.where(np.abs(r[:, idx]) < 0.35, r[:, idx], np.nan), axis=1)
        ind_ret[:, idx] = np.nan_to_num(m)[:, None]
    ind_lvl = np.cumprod(1 + ind_ret, axis=0)

    rows = []
    for t, d in enumerate(dates):
        if d < "20240102" or d > LAST_SIGNAL:
            continue
        e = t + 1
        if e + H > len(dates):
            break
        entry = o[e]
        valid = np.isfinite(entry) & (entry > 0) & np.isfinite(natr[t])
        hi, lo, cl = h[e:e + H] / entry - 1, l[e:e + H] / entry - 1, c[e:e + H] / entry - 1
        n_missing = np.isnan(cl).sum(0)
        up15, dn10, up10 = first(hi >= 0.15), first(lo <= -0.10), first(hi >= 0.10)
        ret20 = cl[H - 1] * 100
        base = np.where((up15 > 0) & ((dn10 == 0) | (up15 < dn10)), 15.0, np.where(dn10 > 0, -10.0, ret20))
        first_ev = np.where(up15 > 0, np.where(dn10 > 0, np.minimum(up15, dn10), up15), dn10)
        x1 = np.where((first_ev > 0) & (first_ev <= 10), base, cl[9] * 100)
        up10_first = (up10 > 0) & ((dn10 == 0) | (up10 < dn10))
        x2 = np.where(up10_first, 0.5 * 10 + 0.5 * base, base)
        k = np.arange(1, H + 1)[:, None]
        be = first((lo <= 0) & (k > up10[None, :]) & (up10[None, :] > 0))
        x3_up = np.where((up15 > 0) & ((be == 0) | (up15 < be)), 15.0, np.where(be > 0, 0.0, ret20))
        x3 = np.where(up10_first, x3_up, base)
        a = natr[t]
        lab_atr = race(first(hi >= 3 * a), first(lo <= -2 * a))
        exc = (c[e:e + H] / entry) / (ind_lvl[e:e + H] / ind_lvl[t]) - 1
        lab_exc = race(first(exc >= 0.10), first(exc <= -0.10))
        rec = pd.DataFrame({"ts_code": codes, "trade_date": d, "horizon_end_date": dates[e + H - 1],
                            "natr14": a.astype(np.float32), "rv20": rv20[t].astype(np.float32),
                            "lab_atr": lab_atr.astype(np.int8), "lab_exc": lab_exc.astype(np.int8),
                            "exc20": (exc[H - 1] * 100).astype(np.float32), "ret20": ret20.astype(np.float32),
                            "up15_day": up15.astype(np.int8), "dn10_day": dn10.astype(np.int8),
                            "ret_base": (base - COST).astype(np.float32), "ret_x1": (x1 - COST).astype(np.float32),
                            "ret_x2": (x2 - COST).astype(np.float32), "ret_x3": (x3 - COST).astype(np.float32)})
        rows.append(rec[valid & (n_missing <= 2)])
    panel = pd.concat(rows, ignore_index=True)
    panel.to_parquet(OUT / "paths.parquet", index=False)
    print(f"rows {len(panel):,} signal days {panel.trade_date.min()}..{panel.trade_date.max()}", flush=True)
    for col in ("lab_atr", "lab_exc"):
        print(col, panel[col].value_counts(normalize=True).round(3).to_dict(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
