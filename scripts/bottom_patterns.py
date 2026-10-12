#!/usr/bin/env python
"""Daily bottom-pattern flags and a 0-100 bottom score, as pre-registered in wiki/2026-10-12_bottom-pattern-lab.md.

  from bottom_patterns import pattern_panel
  pp = pattern_panel("20240801", "20261009", out_from="20250101")

Every flag on day D uses data up to D only (pivots need 3 bars on the right, so a pivot at t is known at t+3).
  B1   Chan-style MACD bottom divergence (green-bar area of the latest green run < 0.8 x the previous one,
       price at a new low, DIF below zero; the latest run still rising for 2 days or ended within 5 days)
  B2c  W double bottom, fresh close above the neckline     B2f  W double bottom forming (below the neckline)
  B3   two pin bars near the 20-day low within 10 days + MA5 turning up
  B4   RSI(14) bottom divergence    B5  KDJ oversold golden cross    B6  bullish engulfing / morning star at the 20-day low
  B7   volume dry-up then a +3% expansion day near the 60-day low
  zone close <= 85% of the 60-day high
  score = sum of weights (B1 25, B2c 25, B3 20, B4 15, B2f 10, B6 10, B7 10, B5 5), capped at 100, x0.6 outside the zone
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

WEIGHTS = {"B1": 25, "B2c": 25, "B3": 20, "B4": 15, "B2f": 10, "B6": 10, "B7": 10, "B5": 5}
FLAGS = list(WEIGHTS)


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def _pivots_low(l: np.ndarray) -> np.ndarray:
    s = pd.Series(l)
    return (s == s.rolling(7, center=True, min_periods=7).min()).to_numpy()


def one_stock(o, h, l, c, v):
    n = len(c)
    out = {k: np.zeros(n, bool) for k in FLAGS}
    out["B1_strength"] = np.full(n, np.nan)
    out["B2c_first"] = np.zeros(n, bool)   # post-hoc sensitivity (2026-10-12): first close above the neckline since L2
    if n < 90:
        out["zone"] = np.zeros(n, bool)
        return out
    dif = _ema(c, 12) - _ema(c, 26)
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().to_numpy()
    hist = 2 * (dif - dea)
    ma5 = pd.Series(c).rolling(5).mean().to_numpy()
    d = np.diff(c, prepend=np.nan)
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).to_numpy()
    lo9 = pd.Series(l).rolling(9, min_periods=9).min().to_numpy()
    hi9 = pd.Series(h).rolling(9, min_periods=9).max().to_numpy()
    rsv = np.where(hi9 > lo9, (c - lo9) / (hi9 - lo9) * 100, 50.0)
    K, Dk = np.full(n, 50.0), np.full(n, 50.0)
    for t in range(1, n):
        r = rsv[t] if np.isfinite(rsv[t]) else 50.0
        K[t] = 2 / 3 * K[t - 1] + r / 3
        Dk[t] = 2 / 3 * Dk[t - 1] + K[t] / 3
    J = 3 * K - 2 * Dk
    low20 = pd.Series(l).rolling(20, min_periods=20).min().to_numpy()
    low60 = pd.Series(l).rolling(60, min_periods=60).min().to_numpy()
    high60 = pd.Series(h).rolling(60, min_periods=60).max().to_numpy()
    v60 = pd.Series(v).shift(1).rolling(60, min_periods=60).mean().to_numpy()
    v10 = pd.Series(v).shift(1).rolling(10, min_periods=10).mean().to_numpy()
    v5 = pd.Series(v).shift(1).rolling(5, min_periods=5).mean().to_numpy()
    body = np.abs(c - o)
    rng = h - l
    pc = np.r_[np.nan, c[:-1]]
    lower = np.minimum(o, c) - l
    pin = (lower >= 2 * body) & (lower >= 0.6 * rng) & (rng / pc >= 0.03)
    piv = _pivots_low(l)
    piv_idx = np.where(piv)[0]

    # green runs of HIST < 0: start, end, prefix sums for area, running min of low and dif inside each run
    neg = hist < 0
    run_id = np.full(n, -1)
    starts, ends = [], []
    for t in range(n):
        if neg[t]:
            if t == 0 or not neg[t - 1]:
                starts.append(t)
                ends.append(t)
            ends[-1] = t
            run_id[t] = len(starts) - 1
    cs = np.cumsum(np.where(neg, -hist, 0.0))
    rmin_low = np.full(n, np.nan)
    rmin_dif = np.full(n, np.nan)
    for s, e in zip(starts, ends):
        rmin_low[s:e + 1] = np.minimum.accumulate(l[s:e + 1])
        rmin_dif[s:e + 1] = np.minimum.accumulate(dif[s:e + 1])
    starts, ends = np.array(starts), np.array(ends)

    def area(s, e):
        return cs[e] - (cs[s - 1] if s > 0 else 0.0)

    for D in range(80, n):
        # B1
        k = np.searchsorted(starts, D, side="right") - 1
        if k >= 1:
            s2, e2 = starts[k], ends[k]
            ok = False
            if s2 <= D <= e2:
                e2c = D
                ok = hist[D] > hist[D - 1] > hist[D - 2]
            else:
                e2c = e2
                ok = D - e2 <= 5
            s1, e1 = starts[k - 1], ends[k - 1]
            if ok and s1 >= D - 79:
                a1, a2 = area(s1, e1), area(s2, e2c)
                if (rmin_low[e2c] < rmin_low[e1]) and a1 > 0 and a2 < 0.8 * a1 and rmin_dif[e2c] < 0:
                    out["B1"][D] = True
                    out["B1_strength"][D] = 1 - a2 / a1
        # B2 (W double bottom)
        conf = piv_idx[piv_idx <= D - 3]
        l2c = conf[conf >= D - 24]
        if len(l2c):
            L2 = l2c[-1]
            cand = conf[(conf <= L2 - 8) & (conf >= L2 - 40)]
            for L1 in cand[::-1]:
                if abs(l[L2] / l[L1] - 1) > 0.04 or L1 < 40:
                    continue
                N = h[L1 + 1:L2].max()
                if N < 1.06 * max(l[L1], l[L2]) or h[L1 - 40:L1].max() < 1.15 * l[L1]:
                    continue
                if c[D] > N and (c[D - 5:D] <= N).any():
                    out["B2c"][D] = True
                    if not (c[L2 + 1:D] > N).any():
                        out["B2c_first"][D] = True
                if L2 >= D - 9 and 1.03 * l[L2] <= c[D] < N:
                    out["B2f"][D] = True
                break
        # B3 (two pins near the 20-day low + MA5 turn)
        w = np.arange(D - 9, D + 1)
        pins = w[pin[w] & (l[w] <= 1.03 * low20[D])]
        if len(pins) >= 2 and ma5[D] > ma5[D - 1]:
            if any(ma5[t] > ma5[t - 1] and ma5[t - 1] <= ma5[t - 2] for t in (D - 2, D - 1, D)):
                out["B3"][D] = True
        # B4 (RSI divergence)
        p2c = conf[conf >= D - 14]
        if len(p2c):
            P2 = p2c[-1]
            for P1 in conf[(conf <= P2 - 10) & (conf >= P2 - 60)]:
                if l[P2] < l[P1] and rsi[P2] > rsi[P1] + 3 and rsi[P1] < 35 and rsi[D] > rsi[P2]:
                    out["B4"][D] = True
                    break
        # B5 (KDJ)
        if J[D - 4:D + 1].min() < 0 and any(K[t] > Dk[t] and K[t - 1] <= Dk[t - 1] for t in (D - 1, D)):
            out["B5"][D] = True
        # B6 (engulfing / morning star at the 20-day low)
        for t in (D - 2, D - 1, D):
            at_low = any(l[x] <= low20[x] + 1e-9 for x in (t, t - 1, t - 2) if np.isfinite(low20[x]))
            if not at_low:
                continue
            engulf = c[t - 1] < o[t - 1] and c[t] > o[t] and o[t] <= c[t - 1] and c[t] >= o[t - 1]
            star = (c[t - 2] / c[t - 3] - 1 <= -0.03 and c[t - 2] < o[t - 2] and abs(c[t - 1] - o[t - 1]) / c[t - 2] <= 0.01
                    and c[t] > o[t] and c[t] > (o[t - 2] + c[t - 2]) / 2)
            if engulf or star:
                out["B6"][D] = True
                break
        # B7 (dry-up then expansion near the 60-day low)
        if (np.isfinite(v60[D]) and v10[D] <= 0.6 * v60[D] and v[D] >= 1.8 * v5[D] and c[D] > o[D]
                and c[D] / c[D - 1] - 1 >= 0.03 and c[D] <= 1.15 * low60[D]):
            out["B7"][D] = True
    out["zone"] = c <= 0.85 * high60
    return out


def _chunk(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for code, g in frame.groupby("ts_code", sort=False):
        r = one_stock(*(g[k].to_numpy(float) for k in ("open", "high", "low", "close", "vol")))
        parts.append(pd.DataFrame({"ts_code": code, "trade_date": g.trade_date.to_numpy(), **r}))
    return pd.concat(parts, ignore_index=True)


def score(pp: pd.DataFrame) -> pd.Series:
    s = sum(pp[k].astype(int) * w for k, w in WEIGHTS.items()).clip(upper=100)
    return (s * np.where(pp.zone, 1.0, 0.6)).round(1)


def pattern_panel(start: str, end: str, out_from: str, codes: set[str] | None = None, workers: int = 8) -> pd.DataFrame:
    from price_panel import adjusted_daily
    px = adjusted_daily(start, end, codes)
    groups = px.ts_code.unique()
    chunks = [px[px.ts_code.isin(set(groups[i::workers]))] for i in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        pp = pd.concat(ex.map(_chunk, chunks), ignore_index=True)
    pp = pp[pp.trade_date >= out_from].reset_index(drop=True)
    pp["bottom_score"] = score(pp)
    return pp
