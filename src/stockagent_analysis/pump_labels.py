"""Clean-start labels for the pump v5 start-up / start-down classifier.

A start-up is a clean move, not a big one (wiki/2026-10-06_pump-ratio-review.md section 6, method C):
over the next `horizon` sessions after the signal-day close
  * the close path is efficient: (close[t+h] - close[t]) / sum |close[t+k] - close[t+k-1]| >= efficiency
  * the net move is at least `min_move_atr` ATR(14) (the stock's own volatility is the unit, no fixed %)
  * no reverse action: the lowest low stays above close[t] - `max_reverse_atr` * ATR
A start-down is the mirror image. Everything else is neutral. Classes: 0 neutral, 1 down, 2 up.

ATR(14) is known at the signal-day close; the label looks only forward, so it does not reward a trend
that has already happened (unlike "MA5 keeps rising", whose first comparisons are against past closes).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanStart:
    horizon: int = 5
    efficiency: float = 0.6
    min_move_atr: float = 1.0
    max_reverse_atr: float = 0.5
    atr_window: int = 14


def clean_start_labels(px: pd.DataFrame, spec: CleanStart = CleanStart()) -> pd.DataFrame:
    """px: ts_code, trade_date, high, low, close, pre_close for every session (unsampled, sorted or not).

    Returns ts_code, trade_date, atr14, efficiency, net_atr, label (NaN where the horizon is incomplete).
    """
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = px.groupby("ts_code", sort=False)
    tr = np.maximum(px.high - px.low, np.maximum((px.high - px.pre_close).abs(), (px.low - px.pre_close).abs()))
    atr = tr.groupby(px.ts_code, sort=False).transform(lambda s: s.rolling(spec.atr_window, min_periods=10).mean())
    close = px.close
    h = spec.horizon
    future = [g.close.shift(-k) for k in range(h + 1)]                 # future[0] is today's close
    path = sum((future[k] - future[k - 1]).abs() for k in range(1, h + 1))
    net = future[h] - close
    eff = net / path.replace(0, np.nan)
    low_next = pd.concat([g.low.shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1, skipna=False)
    high_next = pd.concat([g.high.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1, skipna=False)
    up = (eff >= spec.efficiency) & (net >= spec.min_move_atr * atr) & (low_next >= close - spec.max_reverse_atr * atr)
    dn = (eff <= -spec.efficiency) & (-net >= spec.min_move_atr * atr) & (high_next <= close + spec.max_reverse_atr * atr)
    label = np.where(up, 2, np.where(dn, 1, 0)).astype(float)
    label[(future[h].isna() | atr.isna() | low_next.isna()).to_numpy()] = np.nan
    return pd.DataFrame({"ts_code": px.ts_code, "trade_date": px.trade_date, "atr14": atr,
                         "efficiency": eff, "net_atr": net / atr, "label": label})
