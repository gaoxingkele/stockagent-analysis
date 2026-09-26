"""S20 opportunity/risk research labels; no production score dependencies."""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NAMES = (
    "immediate20", "immediate15", "delayed20",
    "negative_flat", "negative_down", "negative_unsafe15",
)


def path_labels(entry, highs, lows):
    """Twenty-session labels relative to D+1 open, with conservative OHLC ties.

    A +20% first touch completes the opportunity; later falls do not undo it.
    The 15--20% fallback requires the entire window to stay above -10%.
    A same-day first stop/target touch has unknown order and is excluded.
    """
    entry = np.asarray(entry, dtype=float)
    highs, lows = np.asarray(highs, dtype=float), np.asarray(lows, dtype=float)
    if (highs.ndim != 2 or highs.shape != lows.shape or highs.shape[1] == 0
            or len(entry) != len(highs) or (entry <= 0).any()
            or not np.isfinite(entry).all() or not np.isfinite(highs).all()
            or not np.isfinite(lows).all() or (lows > highs).any()):
        raise ValueError("invalid entry or OHLC paths")
    hit = highs >= entry[:, None] * 1.20
    stop = lows < entry[:, None] * .90
    hd = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)
    sd = np.where(stop.any(axis=1), stop.argmax(axis=1) + 1, 0)
    gain = (highs.max(axis=1) / entry - 1) * 100
    mae = (lows.min(axis=1) / entry - 1) * 100
    hit15 = (highs >= entry[:, None] * 1.15).any(axis=1)
    cls = np.full(len(entry), 3, dtype=np.int8)
    cls[(hd == 0) & (sd > 0)] = 4
    cls[(hd == 0) & hit15 & (sd == 0)] = 1
    cls[(hd == 0) & hit15 & (sd > 0)] = 5
    cls[(hd > 0) & ((sd == 0) | (sd > hd))] = 0
    cls[(hd > 0) & (sd > 0) & (sd < hd)] = 2
    cls[(hd > 0) & (sd == hd)] = -1
    return pd.DataFrame({
        "s20_class": cls,
        "reason": [CLASS_NAMES[c] if c >= 0 else "ambiguous_same_day" for c in cls],
        "hit20_day": hd, "stop10_day": sd,
        "max_gain20": gain, "window_mae20": mae,
        "immediate": np.where(cls < 0, -1, np.isin(cls, [0, 1]).astype(int)),
        "opportunity": np.where(cls < 0, -1, np.isin(cls, [0, 1, 2]).astype(int)),
        "down_risk": np.where(cls < 0, -1, np.isin(cls, [2, 4, 5]).astype(int)),
        "negative": np.where(cls < 0, -1, np.isin(cls, [3, 4, 5]).astype(int)),
    })


def daily_labels(daily, horizon=20):
    frame = daily.sort_values("trade_date").reset_index(drop=True)
    if frame.ts_code.nunique() != 1 or frame.trade_date.duplicated().any():
        raise ValueError("expected one stock with unique trading dates")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(frame) <= horizon:
        return pd.DataFrame()
    n = len(frame) - horizon
    entry = frame.open.to_numpy(dtype=float)[1:n + 1]
    high = np.lib.stride_tricks.sliding_window_view(frame.high.to_numpy(dtype=float)[1:], horizon)
    low = np.lib.stride_tricks.sliding_window_view(frame.low.to_numpy(dtype=float)[1:], horizon)
    labels = path_labels(entry, high, low)
    prefix = pd.DataFrame({
        "ts_code": frame.ts_code.iloc[:n].astype(str).to_numpy(),
        "trade_date": frame.trade_date.iloc[:n].astype(str).to_numpy(),
        "entry_date": frame.trade_date.iloc[1:n + 1].astype(str).to_numpy(),
        "horizon_end_date": frame.trade_date.iloc[horizon:].astype(str).to_numpy(),
        "entry_open": entry,
        "close_return20": (frame.close.to_numpy(dtype=float)[horizon:] / entry - 1) * 100,
    })
    return pd.concat([prefix, labels], axis=1)


def probability_outputs(probabilities, risk_weight=1.0):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 6 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("expected normalized six-class probabilities")
    good = p[:, 0] + p[:, 1]
    down = p[:, 2] + p[:, 4] + p[:, 5]
    if not np.isfinite(risk_weight) or risk_weight < 0:
        raise ValueError("risk weight must be finite and nonnegative")
    return pd.DataFrame({
        "p_immediate": good, "p_delayed": p[:, 2],
        "p_opportunity": good + p[:, 2], "p_negative": p[:, 3:].sum(axis=1),
        "p_down": down, "p_flat": p[:, 3],
        "score": 100 * (risk_weight + good - risk_weight * down) / (1 + risk_weight),
    })


def select_low_correlation(features, importance, correlation, max_abs=.7, limit=24):
    """Greedy representatives from fit-only Spearman matrix, not independence."""
    selected = []
    for feature in sorted(features, key=lambda f: (-importance.get(f, 0), f)):
        if importance.get(feature, 0) <= 0:
            continue
        if all(abs(correlation.loc[feature, other]) <= max_abs for other in selected):
            selected.append(feature)
        if len(selected) >= limit:
            break
    return selected
