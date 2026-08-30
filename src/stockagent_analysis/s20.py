"""Path-dependent labels and score semantics for the S20 research index.

S20 is an independent research track.  It does not replace or feed the frozen
production R20 score.  Its target is the probability that an upside barrier is
reached before a downside barrier within 20 trading sessions.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_UPSIDE_PCT = (15.0, 20.0, 25.0)
DEFAULT_DOWNSIDE_PCT = -15.0


def _first_true_day(mask: np.ndarray) -> np.ndarray:
    """Return one-based first-hit day, or zero when a row never hits."""
    any_hit = mask.any(axis=1)
    day = np.argmax(mask, axis=1) + 1
    return np.where(any_hit, day, 0).astype(np.int16)


def build_first_passage_labels(
    entry_price: Iterable[float],
    future_high: np.ndarray,
    future_low: np.ndarray,
    *,
    upside_pct: Iterable[float] = DEFAULT_UPSIDE_PCT,
    downside_pct: float = DEFAULT_DOWNSIDE_PCT,
) -> pd.DataFrame:
    """Build competing-risk labels from daily highs and lows.

    Each high/low row contains the sessions after entry in chronological order.
    Event values are ``up_first``, ``down_first``, ``censored`` and
    ``ambiguous``.  A same-session touch of both barriers is ambiguous because
    daily OHLC data cannot reveal which barrier was reached first.
    """
    entry = np.asarray(list(entry_price), dtype=float)
    highs = np.asarray(future_high, dtype=float)
    lows = np.asarray(future_low, dtype=float)
    thresholds = tuple(float(value) for value in upside_pct)

    if highs.ndim != 2 or lows.ndim != 2 or highs.shape != lows.shape:
        raise ValueError("future_high and future_low must be equal 2-D arrays")
    if highs.shape[0] != entry.shape[0]:
        raise ValueError("entry_price row count must match price paths")
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("upside_pct must contain positive thresholds")
    if downside_pct >= 0:
        raise ValueError("downside_pct must be negative")
    if (
        not np.isfinite(entry).all()
        or not np.isfinite(highs).all()
        or not np.isfinite(lows).all()
        or (entry <= 0).any()
    ):
        raise ValueError("prices must be finite and entry_price must be positive")
    if (lows > highs).any():
        raise ValueError("future_low cannot exceed future_high")

    down_day = _first_true_day(lows <= entry[:, None] * (1 + downside_pct / 100))
    result: dict[str, np.ndarray] = {"down_day": down_day}

    for threshold in thresholds:
        label = f"{int(threshold) if threshold.is_integer() else threshold:g}"
        up_day = _first_true_day(highs >= entry[:, None] * (1 + threshold / 100))
        up_first = (up_day > 0) & ((down_day == 0) | (up_day < down_day))
        down_first = (down_day > 0) & ((up_day == 0) | (down_day < up_day))
        ambiguous = (up_day > 0) & (up_day == down_day)
        event = np.full(entry.shape[0], "censored", dtype=object)
        event[up_first] = "up_first"
        event[down_first] = "down_first"
        event[ambiguous] = "ambiguous"

        result[f"up{label}_day"] = up_day
        result[f"event{label}"] = event
        result[f"target{label}_up_first"] = up_first.astype(np.int8)

    return pd.DataFrame(result)


def s20_score(probability_up25_first: Iterable[float]) -> np.ndarray:
    """Map calibrated P(+25% before -15% by day 20) to the 0-100 S20 scale."""
    probability = np.asarray(list(probability_up25_first), dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return probability * 100.0
