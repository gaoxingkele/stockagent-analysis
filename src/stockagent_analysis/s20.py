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


def build_daily_first_passage_labels(
    daily: pd.DataFrame,
    *,
    horizon_sessions: int = 20,
    upside_pct: Iterable[float] = DEFAULT_UPSIDE_PCT,
    downside_pct: float = DEFAULT_DOWNSIDE_PCT,
) -> pd.DataFrame:
    """Build S20 labels for one stock from chronologically ordered daily OHLC.

    The signal is observed on row ``t``. Entry is the next observed trading
    session's open, and the path contains that session through session 20.
    This exactly matches the existing max-gain label timing convention.
    """
    required = {"ts_code", "trade_date", "open", "high", "low", "close"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"missing daily columns: {sorted(missing)}")
    if horizon_sessions <= 0:
        raise ValueError("horizon_sessions must be positive")

    frame = daily.sort_values("trade_date").reset_index(drop=True).copy()
    if frame["ts_code"].astype(str).nunique() > 1:
        raise ValueError("daily frame must contain exactly one stock")
    if frame["trade_date"].astype(str).duplicated().any():
        raise ValueError("daily frame contains duplicate trade dates")
    if len(frame) <= horizon_sessions:
        return pd.DataFrame()

    numeric = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("daily OHLC values must be finite")

    window = horizon_sessions
    high_paths = np.lib.stride_tricks.sliding_window_view(
        numeric["high"].to_numpy()[1:], window
    )
    low_paths = np.lib.stride_tricks.sliding_window_view(
        numeric["low"].to_numpy()[1:], window
    )
    entry = numeric["open"].to_numpy()[1 : len(frame) - window + 1]
    close_at_horizon = numeric["close"].to_numpy()[window:]
    labels = build_first_passage_labels(
        entry,
        high_paths,
        low_paths,
        upside_pct=upside_pct,
        downside_pct=downside_pct,
    )

    thresholds = tuple(float(value) for value in upside_pct)
    max_gain = (high_paths.max(axis=1) / entry - 1) * 100
    max_dd = (low_paths.min(axis=1) / entry - 1) * 100
    prefix = pd.DataFrame(
        {
            "ts_code": frame["ts_code"].astype(str).iloc[: len(labels)].to_numpy(),
            "trade_date": frame["trade_date"].astype(str).iloc[: len(labels)].to_numpy(),
            "entry_date": frame["trade_date"].astype(str).iloc[1 : len(labels) + 1].to_numpy(),
            "horizon_end_date": frame["trade_date"].astype(str).iloc[window:].to_numpy(),
            "entry_open": entry,
            "max_gain_20": max_gain,
            "max_dd_20": max_dd,
            "r20_close": (close_at_horizon / entry - 1) * 100,
        }
    )
    result = pd.concat([prefix, labels], axis=1)
    for threshold in thresholds:
        label = f"{int(threshold) if threshold.is_integer() else threshold:g}"
        result[f"target{label}_window_safe"] = (
            (max_gain >= threshold) & (max_dd >= downside_pct)
        ).astype(np.int8)
        result[f"late_down_after_up{label}"] = (
            (result[f"up{label}_day"] > 0)
            & (result["down_day"] > result[f"up{label}_day"])
        ).astype(np.int8)
    return result


def s20_score(probability_up25_first: Iterable[float]) -> np.ndarray:
    """Map calibrated P(+25% before -15% by day 20) to the 0-100 S20 scale."""
    probability = np.asarray(list(probability_up25_first), dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return probability * 100.0


def cumulative_incidence(
    interval_probabilities: Iterable[np.ndarray],
) -> dict[str, np.ndarray]:
    """Combine interval probabilities [survive, upside, downside] into CIFs."""
    probabilities = [np.asarray(item, dtype=float) for item in interval_probabilities]
    if not probabilities:
        raise ValueError("at least one interval probability matrix is required")
    rows = probabilities[0].shape[0]
    survival = np.ones(rows, dtype=float)
    upside = np.zeros(rows, dtype=float)
    downside = np.zeros(rows, dtype=float)
    for probability in probabilities:
        if probability.shape != (rows, 3):
            raise ValueError("each interval matrix must have shape (rows, 3)")
        if (
            not np.isfinite(probability).all()
            or (probability < 0).any()
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6)
        ):
            raise ValueError("interval probabilities must be valid three-class rows")
        upside += survival * probability[:, 1]
        downside += survival * probability[:, 2]
        survival *= probability[:, 0]
    return {"upside": upside, "downside": downside, "survival": survival}
