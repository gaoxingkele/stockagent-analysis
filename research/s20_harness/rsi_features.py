"""RSI-style split of upside vs downside. Not a trading signal by itself.

Wilder RSI maps average gain / average loss into a bounded unit. ATR mixes
those two magnitudes. S20 dual-high names are exactly the ATR failure mode:
large moves both ways. These features keep the ratio *and* the two magnitudes.
Lookback only; no future bars.
"""
from __future__ import annotations

import numpy as np

RSI_PERIOD = 14
RSI_COLUMNS = (
    "rsi14_wilder",
    "rsi_avg_gain14",
    "rsi_avg_loss14",
    "rsi_down_tr14",
)


def _require_lookback(close, high, low):
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    if close.ndim != 2 or close.shape != high.shape or close.shape != low.shape:
        raise ValueError("close/high/low must be (sessions, names)")
    if close.shape[0] < RSI_PERIOD + 1:
        raise ValueError("need period+1 sessions of lookback")
    return close, high, low


def wilder_avg(values, period=RSI_PERIOD):
    """Wilder smoothing: SMA seed on the first `period` rows, then recursive."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < period:
        raise ValueError("Wilder average needs a 2d window of length >= period")
    avg = values[:period].mean(axis=0)
    for t in range(period, values.shape[0]):
        avg = (avg * (period - 1) + values[t]) / period
    return avg


def split_close_moves(close):
    """Percent close-to-close gains and losses. No lookahead."""
    close = np.asarray(close, dtype=float)
    if close.ndim != 2 or close.shape[0] < 2:
        raise ValueError("need at least two close sessions")
    with np.errstate(divide="ignore", invalid="ignore"):
        change = close[1:] / close[:-1] - 1.0
    gain = np.clip(change, 0.0, None)
    loss = np.clip(-change, 0.0, None)
    return gain, loss, change


def rsi_from_averages(avg_gain, avg_loss):
    avg_gain = np.asarray(avg_gain, dtype=float)
    avg_loss = np.asarray(avg_loss, dtype=float)
    zero = (avg_gain == 0) & (avg_loss == 0)
    only_up = (avg_loss == 0) & (avg_gain > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > 0)
        rsi = 1.0 - 1.0 / (1.0 + rs)
    rsi = np.where(only_up, 1.0, rsi)
    rsi = np.where(zero, 0.5, rsi)
    return rsi


def down_true_range(close, high, low, change):
    prev = close[:-1]
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)))
    down = np.where(change < 0, tr, np.nan)
    with np.errstate(all="ignore"):
        count = np.sum(np.isfinite(down[-RSI_PERIOD:]), axis=0)
        total = np.nansum(down[-RSI_PERIOD:], axis=0)
        mean_down = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return np.where(np.isfinite(mean_down), mean_down, 0.0)


def compute(close, high, low):
    """Return a dict of (n_names,) arrays from a lookback window ending at the signal."""
    close, high, low = _require_lookback(close, high, low)
    gain, loss, change = split_close_moves(close)
    if gain.shape[0] < RSI_PERIOD:
        raise ValueError("need period close-to-close moves")
    avg_gain = wilder_avg(gain)
    avg_loss = wilder_avg(loss)
    last = close[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_pct = avg_gain
        loss_pct = avg_loss
        down_tr = down_true_range(close, high, low, change) / np.where(last > 0, last, np.nan)
    return {
        "rsi14_wilder": rsi_from_averages(avg_gain, avg_loss),
        "rsi_avg_gain14": gain_pct,
        "rsi_avg_loss14": loss_pct,
        "rsi_down_tr14": np.where(np.isfinite(down_tr), down_tr, np.nan),
    }


def shuffle_time(close, high, low, rng):
    """Permute lookback sessions together so OHLC relations survive, time order does not."""
    close, high, low = _require_lookback(close, high, low)
    order = rng.permutation(close.shape[0])
    return close[order], high[order], low[order]
