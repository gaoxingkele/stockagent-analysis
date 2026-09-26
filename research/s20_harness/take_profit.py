"""Take-profit inside a 20-session window. Holding to D+20 is the fallback, not the only win.

Preregistered targets only. Same-day OHLC order is unknown: those paths stay in
the set with bounds. T+1: the buy session cannot be the sell session.
"""
from __future__ import annotations

import numpy as np

from .labels import B5_RELATIVE, P_TRACK_BUY_COST, P_TRACK_SELL_COST

# User-paired regimes (not mixed, not searched on the evaluation window):
#   +15% expected profit  ↔  −10% path risk
#   +25% expected profit  ↔  −15% path risk
TARGET_RISK_PAIRS = ((0.15, 0.10), (0.25, 0.15))
TAKE_PROFIT_PCTS = tuple(tp for tp, _ in TARGET_RISK_PAIRS)
PRIMARY_TP = 0.15
DRAWDOWN_FLOORS = (0.05, 0.08, 0.10, 0.15)
USER_DRAWDOWN_BAND = (0.10, 0.15)
PRIMARY_DRAWDOWN = 0.10
SILENCE_MAX_GAIN = 0.08


def window_mae(entry, low) -> float:
    low = np.asarray(low, dtype=float)
    finite = low[np.isfinite(low)]
    if finite.size == 0 or not np.isfinite(entry) or entry <= 0:
        return float("nan")
    return float(finite.min() / float(entry) - 1.0)


def window_max_gain(entry, high) -> float:
    high = np.asarray(high, dtype=float)
    finite = high[np.isfinite(high)]
    if finite.size == 0 or not np.isfinite(entry) or entry <= 0:
        return float("nan")
    return float(finite.max() / float(entry) - 1.0)


def is_silent(max_gain, path_risk, *, cut=SILENCE_MAX_GAIN) -> bool:
    """Dead-money path: never printed `cut` and did not breach the paired floor."""
    if max_gain != max_gain:
        return False
    return bool(max_gain < float(cut)) and not bool(path_risk)


def specified_rise(result) -> bool | None:
    """Captured +15% take-profit after T+1. Horizon grinders are not a specified rise."""
    if result.get("profit") is None:
        return None
    return result.get("exit") == "take_profit"


def breached_drawdown(mae, floor) -> bool:
    if mae != mae:
        return False
    return bool(mae < -abs(float(floor)))


def _net(entry, exit_px, buy_cost, sell_cost):
    return float(exit_px) * (1.0 - sell_cost) / (float(entry) * (1.0 + buy_cost)) - 1.0


def first_take_profit(
    entry,
    high,
    low,
    close,
    *,
    take_profit=PRIMARY_TP,
    buy_cost=P_TRACK_BUY_COST,
    sell_cost=P_TRACK_SELL_COST,
    b5=B5_RELATIVE,
):
    """Exit at the first post-T+1 touch of `take_profit`, else the D+20 close.

    Fill is assumed at the target price, not the day's high (conservative).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    if high.shape != low.shape or high.shape != close.shape or high.ndim != 1:
        raise ValueError("high/low/close must be 1d and aligned")
    n = len(high)
    if n < 2:
        raise ValueError("need at least two sessions: buy day plus one sellable day")
    entry = float(entry)
    if not np.isfinite(entry) or entry <= 0:
        raise ValueError("invalid entry")
    if take_profit <= 0:
        raise ValueError("take-profit must be positive")
    tp_px = entry * (1.0 + take_profit)
    b5_px = entry * (1.0 + buy_cost) * (1.0 - b5)
    mae = 0.0
    saw_b5 = False
    max_gain = window_max_gain(entry, high)
    for t in range(n):
        if np.isfinite(low[t]):
            mae = min(mae, float(low[t]) / entry - 1.0)
            if low[t] < b5_px:
                saw_b5 = True
        if t == 0:
            continue
        hit_tp = np.isfinite(high[t]) and high[t] >= tp_px
        hit_b5 = np.isfinite(low[t]) and low[t] < b5_px
        if hit_tp and hit_b5:
            opt = _net(entry, tp_px, buy_cost, sell_cost)
            pes = _net(entry, close[t] if np.isfinite(close[t]) else low[t], buy_cost, sell_cost)
            return dict(
                exit="ambiguous_same_day",
                exit_day=t + 1,
                net=None,
                profit=None,
                path_risk=True,
                mae=mae,
                max_gain=max_gain,
                take_profit=float(take_profit),
                bound_optimistic_net=opt,
                bound_pessimistic_net=pes,
                keep_ex_ante=True,
            )
        if hit_tp:
            net = _net(entry, tp_px, buy_cost, sell_cost)
            return dict(
                exit="take_profit",
                exit_day=t + 1,
                net=net,
                profit=net > 0,
                path_risk=saw_b5,
                mae=mae,
                max_gain=max_gain,
                take_profit=float(take_profit),
                keep_ex_ante=True,
            )
    last = close[-1]
    if not np.isfinite(last):
        return dict(exit="unresolved", exit_day=None, net=None, profit=None,
                    path_risk=saw_b5, mae=mae, max_gain=max_gain,
                    take_profit=float(take_profit), keep_ex_ante=True)
    net = _net(entry, last, buy_cost, sell_cost)
    return dict(
        exit="horizon",
        exit_day=n,
        net=net,
        profit=net > 0,
        path_risk=saw_b5,
        mae=mae,
        max_gain=max_gain,
        take_profit=float(take_profit),
        keep_ex_ante=True,
    )


def paired_class(result) -> str | None:
    """Four-class outcome under a frozen (take-profit, drawdown) pair.

    A: made money (TP or horizon) without the paired drawdown
    B: made money but saw the paired drawdown before exit
    C: did not make money, no paired drawdown
    D: did not make money and saw the paired drawdown
    Ambiguous/unresolved stay unlabeled.
    """
    if result.get("profit") is None:
        return None
    profit, risk = bool(result["profit"]), bool(result["path_risk"])
    if profit and not risk:
        return "A"
    if profit and risk:
        return "B"
    if (not profit) and (not risk):
        return "C"
    return "D"


def pool_take_profit(results) -> dict:
    rows = list(results)
    n = len(rows)
    amb = sum(r["exit"] == "ambiguous_same_day" for r in rows)
    resolved = [r for r in rows if r["profit"] is not None]
    if not resolved:
        return dict(n=n, n_resolved=0, n_ambiguous=amb, profit_rate=None, path_risk_rate=None,
                    take_profit_exit_rate=None, mean_net=None)
    return dict(
        n=n,
        n_resolved=len(resolved),
        n_ambiguous=amb,
        profit_rate=float(sum(r["profit"] for r in resolved) / len(resolved)),
        path_risk_rate=float(sum(r["path_risk"] for r in resolved) / len(resolved)),
        take_profit_exit_rate=float(sum(r["exit"] == "take_profit" for r in resolved) / len(resolved)),
        horizon_exit_rate=float(sum(r["exit"] == "horizon" for r in resolved) / len(resolved)),
        mean_net=float(np.mean([r["net"] for r in resolved])),
        mean_exit_day=float(np.mean([r["exit_day"] for r in resolved if r["exit_day"]])),
    )
