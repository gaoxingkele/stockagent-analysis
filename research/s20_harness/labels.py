"""P-track safe-profit labels and separate O-track opportunity labels."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.s20_harness.execution import decide_open_buy, market_horizon

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from stockagent_analysis.s20_v3 import path_labels  # noqa: E402

P_TRACK_BUY_COST = 0.001
P_TRACK_SELL_COST = 0.0015
B5_RELATIVE = 0.05
UNFILLED_CLASS = None


def _as_date_str(value) -> str:
    return str(int(value)) if isinstance(value, (int, np.integer)) else str(value)


def _stock_frame(daily: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    frame = daily.loc[daily["ts_code"].astype(str) == str(ts_code)].copy()
    frame["trade_date"] = frame["trade_date"].map(_as_date_str)
    if frame["trade_date"].duplicated().any():
        raise ValueError("duplicate stock/date quotes require identity reconciliation")
    return frame.sort_values("trade_date")


def _align_window(stock: pd.DataFrame, window: list[str]) -> pd.DataFrame:
    indexed = stock.set_index("trade_date")
    aligned = pd.DataFrame(index=list(window))
    for col in ("open", "high", "low", "close"):
        if col in indexed.columns:
            aligned[col] = pd.to_numeric(indexed[col], errors="coerce")
        else:
            aligned[col] = np.nan
    return aligned


def _invalid_quote_dates(aligned: pd.DataFrame) -> list[str]:
    values = aligned.to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    valid &= (aligned.high >= aligned[["open", "close", "low"]].max(axis=1)).to_numpy()
    valid &= (aligned.low <= aligned[["open", "close", "high"]].min(axis=1)).to_numpy()
    return aligned.index[~valid].tolist()


def _path_days(entry: float, threshold_high: float, threshold_low: float, aligned: pd.DataFrame):
    first_up = first_down = time_to_b5 = None
    for i, date in enumerate(aligned.index, start=1):
        high = aligned.at[date, "high"]
        low = aligned.at[date, "low"]
        if first_up is None and np.isfinite(high) and high > entry:
            first_up = i
        if first_down is None and np.isfinite(low) and low < entry:
            first_down = i
        if time_to_b5 is None and np.isfinite(low) and low < threshold_low:
            time_to_b5 = i
    return first_up, first_down, time_to_b5


def label_p_track(
    daily: pd.DataFrame,
    calendar: list[str],
    signal_date: str,
    ts_code: str,
    *,
    fillable: bool = True,
    buy_cost: float = P_TRACK_BUY_COST,
    sell_cost: float = P_TRACK_SELL_COST,
    suspended: bool = False,
    limit_up=None,
    limit_down=None,
    pct_chg=None,
    horizon: int = 20,
    distributions: list | None = None,
    limit_state=None,
    execution_at=None,
    execution_halt_status=None,
    execution_code: str | None = None,
) -> dict:
    """Market-calendar D+1 open to D+20 close four-class safe-profit label.

    Unfilled D+1 keeps the recommendation and does not receive A/B/C/D.
    """
    horizon_info = market_horizon(calendar, signal_date, horizon=horizon)
    if not np.isfinite(buy_cost) or not np.isfinite(sell_cost) or buy_cost < 0 or not 0 <= sell_cost < 1:
        raise ValueError("invalid fixed label costs")
    stock = _stock_frame(daily, ts_code)
    aligned = _align_window(stock, horizon_info["window"])
    entry_date = horizon_info["entry_date"]
    raw_open = aligned.at[entry_date, "open"] if entry_date in aligned.index else np.nan
    fill = decide_open_buy(
        raw_open,
        suspended=suspended or not fillable,
        limit_up=limit_up,
        limit_down=limit_down,
        pct_chg=pct_chg,
    )
    if limit_state is not None:
        from .tradability import reference_open_buy
        if execution_at is None:
            raise ValueError("execution timestamp required for strict limit-state labels")
        fill = reference_open_buy(raw_open, limit_state, ts_code=execution_code or ts_code,
                                  trade_date=entry_date, decision_at=execution_at,
                                  suspended=True if suspended or not fillable else execution_halt_status)
    base = {
        "track": "safe_profit_20d_research",
        "ts_code": str(ts_code),
        "signal_date": str(signal_date),
        "entry_date": entry_date,
        "horizon_end": horizon_info["horizon_end"],
        "horizon_sessions": horizon,
        "fill_status": "filled" if fill.filled else "unfilled",
        "reject_postpone_entry": True,
        "p_class": UNFILLED_CLASS,
        "up_event": None,
        "b5": None,
        "first_up_day": None,
        "first_down_day": None,
        "time_to_b5": None,
        "mae": None,
        "max_drawdown": None,
        "terminal_net": None,
        "recommendation_kept": True,
        "label_realized": False,
        "economic_action_adjustment_verified": False,
        "formal_training_eligible": False,
        "price_basis": "raw_quote_diagnostic_not_economic_ledger",
        "execution_basis": "availability_gated_reference" if limit_state is not None else "raw_reference_diagnostic",
    }
    if not fill.filled:
        base["fill_reason"] = fill.reason
        return base

    entry = float(fill.price)
    entry_cost = entry * (1.0 + buy_cost)
    invalid_dates = _invalid_quote_dates(aligned)
    if invalid_dates:
        # Absence of a quote does not prove suspension, zero return, no barrier
        # breach, or an executable exit. Keep the recommendation unresolved.
        base.update(entry_open=entry, entry_cost=entry_cost, fill_reason=fill.reason,
                    label_status="unknown_path", invalid_or_missing_dates=invalid_dates,
                    exit_pending=horizon_info["horizon_end"] in invalid_dates,
                    reason="missing_or_invalid_quotes_no_imputation")
        return base
    eco = aligned
    accounting = None
    if distributions is not None:
        from .economic_window import value_window
        eco, accounting = value_window(aligned, distributions)
        base.update(price_basis="economic_value_supplied_events_coverage_unverified",
                    economic_ledger=accounting["ledger"],
                    settlement_pending=accounting["settlement_pending"],
                    corporate_action_coverage_proven=False)
    min_low = float(np.nanmin(eco["low"].to_numpy(dtype=float)))
    mae = min_low / entry - 1.0
    b5 = bool(min_low < entry_cost * (1.0 - B5_RELATIVE))
    peak = eco["high"].cummax()
    drawdown = eco["low"] / peak - 1.0
    max_drawdown = float(drawdown.min())
    last_close = eco["close"].iloc[-1]
    unresolved_exit = bool(pd.isna(aligned["close"].iloc[-1]))
    # Cash and receivables are not stock sales. Charge the reference sell cost
    # only on equity value; pending bonus settlement is disclosed separately.
    cash_value = (accounting["cash"] + accounting["receivable_cash"]) if accounting else 0.
    proceeds = (float(last_close) - cash_value) * (1.0 - sell_cost) + cash_value
    terminal_net = proceeds / entry_cost - 1.0
    up_event = bool(terminal_net > 0)
    if up_event and not b5:
        p_class = "A"
    elif up_event and b5:
        p_class = "B"
    elif (not up_event) and (not b5):
        p_class = "C"
    else:
        p_class = "D"
    b5_price = entry_cost * (1.0 - B5_RELATIVE)
    first_up, first_down, time_to_b5 = _path_days(entry, entry, b5_price, eco)
    base.update({
        "fill_reason": fill.reason,
        "entry_open": entry,
        "entry_cost": entry_cost,
        "p_class": p_class,
        "up_event": up_event,
        "b5": b5,
        "first_up_day": first_up,
        "first_down_day": first_down,
        "time_to_b5": time_to_b5,
        "mae": mae,
        "max_drawdown": max_drawdown,
        "terminal_net": terminal_net,
        "exit_pending": unresolved_exit,
        "label_realized": not unresolved_exit,
        "label_status": "complete_economic_diagnostic" if accounting else "complete_raw_quote_diagnostic",
    })
    return base


def label_o_track(
    daily: pd.DataFrame,
    calendar: list[str] | None,
    signal_date: str,
    ts_code: str,
    *,
    horizon: int = 20,
    mode: str = "stock_session_v3_compat",
    distributions: list | None = None,
) -> dict:
    """Opportunity track. ``mode`` selects v3 stock-session windows or calendar PIT.

    This does not overwrite P-track classes.
    """
    stock = _stock_frame(daily, ts_code)
    signal_date = str(signal_date)
    accounting = None
    if mode == "stock_session_v3_compat":
        if distributions is not None:
            raise ValueError("economic actions require calendar_pit_v4; compatibility labels are frozen")
        dates = stock["trade_date"].tolist()
        if signal_date not in dates:
            raise ValueError("signal_date missing from stock sessions")
        i = dates.index(signal_date)
        if i + horizon >= len(dates):
            raise ValueError("not enough stock sessions for v3-compat window")
        entry_date = dates[i + 1]
        horizon_end = dates[i + horizon]
        window_dates = dates[i + 1 : i + 1 + horizon]
        entry = float(stock.loc[stock["trade_date"] == entry_date, "open"].iloc[0])
        highs = stock.loc[stock["trade_date"].isin(window_dates), "high"].to_numpy(dtype=float)
        lows = stock.loc[stock["trade_date"].isin(window_dates), "low"].to_numpy(dtype=float)
    elif mode == "calendar_pit_v4":
        if calendar is None:
            raise ValueError("calendar required for calendar_pit_v4")
        info = market_horizon(calendar, signal_date, horizon=horizon)
        entry_date, horizon_end, window_dates = info["entry_date"], info["horizon_end"], info["window"]
        aligned = _align_window(stock, window_dates)
        raw_open = aligned.at[entry_date, "open"]
        if not np.isfinite(raw_open):
            return {
                "track": "opportunity_calendar_pit_v4",
                "mode": mode,
                "ts_code": str(ts_code),
                "signal_date": signal_date,
                "entry_date": entry_date,
                "horizon_end": horizon_end,
                "o_class": None,
                "reason": "unfilled_or_missing_entry",
                "recommendation_kept": True,
                "formal_training_eligible": False,
            }
        entry = float(raw_open)
        invalid_dates = _invalid_quote_dates(aligned)
        if invalid_dates:
            return {"track": "opportunity_calendar_pit_v4", "mode": mode,
                    "ts_code": str(ts_code), "signal_date": signal_date,
                    "entry_date": entry_date, "horizon_end": horizon_end,
                    "o_class": None, "recommendation_kept": True,
                    "invalid_or_missing_dates": invalid_dates,
                    "reason": "missing_or_invalid_quotes_no_imputation",
                    "formal_training_eligible": False}
        highs = aligned["high"].to_numpy(dtype=float)
        lows = aligned["low"].to_numpy(dtype=float)
        if distributions is not None:
            from .economic_window import value_window
            economic, accounting = value_window(aligned, distributions)
            highs = economic.high.to_numpy(dtype=float)
            lows = economic.low.to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown O-track mode {mode}")
    path = path_labels([entry], highs.reshape(1, -1), lows.reshape(1, -1)).iloc[0]
    return {
        "track": "opportunity_v3_compat" if mode == "stock_session_v3_compat" else "opportunity_calendar_pit_v4",
        "mode": mode,
        "ts_code": str(ts_code),
        "signal_date": signal_date,
        "entry_date": entry_date,
        "horizon_end": horizon_end,
        "o_class": int(path.s20_class),
        "reason": path.reason,
        "hit20_day": int(path.hit20_day),
        "stop10_day": int(path.stop10_day),
        "max_gain20": float(path.max_gain20),
        "window_mae20": float(path.window_mae20),
        "immediate": int(path.immediate),
        "opportunity": int(path.opportunity),
        "down_risk": int(path.down_risk),
        "formal_training_eligible": False,
        "economic_action_adjustment_verified": False,
        "price_basis": "economic_value_supplied_events_coverage_unverified" if accounting else "raw_quotes",
        "return_units": "percentage_points_v3_compat",
        "economic_ledger": accounting["ledger"] if accounting else None,
        "settlement_pending": accounting["settlement_pending"] if accounting else None,
        "recommendation_kept": True,
    }
