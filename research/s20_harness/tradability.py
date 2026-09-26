"""Explicit limit states and availability-gated reference execution decisions."""
from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from .execution import FillDecision, decide_open_buy
from .label_availability import _instant


@dataclass(frozen=True)
class LimitState:
    ts_code: str
    trade_date: str
    status: str
    up_limit: float | None = None
    down_limit: float | None = None
    available_at: str | None = None


def resolve_limits(frame, ts_code, trade_date, *, available_at=None):
    code, date = str(ts_code), str(trade_date)
    if not {"ts_code", "trade_date", "up_limit", "down_limit"}.issubset(frame.columns):
        raise ValueError("missing limit fields")
    rows = frame.loc[frame.ts_code.eq(code) & frame.trade_date.astype(str).eq(date)]
    common = dict(ts_code=code, trade_date=date, available_at=available_at)
    if rows.empty:
        return LimitState(**common, status="missing")
    if len(rows) != 1:
        return LimitState(**common, status="duplicate_or_conflicting")
    try:
        up, down = float(rows.up_limit.iloc[0]), float(rows.down_limit.iloc[0])
    except (TypeError, ValueError):
        return LimitState(**common, status="invalid")
    if up == 99999.99 and down == 0:
        return LimitState(**common, status="sentinel_requires_rule_evidence")
    if not all(math.isfinite(x) and x > 0 for x in (up, down)) or up < down:
        return LimitState(**common, status="invalid")
    return LimitState(**common, status="bounded", up_limit=up, down_limit=down)


def reference_open_buy(open_price, state, *, ts_code, trade_date, decision_at, suspended):
    """Requires date/code identity, known halt state and strictly earlier inputs.

    A successful result remains a reference-price fill, not queue/capacity proof.
    Special no-limit events need a separate rule/auction adapter; None bounds do
    not authorize them. Retrospective receipt timestamps cannot be backdated.
    """
    instant = _instant(decision_at)
    if str(ts_code) != state.ts_code or str(trade_date) != state.trade_date:
        return FillDecision(False, None, "limit_identity_mismatch")
    if instant.tz_convert("Asia/Shanghai").strftime("%Y%m%d") != state.trade_date:
        return FillDecision(False, None, "execution_date_mismatch")
    if suspended is not True and suspended is not False:
        return FillDecision(False, None, "halt_status_unknown")
    if suspended:
        return FillDecision(False, None, "suspended")
    if state.status != "bounded":
        return FillDecision(False, None, "limit_" + state.status)
    if state.up_limit is None or state.down_limit is None:
        return FillDecision(False, None, "invalid_limit_data")
    if state.available_at is None:
        return FillDecision(False, None, "limit_availability_unknown")
    if _instant(state.available_at) >= instant:
        return FillDecision(False, None, "limit_not_available_before_execution")
    return decide_open_buy(open_price, limit_up=state.up_limit, limit_down=state.down_limit)
