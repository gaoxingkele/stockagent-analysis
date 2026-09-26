"""Single-lot exit accounting for H02; consumes fills, never invents them.

No order transmission or tradability inference. Frozen-horizon P labels stay
separate from this extended T-track position lifecycle. Entry cost includes
entry fees; exit fees are explicit currency amounts, not percentages.
"""
from dataclasses import dataclass, replace
import math

from .execution import t_plus_one_can_sell


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError("invalid " + name)


@dataclass(frozen=True)
class BasisTransfer:
    record_quantity: int
    amount: float


@dataclass(frozen=True)
class ExitLot:
    position_id: str
    ts_code: str
    buy_date: str
    due_date: str
    initial_quantity: int
    remaining_quantity: int
    entry_cost: float
    net_exit_cash: float = 0.0
    exit_fees: float = 0.0
    realized_cost: float = 0.0
    realized_pnl: float = 0.0
    last_attempt_date: str | None = None
    attempt_ids: tuple[str, ...] = ()
    status: str = "OPEN"
    exit_policy_id: str | None = None
    exit_policy_frozen_at: str | None = None
    exit_calendar: tuple[str, ...] | None = None
    basis_transfers: tuple[BasisTransfer, ...] = ()
    explicit_sellable_from: str | None = None


def open_lot(position_id, ts_code, buy_date, due_date, quantity, entry_cost, calendar):
    """Initialize an already acquired lot, not a buy-fill authorization."""
    validate_calendar(calendar)
    if not position_id or not ts_code or buy_date not in calendar or due_date not in calendar:
        raise ValueError("missing position/date identity")
    if calendar.index(due_date) < calendar.index(buy_date):
        raise ValueError("exit due before acquisition")
    if type(quantity) is not int or quantity <= 0:
        raise ValueError("invalid quantity")
    positive(entry_cost, "entry cost")
    return ExitLot(position_id, ts_code, buy_date, due_date, quantity, quantity, float(entry_cost))


def validate_calendar(calendar):
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("unique ordered calendar required")


def allocated_exit_cost(lot, remaining):
    """Retain pre-record sold costs while transferring eligible units' basis."""
    if remaining == 0: return lot.entry_cost
    original=lot.entry_cost+sum(t.amount for t in lot.basis_transfers)
    return (original*(lot.initial_quantity-remaining)/lot.initial_quantity
            -sum(t.amount*max(0,t.record_quantity-remaining)/t.record_quantity
                 for t in lot.basis_transfers))


def apply_exit(lot, *, attempt_id, position_id, ts_code, trade_date, calendar,
               filled_quantity, price=None, fee=0.0, reason, evidence_id):
    """Apply one externally supplied fill/no-fill at the frozen exit or later.

    Call after the scheduled close attempt on due_date, or the separately
    frozen retry policy thereafter. No same-day T+1 sale, forced liquidation,
    released capital on rejection, or hindsight price substitution. Evidence
    IDs are traceability references, not assertions of independent validation.
    """
    validate_calendar(calendar)
    if (lot.position_id, lot.ts_code) != (position_id, ts_code):
        raise ValueError("position identity mismatch")
    if not attempt_id or attempt_id in lot.attempt_ids:
        raise ValueError("missing/duplicate attempt")
    if not reason or not evidence_id:
        raise ValueError("explicit reason/evidence required")
    if trade_date not in calendar or lot.buy_date not in calendar or lot.due_date not in calendar:
        raise ValueError("date outside calendar")
    if trade_date < lot.due_date or (lot.last_attempt_date and trade_date < lot.last_attempt_date):
        raise ValueError("exit before due or out-of-order attempt")
    if lot.remaining_quantity <= 0:
        raise ValueError("position already closed")
    if type(filled_quantity) is not int or not 0 <= filled_quantity <= lot.remaining_quantity:
        raise ValueError("invalid fill quantity")
    positive(fee, "exit fee", zero=True)
    if filled_quantity:
        if lot.explicit_sellable_from is not None:
            if trade_date < lot.explicit_sellable_from:
                raise ValueError('credited shares not yet sellable')
        elif not t_plus_one_can_sell(lot.buy_date, trade_date, calendar):
            raise ValueError("T+1 sale forbidden")
        positive(price, "fill price")
        gross = filled_quantity * price
        if fee > gross:
            raise ValueError("exit fees exceed proceeds; external cash funding required")
    else:
        if price is not None or fee != 0:
            raise ValueError("no-fill cannot create price/cash/fees")
        gross = 0.0
    remaining = lot.remaining_quantity - filled_quantity
    sold_total = lot.initial_quantity - remaining
    allocated = allocated_exit_cost(lot, remaining)
    cash = lot.net_exit_cash + gross - fee
    result = replace(lot, remaining_quantity=remaining, net_exit_cash=cash,
                     exit_fees=lot.exit_fees + fee, realized_cost=allocated,
                     realized_pnl=cash - allocated, last_attempt_date=trade_date,
                     attempt_ids=lot.attempt_ids + (attempt_id,),
                     status="CLOSED" if remaining == 0 else "EXIT_PENDING")
    record = dict(attempt_id=attempt_id, position_id=position_id, ts_code=ts_code,
                  trade_date=trade_date, due_date=lot.due_date,
                  filled_quantity=filled_quantity, fill_price=price, fee_currency=fee,
                  cash_delta=gross-fee, remaining_quantity=remaining,
                  remaining_cost_basis=lot.entry_cost-allocated,
                  cumulative_realized_pnl=result.realized_pnl, reason=reason,
                  evidence_id=evidence_id, exit_pending=remaining > 0,
                  risk_exposure_continues=remaining > 0,
                  remaining_market_value=None if remaining else 0.0,
                  original_horizon_label_unchanged=True,
                  fill_independently_verified=False, formal_training_authorized=False)
    return result, record
