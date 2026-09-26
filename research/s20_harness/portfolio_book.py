"""Cash-constrained reference portfolio; no real orders or inferred fills.

Buy requests reserve worst-case allowed cash. One aggregate result closes each
request; any unfilled remainder is cancelled, not silently carried forward.
Repeated recommendations remain in the journal but do not add to held names.
Corporate actions, valuation and independent execution proof are separate.
"""
from dataclasses import dataclass, replace
import math
from decimal import Decimal

import pandas as pd

from .exit_book import ExitLot, open_lot, positive, validate_calendar
from .exit_policy import ExitRequest, request_exit, settle_exit
from .label_availability import _instant
from .corporate_actions import Distribution


@dataclass(frozen=True)
class CashClaim:
    ts_code: str
    distribution: Distribution
    record_quantity: int
    gross_amount: float
    source_id: str
    recognized: bool = False
    paid: bool = False


@dataclass(frozen=True)
class ShareReceipt:
    quantity: int
    credited_at: str
    received_at: str
    evidence_id: str
    evidence_sha256: str


@dataclass(frozen=True)
class FractionalSettlement:
    quantity: str
    net_cash: float
    settled_at: str
    received_at: str
    evidence_id: str
    evidence_sha256: str


@dataclass(frozen=True)
class ShareClaim:
    ts_code: str
    distribution: Distribution
    record_quantity: int
    contractual_quantity: float
    source_id: str
    source_sha256: str
    terms_available_at: str
    receipts: tuple[ShareReceipt, ...] = ()
    fractional_settlement: FractionalSettlement | None = None
    record_lots: tuple[tuple[str, int], ...] = ()
    allocated_basis: float | None = None
    transferred_position_id: str | None = None


@dataclass(frozen=True)
class FollowOnEligibility:
    source_distribution_id: str
    target_distribution: Distribution
    position_id: str
    eligible: bool
    available_at: str
    evidence_id: str
    evidence_sha256: str


@dataclass(frozen=True)
class BuyRequest:
    order_id: str
    ts_code: str
    buy_date: str
    due_date: str
    quantity: int
    price_cap: float
    fee_cap: float
    reserve: float
    submitted_at: str
    slot_at: str


@dataclass(frozen=True)
class Portfolio:
    initial_cash: float
    cash: float
    calendar: tuple[str, ...]
    buys: tuple[BuyRequest, ...] = ()
    exits: tuple[ExitRequest, ...] = ()
    positions: tuple[ExitLot, ...] = ()
    event_ids: tuple[str, ...] = ()
    last_processed_at: str | None = None
    journal: tuple[dict, ...] = ()
    cash_claims: tuple[CashClaim, ...] = ()
    distribution_income: float = 0.0
    distribution_tax_paid: float = 0.0
    share_claims: tuple[ShareClaim, ...] = ()
    share_settlement_net_income: float = 0.0
    follow_on_eligibility: tuple[FollowOnEligibility, ...] = ()


def create(cash, calendar):
    positive(cash, "initial cash")
    validate_calendar(calendar)
    return Portfolio(float(cash), float(cash), tuple(calendar))


def clock(book, event_id, at):
    instant = _instant(at)
    if not event_id or event_id in book.event_ids:
        raise ValueError("missing/duplicate portfolio event")
    if book.last_processed_at and instant < _instant(book.last_processed_at):
        raise ValueError("out-of-order portfolio event")
    return instant.isoformat()


def append(book, event_id, at, event, **changes):
    positive(changes.get("cash", book.cash), "remaining cash", zero=True)
    return replace(book, **changes, event_ids=book.event_ids + (event_id,),
                   last_processed_at=at, journal=book.journal + ({**event,
                       "event_id": event_id, "processed_at": at,
                       "formal_training_authorized": False},))


def reserve_buy(book, *, order_id, ts_code, buy_date, due_date, quantity,
                price_cap, fee_cap, submitted_at):
    at = clock(book, order_id, submitted_at)
    if type(quantity) is not int or quantity <= 0 or not ts_code:
        raise ValueError("invalid buy identity/quantity")
    positive(price_cap, "price cap"); positive(fee_cap, "fee cap", zero=True)
    if buy_date not in book.calendar or due_date not in book.calendar or due_date < buy_date:
        raise ValueError("invalid buy/exit dates")
    slot = pd.Timestamp(buy_date + " 09:30", tz="Asia/Shanghai").tz_convert("UTC")
    if _instant(at) >= slot or _instant(at).tz_convert("Asia/Shanghai").strftime("%Y%m%d") != buy_date:
        raise ValueError("buy request must precede same-day reference open")
    reserve = quantity * price_cap + fee_cap
    positive(reserve, "cash reservation")
    held = (any(p.ts_code == ts_code and p.remaining_quantity for p in book.positions)
            or any(c.ts_code==ts_code and (share_contract_residual(c)>0
                or (c.receipts and c.transferred_position_id is None)) for c in book.share_claims))
    pending = any(o.ts_code == ts_code for o in book.buys)
    reason = "already_held" if held else "buy_pending" if pending else "insufficient_cash" if reserve > book.cash else "reserved"
    event = dict(kind="buy_request", order_id=order_id, ts_code=ts_code, reason=reason,
                 recommendation_kept=True, buy_date=buy_date, due_date=due_date,
                 quantity=quantity, price_cap=price_cap, fee_cap=fee_cap,
                 slot_at=slot.isoformat(), cash_reserved=reserve if reason == "reserved" else 0.0)
    if reason != "reserved":
        return append(book, order_id, at, event)
    req = BuyRequest(order_id, ts_code, buy_date, due_date, quantity, price_cap, fee_cap, reserve, at, slot.isoformat())
    return append(book, order_id, at, event, cash=book.cash-reserve, buys=book.buys + (req,))


def settle_buy(book, *, event_id, order_id, outcome_at, received_at, processing_at,
               filled_quantity, price=None, fee=0.0, reason, evidence_id):
    at = clock(book, event_id, processing_at)
    found = [o for o in book.buys if o.order_id == order_id]
    if len(found) != 1:
        raise ValueError("buy order missing/already settled")
    req = found[0]
    outcome, received = _instant(outcome_at), _instant(received_at)
    if outcome != _instant(req.slot_at) or not outcome <= received <= _instant(at):
        raise ValueError("buy outcome/receipt timing mismatch")
    if not reason or not evidence_id:
        raise ValueError("buy result needs reason/evidence")
    if type(filled_quantity) is not int or not 0 <= filled_quantity <= req.quantity:
        raise ValueError("invalid buy fill quantity")
    positive(fee, "buy fee", zero=True)
    if fee > req.fee_cap:
        raise ValueError("buy fee exceeds reserved cap")
    positions = book.positions
    if filled_quantity:
        positive(price, "buy price")
        if price > req.price_cap:
            raise ValueError("buy price exceeds reserved cap")
        spent = price*filled_quantity + fee
        positive(spent, "entry spend")
        if spent > req.reserve:
            raise ValueError("buy cost exceeds reservation")
        positions += (open_lot(order_id, req.ts_code, req.buy_date, req.due_date,
                               filled_quantity, spent, list(book.calendar)),)
    else:
        if price is not None or fee:
            raise ValueError("no-fill cannot charge or invent price")
        spent = 0.0
    return append(book, event_id, at, dict(kind="buy_result", order_id=order_id,
        ts_code=req.ts_code, price=price, fee_currency=fee,
        filled_quantity=filled_quantity, unfilled_cancelled=req.quantity-filled_quantity,
        cash_spent=spent, cash_released=req.reserve-spent, evidence_id=evidence_id,
        reason=reason, outcome_at=outcome.isoformat(), received_at=received.isoformat(),
        fill_independently_verified=False), cash=book.cash+req.reserve-spent,
        buys=tuple(o for o in book.buys if o.order_id != order_id), positions=positions)


def reserve_exit(book, *, order_id, position_id, policy, submitted_at):
    at = clock(book, order_id, submitted_at)
    found = [p for p in book.positions if p.position_id == position_id]
    if len(found) != 1 or any(o.lot.position_id == position_id for o in book.exits):
        raise ValueError("missing position or exit already pending")
    req = request_exit(found[0], policy, order_id=order_id, submitted_at=at, calendar=list(book.calendar))
    return append(book, order_id, at, dict(kind="exit_request", position_id=position_id,
        order_id=order_id, phase=req.phase, slot_at=req.slot_at, cash_released=0.0,
        policy_id=policy.policy_id, policy_frozen_at=policy.frozen_at,
        remaining_quantity=req.lot.remaining_quantity), exits=book.exits+(req,))


def settle_portfolio_exit(book, *, event_id, order_id, processing_at, **result):
    at = clock(book, event_id, processing_at)
    found = [o for o in book.exits if o.order_id == order_id]
    if len(found) != 1:
        raise ValueError("exit order missing/already settled")
    req = found[0]
    lot = next(p for p in book.positions if p.position_id == req.lot.position_id)
    new, event = settle_exit(lot, req, processing_at=at, **result)
    return append(book, event_id, at, dict(kind="exit_result", **event),
        cash=book.cash+event["cash_delta"], exits=tuple(o for o in book.exits if o.order_id != order_id),
        positions=tuple(new if p.position_id == new.position_id else p for p in book.positions))


def share_contract_residual(claim):
    return (Decimal(claim.record_quantity)*Decimal(str(claim.distribution.bonus_per_share))
            -sum(r.quantity for r in claim.receipts)
            -(Decimal(claim.fractional_settlement.quantity) if claim.fractional_settlement else 0))


def record_date_lots(book, ts_code, instant, distribution=None):
    """Resolve event-specific eligibility; sellability alone is insufficient."""
    day=instant.tz_convert('Asia/Shanghai').strftime('%Y%m%d')
    excluded=set()
    for c in book.share_claims:
        if c.ts_code!=ts_code or c.distribution.ex_date>day: continue
        if share_contract_residual(c)>0 or (c.receipts and c.transferred_position_id is None):
            raise ValueError('unresolved share eligibility at record date')
        if c.transferred_position_id is None: continue
        lots=[p for p in book.positions if p.position_id==c.transferred_position_id]
        if len(lots)!=1: raise ValueError('transferred share position missing')
        if lots[0].remaining_quantity==0: continue
        decisions=[e for e in book.follow_on_eligibility
            if e.source_distribution_id==c.distribution.event_id and e.target_distribution==distribution
            and e.position_id==c.transferred_position_id and _instant(e.available_at)<=instant]
        if len(decisions)!=1: raise ValueError('unresolved share eligibility at record date')
        if not decisions[0].eligible: excluded.add(c.transferred_position_id)
    if any(o.ts_code==ts_code and _instant(o.slot_at)<=instant for o in book.buys) or any(
            o.lot.ts_code==ts_code and _instant(o.slot_at)<=instant for o in book.exits):
        raise ValueError('record holdings contain unsettled execution result')
    return [p for p in book.positions if p.ts_code==ts_code and p.remaining_quantity>0 and p.position_id not in excluded]


def record_date_quantity(book, ts_code, instant, distribution=None):
    return sum(p.remaining_quantity for p in record_date_lots(book,ts_code,instant,distribution))


def accounting_snapshot(book):
    """Cost accounting identity; never reports cost basis as market NAV."""
    reserved = sum(o.reserve for o in book.buys)
    credit_basis=sum(c.allocated_basis or 0 for c in book.share_claims if c.transferred_position_id is None)
    cost = sum(p.entry_cost-p.realized_cost for p in book.positions)+credit_basis
    pnl = sum(p.realized_pnl for p in book.positions)
    receivable = sum(c.gross_amount for c in book.cash_claims if c.recognized and not c.paid)
    lhs, rhs = book.cash+reserved+cost+receivable, book.initial_cash+pnl+book.distribution_income-book.distribution_tax_paid+book.share_settlement_net_income
    if not math.isclose(lhs, rhs, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("cash/position accounting conservation failed")
    held = sum(p.remaining_quantity > 0 for p in book.positions)
    share_assets=any((c.receipts and c.transferred_position_id is None) or share_contract_residual(c)>0 for c in book.share_claims)
    return dict(cash=book.cash, reserved_buy_cash=reserved, remaining_cost_basis=cost,
                credited_share_cost_basis=credit_basis,
                gross_dividend_receivable=receivable, distribution_income=book.distribution_income,
                distribution_tax_paid=book.distribution_tax_paid,
                share_settlement_net_income=book.share_settlement_net_income,
                realized_pnl=pnl, held_positions=held, pending_exits=sum(p.status == "EXIT_PENDING" for p in book.positions),
                conservation_error=lhs-rhs, nav=None if held or share_assets else book.cash+reserved+receivable,
                nav_basis="unknown without independently supplied valuation" if held or share_assets else "cash plus gross dividend receivables; not final-tax NAV",
                formal_training_authorized=False)
