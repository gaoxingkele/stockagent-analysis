"""Explicit same-time long-position marks; missing values remain unknown.

This is reference mark-to-market, not executable liquidation value or an
independently audited NAV. Prices must use each position's trading unit and
the portfolio currency; corporate-action reconciliation is upstream work.
"""
from dataclasses import dataclass
from collections import Counter
from decimal import Decimal
import re

from .exit_book import positive
from .label_availability import _instant
from .portfolio_book import accounting_snapshot, share_contract_residual


@dataclass(frozen=True)
class PriceMark:
    ts_code: str
    price_at: str
    available_at: str
    status: str
    price: float | None
    source_id: str
    source_sha256: str
    basis: str = "portfolio_currency_per_position_unit"


@dataclass(frozen=True)
class ShareCreditMark:
    distribution_id: str
    price_at: str
    available_at: str
    status: str
    price: float | None
    source_id: str
    source_sha256: str
    basis: str = "portfolio_currency_per_credited_share"


def value_portfolio(book, marks, *, valuation_at, knowledge_at, share_marks=()):
    valuation, knowledge = _instant(valuation_at), _instant(knowledge_at)
    if valuation > knowledge:
        raise ValueError("valuation later than knowledge time")
    # A later mutable book cannot be relabeled as an earlier position snapshot.
    if book.last_processed_at and _instant(book.last_processed_at) > valuation:
        raise ValueError("book snapshot contains later processing")
    accounting = accounting_snapshot(book)
    held = [p for p in book.positions if p.remaining_quantity > 0]
    codes = {p.ts_code for p in held}
    indexed = {}
    for mark in marks:
        if mark.ts_code in indexed or mark.ts_code not in codes:
            raise ValueError("duplicate or non-held mark identity")
        if (not mark.source_id or not re.fullmatch(r"[0-9a-f]{64}", mark.source_sha256 or "")
                or mark.basis != "portfolio_currency_per_position_unit"):
            raise ValueError("missing provenance or incompatible price units")
        price_at, available = _instant(mark.price_at), _instant(mark.available_at)
        if price_at > valuation or available > knowledge or available < price_at:
            raise ValueError("future price or invalid availability")
        if mark.status not in {"observed", "suspended", "missing"}:
            raise ValueError("unknown mark status")
        if mark.status == "observed":
            positive(mark.price, "mark price")
        elif mark.price is not None:
            raise ValueError("unobserved mark cannot supply price")
        indexed[mark.ts_code] = mark
    rows = []
    pending_exits = [o for o in book.exits if _instant(o.slot_at) <= valuation]
    uncertain_positions = {o.lot.position_id for o in pending_exits}
    for position in held:
        mark = indexed.get(position.ts_code)
        status = "missing_mark" if mark is None else "stale_mark" if _instant(mark.price_at) != valuation else mark.status
        if position.position_id in uncertain_positions:
            status = "exit_outcome_pending"
        value = None
        if status == "observed":
            value = position.remaining_quantity * mark.price
            positive(value, "position market value")
        rows.append(dict(position_id=position.position_id, ts_code=position.ts_code,
            remaining_quantity=position.remaining_quantity,
            remaining_cost_basis=position.entry_cost-position.realized_cost,
            exit_pending=position.status == "EXIT_PENDING", valuation_status=status,
            market_value=value, source_id=mark.source_id if mark else None,
            source_sha256=mark.source_sha256 if mark else None,
            price_at=mark.price_at if mark else None, available_at=mark.available_at if mark else None))
    unresolved = sum(r["market_value"] is None for r in rows)
    # After an order's slot, a missing aggregate receipt may conceal a fill.
    # Reserved money remains an accounting asset but its market value is not
    # known cash until the result is reconciled.
    pending_buys = [o for o in book.buys if _instant(o.slot_at) <= valuation]
    uncertain_reserve = sum(o.reserve for o in pending_buys)
    known = accounting["cash"] + accounting["reserved_buy_cash"] + accounting["gross_dividend_receivable"] - uncertain_reserve + sum(r["market_value"] or 0 for r in rows)
    positive(known, "known component value", zero=True)
    valuation_date = valuation.tz_convert("Asia/Shanghai").strftime("%Y%m%d")
    unprocessed_cash = [c.distribution.event_id for c in book.cash_claims
                        if not c.recognized and c.distribution.ex_date <= valuation_date]
    # Credit assets need their own explicitly supplied marks. Do not assume
    # the original stock's quote applies to an unlisted/restricted credit.
    credits = {c.distribution.event_id: c for c in book.share_claims if c.receipts and c.transferred_position_id is None}
    credit_marks = {}
    for mark in share_marks:
        if mark.distribution_id not in credits or mark.distribution_id in credit_marks:
            raise ValueError('duplicate or non-credited share mark')
        if (not mark.source_id or not re.fullmatch(r'[0-9a-f]{64}',mark.source_sha256 or '')
                or mark.basis != 'portfolio_currency_per_credited_share'):
            raise ValueError('share mark provenance/unit mismatch')
        price_at,available=_instant(mark.price_at),_instant(mark.available_at)
        if price_at>valuation or available>knowledge or available<price_at:
            raise ValueError('share mark chronology invalid')
        if mark.status not in {'observed','suspended','missing'}:
            raise ValueError('unknown share mark status')
        if mark.status=='observed': positive(mark.price,'share credit mark')
        elif mark.price is not None: raise ValueError('unobserved share mark has price')
        credit_marks[mark.distribution_id]=mark
    credit_rows=[];pending_shares=[]
    for claim in book.share_claims:
        if claim.transferred_position_id is not None:
            continue  # Valued once as an ordinary position, never twice as a credit.
        quantity=sum(r.quantity for r in claim.receipts)
        residual=share_contract_residual(claim)
        unresolved_right=residual>0 and claim.distribution.ex_date<=valuation_date
        mark=credit_marks.get(claim.distribution.event_id)
        status=('no_credit' if not quantity else 'missing_mark' if mark is None
                else 'stale_mark' if _instant(mark.price_at)!=valuation else mark.status)
        market_value=quantity*mark.price if status=='observed' else None
        if market_value is not None: positive(market_value,'credited share market value')
        if unresolved_right or (quantity and market_value is None):
            pending_shares.append(claim.distribution.event_id)
        credit_rows.append(dict(distribution_id=claim.distribution.event_id,ts_code=claim.ts_code,
            credited_quantity=quantity,contractual_residual=str(residual),
            valuation_status=status,market_value=market_value,
            source_id=mark.source_id if mark else None,source_sha256=mark.source_sha256 if mark else None,
            price_at=mark.price_at if mark else None,available_at=mark.available_at if mark else None,
            cost_basis_allocated=claim.allocated_basis is not None,
            allocated_cost_basis=claim.allocated_basis,tradable_quantity=None))
    credit_value=sum(r['market_value'] or 0 for r in credit_rows)
    known+=credit_value
    positive(known,'known component value',zero=True)
    complete = not unresolved and not pending_buys and not unprocessed_cash and not pending_shares
    nav = known if complete else None
    return dict(valuation_at=valuation.isoformat(), knowledge_at=knowledge.isoformat(),
        cash=accounting["cash"], reserved_buy_cash=accounting["reserved_buy_cash"],
        gross_dividend_receivable=accounting["gross_dividend_receivable"],
        final_personal_dividend_tax_verified=False,
        initial_cash=book.initial_cash, held_positions=len(held), unresolved_positions=unresolved,
        unresolved_buy_order_ids=[o.order_id for o in pending_buys],
        unresolved_buy_reserved_cash=uncertain_reserve,
        unresolved_exit_order_ids=[o.order_id for o in pending_exits],
        unprocessed_cash_distribution_ids=unprocessed_cash,
        unresolved_share_distribution_ids=pending_shares,
        share_credit_valuations=credit_rows,known_share_credit_value=credit_value,
        positions=rows, status_counts=dict(Counter(r["valuation_status"] for r in rows)),
        known_component_value=known, known_component_is_total_nav=complete,
        nav=nav, cumulative_reference_return=None if nav is None else nav/book.initial_cash-1,
        market_value_is_not_liquidation_proceeds=True,
        source_bytes_verified=False, corporate_action_coverage_verified=False,
        formal_training_authorized=False)
