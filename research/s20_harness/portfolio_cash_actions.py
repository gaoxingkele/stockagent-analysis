"""Record-date cash claims, ex-date accrual and explicit payment receipts.

Gross reference accounting only. No automatic payment from an ex/pay date,
no investor-specific final tax inference, no bonus/rights/merger handling.
"""
from dataclasses import replace

import pandas as pd

from .label_availability import _instant
from .portfolio_book import CashClaim, append, clock, record_date_quantity
from .exit_book import positive


def record_cash_claim(book, *, event_id, ts_code, distribution, processing_at,
                      terms_available_at, source_id):
    at = clock(book, event_id, processing_at)
    instant = _instant(at)
    day = instant.tz_convert("Asia/Shanghai").strftime("%Y%m%d")
    close = pd.Timestamp(distribution.record_date + " 15:00", tz="Asia/Shanghai")
    if day != distribution.record_date or instant < close or not source_id:
        raise ValueError("record-date close snapshot/source required")
    if _instant(terms_available_at) > instant:
        raise ValueError("distribution terms unavailable")
    if distribution.bonus_per_share or distribution.quote_unit != "ordinary_share":
        raise ValueError("share/CDR action requires separate unit adapter")
    positive(distribution.cash_per_share, "cash distribution")
    if any(c.distribution.event_id == distribution.event_id for c in book.cash_claims):
        raise ValueError("duplicate distribution claim")
    quantity = record_date_quantity(book,ts_code,instant,distribution)
    amount = quantity * distribution.cash_per_share
    positive(amount, "claim amount", zero=True)
    claim = CashClaim(ts_code, distribution, quantity, amount, source_id)
    return append(book, event_id, at, dict(kind="cash_record", distribution_id=distribution.event_id,
        ts_code=ts_code, record_quantity=quantity, gross_amount=amount, source_id=source_id,
        terms_available_at=_instant(terms_available_at).isoformat(), cash_delta=0.0),
        cash_claims=book.cash_claims+(claim,))


def accrue_cash_claim(book, *, event_id, distribution_id, processing_at):
    at = clock(book, event_id, processing_at)
    found = [c for c in book.cash_claims if c.distribution.event_id == distribution_id]
    if len(found) != 1 or found[0].recognized:
        raise ValueError("missing/already recognized claim")
    c = found[0]
    if _instant(at).tz_convert("Asia/Shanghai").strftime("%Y%m%d") < c.distribution.ex_date:
        raise ValueError("cannot accrue before ex-date")
    new = replace(c, recognized=True)
    return append(book, event_id, at, dict(kind="cash_accrual", distribution_id=distribution_id,
        receivable_delta=c.gross_amount, cash_delta=0.0, tax_policy="gross_reference_not_final_tax"),
        cash_claims=tuple(new if x is c else x for x in book.cash_claims),
        distribution_income=book.distribution_income+c.gross_amount)


def receive_cash_claim(book, *, event_id, distribution_id, received_at, processing_at,
                       cash_received, tax_withheld, evidence_id):
    at = clock(book, event_id, processing_at)
    received = _instant(received_at)
    found = [c for c in book.cash_claims if c.distribution.event_id == distribution_id]
    if len(found) != 1 or not found[0].recognized or found[0].paid:
        raise ValueError("missing/unrecognized/already paid claim")
    c = found[0]
    if received > _instant(at) or received.tz_convert("Asia/Shanghai").strftime("%Y%m%d") < c.distribution.pay_date:
        raise ValueError("invalid payment receipt time")
    positive(cash_received, "received dividend", zero=True); positive(tax_withheld, "withheld tax", zero=True)
    import math
    if not evidence_id or not math.isclose(cash_received+tax_withheld, c.gross_amount, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("payment amount/provenance mismatch")
    new = replace(c, paid=True)
    return append(book, event_id, at, dict(kind="cash_receipt", distribution_id=distribution_id,
        cash_delta=cash_received, tax_withheld=tax_withheld, evidence_id=evidence_id,
        received_at=received.isoformat(), final_personal_tax_proven=False),
        cash=book.cash+cash_received, distribution_tax_paid=book.distribution_tax_paid+tax_withheld,
        cash_claims=tuple(new if x is c else x for x in book.cash_claims))


def settle_dividend_tax(book, *, event_id, distribution_id, tax_delta,
                        settled_at, received_at, processing_at, evidence_id):
    """Book an explicit currency debit (+) or refund (-), never infer tax law.

    Paid claims survive position closure. The caller supplies an observed
    settlement; this does not establish final investor tax or issuer authenticity.
    """
    import math
    at=clock(book,event_id,processing_at)
    settled,received=_instant(settled_at),_instant(received_at)
    if not settled<=received<=_instant(at):
        raise ValueError('invalid tax settlement chronology')
    if isinstance(tax_delta,bool) or not isinstance(tax_delta,(int,float)) or not math.isfinite(tax_delta) or tax_delta==0:
        raise ValueError('nonzero finite tax currency delta required')
    if not isinstance(evidence_id,str) or not evidence_id.strip():
        raise ValueError('tax receipt evidence required')
    if any(e.get('kind')=='dividend_tax_settlement' and e.get('evidence_id')==evidence_id for e in book.journal):
        raise ValueError('duplicate tax receipt')
    claims=[c for c in book.cash_claims if c.distribution.event_id==distribution_id]
    payments=[e for e in book.journal if e.get('kind')=='cash_receipt' and e.get('distribution_id')==distribution_id]
    if len(claims)!=1 or not claims[0].paid or len(payments)!=1:
        raise ValueError('paid dividend claim required')
    # The payment receipt gives a conservative local ordering boundary.
    if settled<_instant(payments[0]['received_at']):
        raise ValueError('tax settlement predates recorded dividend payment')
    paid=payments[0]['tax_withheld']+sum(e['tax_delta'] for e in book.journal
        if e.get('kind')=='dividend_tax_settlement' and e.get('distribution_id')==distribution_id)
    if paid+tax_delta<0 or paid+tax_delta>claims[0].gross_amount:
        raise ValueError('tax correction outside recorded dividend amount')
    # append rejects insufficient unreserved cash atomically; no silent debt.
    return append(book,event_id,at,dict(kind='dividend_tax_settlement',distribution_id=distribution_id,
        tax_delta=tax_delta,cash_delta=-tax_delta,evidence_id=evidence_id,
        settled_at=settled.isoformat(),received_at=received.isoformat(),final_personal_tax_proven=False),
        cash=book.cash-tax_delta,distribution_tax_paid=book.distribution_tax_paid+tax_delta)
