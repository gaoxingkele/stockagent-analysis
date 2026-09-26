"""Ordinary SH/SZ personal cash dividend tax, explicit single-lot scope.

Sources: SAT Caishui 2015 No.101 and 2012 No.85. Not a broker tax engine.
Acquisition/transfer dates must be actual legally relevant dates, not inferred
from a 20-market-session horizon. FIFO multi-lot allocation belongs in T ledger.
"""
from __future__ import annotations

from datetime import datetime
import math


POLICY_ID = "shsz_personal_public_cash_2015_101_single_lot_v1"


def _date(value):
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ValueError("explicit YYYYMMDD date required")
    return datetime.strptime(value, "%Y%m%d").date()


def tax_bounds(gross_cash, *, acquisition_date, transfer_settlement_date,
               record_date, investor_scope, market):
    if investor_scope != "personal_public_market_unrestricted_single_lot" or market not in ("SH", "SZ"):
        raise ValueError("unsupported investor/market tax scope")
    if not math.isfinite(gross_cash) or gross_cash < 0:
        raise ValueError("invalid gross cash")
    acquired, recorded = _date(acquisition_date), _date(record_date)
    if recorded <= _date("20150908") or recorded < acquired:
        raise ValueError("out-of-policy date or no record-date holding")
    rates, reason = [0., .1, .2], "transfer_unknown"
    if transfer_settlement_date is not None:
        transfer = _date(transfer_settlement_date)
        if transfer <= recorded:
            raise ValueError("transfer does not establish record-close entitlement")
        # Calendar anniversaries, not 30/365 fixed-day shortcuts. If an
        # anniversary date does not exist, retain adjacent-tier uncertainty.
        try:
            anniversary = acquired.replace(year=acquired.year + 1)
        except ValueError:
            anniversary = None
        next_year, next_month = acquired.year + (acquired.month == 12), acquired.month % 12 + 1
        try:
            month = acquired.replace(year=next_year, month=next_month)
        except ValueError:
            month = None
        if anniversary is not None and transfer > anniversary:
            rates, reason = [0.], "over_natural_year"
        elif anniversary is None and transfer.year >= acquired.year + 1:
            rates, reason = [0., .1], "leap_anniversary_requires_settlement_rule"
        elif month is not None:
            rates, reason = ([.2], "within_natural_month") if transfer <= month else ([.1], "over_month_within_year")
        elif (transfer.year, transfer.month) <= (next_year, next_month):
            rates, reason = [.1, .2], "month_end_anniversary_requires_settlement_rule"
        else:
            rates, reason = [.1], "over_month_within_year"
    low, high = min(rates), max(rates)
    return {"policy_id": POLICY_ID, "tax_rates": rates, "reason": reason,
            "gross_cash": gross_cash, "tax_liability_lower": gross_cash * low,
            "tax_liability_upper": gross_cash * high,
            "net_entitlement_lower": gross_cash * (1 - high),
            "net_entitlement_upper": gross_cash * (1 - low),
            "tax_rate_resolved": len(rates) == 1,
            "cash_payment_timing": "separate_gross_receipt_and_later_tax_debit",
            "formal_training_authorized": False}
