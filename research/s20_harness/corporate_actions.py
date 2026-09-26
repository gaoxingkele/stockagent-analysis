"""Unit-notional entitlement accounting; cash/bonus availability stays explicit.

Rates must already be normalized under a declared tax policy. This module does
not infer investor-specific tax from vendor cash_div or resolve conflicting
announcements. Rights issues, mergers and fractional-share settlement need
separate audited adapters and cannot be silently treated as cash dividends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Distribution:
    event_id: str
    record_date: str
    ex_date: str
    known_date: str
    cash_per_share: float = 0.
    bonus_per_share: float = 0.
    pay_date: str | None = None
    bonus_list_date: str | None = None
    beneficiary_scope: str = "unverified"
    quote_unit: str = "ordinary_share"

    def __post_init__(self):
        valid_scope = ((self.beneficiary_scope == "existing_shareholders_verified" and self.quote_unit == "ordinary_share") or
                       (self.beneficiary_scope == "registered_cdr_holders_verified" and self.quote_unit == "domestic_CDR"))
        if not valid_scope:
            raise ValueError("ordinary-holder entitlement requires verified beneficiary scope; restructuring is not ordinary bonus shares")
        if not self.event_id or self.record_date >= self.ex_date or self.known_date > self.record_date:
            raise ValueError("event identity or causal dates invalid")
        for rate in (self.cash_per_share, self.bonus_per_share):
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("invalid distribution rate")
        if self.cash_per_share and (not self.pay_date or self.pay_date < self.ex_date):
            raise ValueError("cash requires an audited payment date")
        if self.bonus_per_share and (not self.bonus_list_date or self.bonus_list_date < self.ex_date):
            raise ValueError("bonus requires an audited listing date")


@dataclass
class Entitlement:
    event: Distribution
    record_shares: float
    recognized: bool = False
    cash_paid: bool = False
    bonus_listed: bool = False


@dataclass
class EconomicPosition:
    shares: float
    cash: float = 0.
    receivable_cash: float = 0.
    pending_bonus_shares: float = 0.
    entitlements: dict[str, Entitlement] = field(default_factory=dict)
    ledger: list[dict] = field(default_factory=list)
    last_advanced_date: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.shares) or self.shares < 0 or not math.isfinite(self.cash):
            raise ValueError("invalid opening position")

    def record_close(self, date: str, event: Distribution):
        """Caller supplies end-of-record-date holdings after that day's fills."""
        if date != event.record_date or date < event.known_date:
            raise ValueError("entitlement must be captured at known record-date close")
        if event.event_id in self.entitlements:
            raise ValueError("duplicate entitlement")
        if self.last_advanced_date is not None and date < self.last_advanced_date:
            raise ValueError("cannot backdate entitlement")
        self.entitlements[event.event_id] = Entitlement(event, self.shares)
        self.ledger.append({"date": date, "event_id": event.event_id, "action": "record", "shares": self.shares})

    def advance(self, date: str):
        """Apply all intervening calendar dates before valuation/trading at date."""
        if self.last_advanced_date is not None and date < self.last_advanced_date:
            raise ValueError("cannot reverse accounting time")
        self.last_advanced_date = date
        for entitlement in self.entitlements.values():
            event = entitlement.event
            qty = entitlement.record_shares
            if not entitlement.recognized and date >= event.ex_date:
                self.receivable_cash += qty * event.cash_per_share
                self.pending_bonus_shares += qty * event.bonus_per_share
                entitlement.recognized = True
                self.ledger.append({"date": event.ex_date, "event_id": event.event_id, "action": "recognize",
                                    "cash_receivable": qty * event.cash_per_share, "bonus_pending": qty * event.bonus_per_share})
            if entitlement.recognized and event.cash_per_share and not entitlement.cash_paid and date >= event.pay_date:
                amount = qty * event.cash_per_share
                self.receivable_cash -= amount
                self.cash += amount
                entitlement.cash_paid = True
                self.ledger.append({"date": event.pay_date, "event_id": event.event_id, "action": "pay", "cash": amount})
            if entitlement.recognized and event.bonus_per_share and not entitlement.bonus_listed and date >= event.bonus_list_date:
                amount = qty * event.bonus_per_share
                self.pending_bonus_shares -= amount
                self.shares += amount
                entitlement.bonus_listed = True
                self.ledger.append({"date": event.bonus_list_date, "event_id": event.event_id, "action": "list_bonus", "shares": amount})

    def economic_value(self, price: float):
        if not math.isfinite(price) or price <= 0:
            raise ValueError("unknown price is not zero or a guaranteed unchanged valuation")
        return (self.shares + self.pending_bonus_shares) * price + self.cash + self.receivable_cash
