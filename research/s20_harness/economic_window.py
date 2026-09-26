"""Convert a fixed holding window into value bars with explicit entitlements.

This is accounting, not a fill simulator or a corporate-action coverage claim.
One original share is bought at the first window open and held throughout.
"""
from __future__ import annotations

import pandas as pd

from .corporate_actions import Distribution, EconomicPosition


def value_window(raw: pd.DataFrame, events: list[Distribution]) -> tuple[pd.DataFrame, dict]:
    from .labels import _invalid_quote_dates

    if raw.empty or raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("nonempty unique chronological window required")
    if _invalid_quote_dates(raw[["open", "high", "low", "close"]]):
        raise ValueError("economic valuation needs valid quotes; missing is unknown")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError("duplicate corporate event identity")
    entry_date, final_date = str(raw.index[0]), str(raw.index[-1])
    relevant = [event for event in events if entry_date <= event.record_date <= final_date]
    if any(event.record_date not in raw.index for event in relevant):
        raise ValueError("record-date holdings cannot be inferred across absent sessions")
    position = EconomicPosition(shares=1.)
    rows = []
    for date, quote in raw.iterrows():
        date = str(date)
        position.advance(date)
        row = {field: position.economic_value(float(quote[field])) for field in ("open", "high", "low", "close")}
        row.update(date=date, sellable_shares=position.shares,
                   pending_bonus_shares=position.pending_bonus_shares,
                   cash=position.cash, receivable_cash=position.receivable_cash)
        rows.append(row)
        for event in relevant:
            if event.record_date == date:
                position.record_close(date, event)
    # A record-date entitlement whose ex-date is beyond the horizon is retained
    # in the ledger but not double-counted in a still-cum-distribution quote.
    final = rows[-1]
    report = {"entry_date": entry_date, "horizon_end": final_date,
              "entry_open": float(raw.open.iloc[0]), "original_shares": 1.,
              "terminal_economic_value": final["close"],
              "terminal_sellable_market_value": position.shares * float(raw.close.iloc[-1]),
              "cash": position.cash, "receivable_cash": position.receivable_cash,
              "pending_bonus_shares": position.pending_bonus_shares,
              "settlement_pending": bool(position.receivable_cash or position.pending_bonus_shares),
              "ledger": position.ledger,
              "corporate_action_coverage_proven": False,
              "tax_policy_verified": False, "formal_training_eligible": False}
    return pd.DataFrame(rows).set_index("date"), report
