"""Gross cash ledger plus separate single-lot tax-liability valuation bounds."""
from __future__ import annotations

import math

from .cash_tax import tax_bounds
from .economic_window import value_window


def value_cash_window(raw, events, *, acquisition_date, transfer_settlement_date,
                      investor_scope, market, cash_rate_basis, instrument_type="ordinary_share"):
    if instrument_type != "ordinary_share" or any(event.quote_unit != "ordinary_share" for event in events):
        raise ValueError("instrument requires separate unit and date-effective tax policy")
    if raw.empty:
        raise ValueError("nonempty quote window required")
    if investor_scope != "personal_public_market_unrestricted_single_lot" or market not in ("SH", "SZ"):
        raise ValueError("unsupported investor/market scope")
    if cash_rate_basis != "gross_per_share":
        raise ValueError("gross event cash required; net input would double-deduct tax")
    if any(event.bonus_per_share != 0 for event in events):
        raise ValueError("cash-only tax window cannot handle bonus tax or changing lot quantity")
    if str(raw.index[0]) != acquisition_date:
        raise ValueError("single-lot acquisition must match reference window entry")
    if transfer_settlement_date is not None and transfer_settlement_date < str(raw.index[-1]):
        raise ValueError("transfer cannot precede held window end")
    gross, accounting = value_window(raw, events)
    liability_low = gross.close * 0.
    liability_high = gross.close * 0.
    tax_ledger = []
    for event in events:
        if not acquisition_date <= event.record_date <= str(raw.index[-1]) or not event.cash_per_share:
            continue
        liability = tax_bounds(event.cash_per_share, acquisition_date=acquisition_date,
                               transfer_settlement_date=transfer_settlement_date,
                               record_date=event.record_date, investor_scope=investor_scope, market=market)
        after_ex = gross.index >= event.ex_date
        liability_low.loc[after_ex] += liability["tax_liability_lower"]
        liability_high.loc[after_ex] += liability["tax_liability_upper"]
        tax_ledger.append({"event_id": event.event_id, "recognized_from": event.ex_date,
                           "recognized_in_window": bool(after_ex.any()), **liability})
    lower, upper = gross.copy(), gross.copy()
    for field in ("open", "high", "low", "close"):
        lower[field] -= liability_high
        upper[field] -= liability_low
    # Cash columns remain gross receipts. These are provisions, not a second
    # cash movement or a statement about spendable balance after broker debits.
    report = {"gross_accounting": accounting, "tax_provision_ledger": tax_ledger,
              "terminal_tax_liability_lower": float(liability_low.iloc[-1]),
              "terminal_tax_liability_upper": float(liability_high.iloc[-1]),
              "terminal_economic_lower": float(lower.close.iloc[-1]),
              "terminal_economic_upper": float(upper.close.iloc[-1]),
              "cash_debit_simulated": False, "liability_valuation_only": True,
              "not_for_prediction_features": True, "formal_training_eligible": False}
    return lower, upper, report


def safe_profit_bounds(raw, events, *, buy_cost=.001, sell_cost=.0015, **tax_context):
    if not math.isfinite(buy_cost) or buy_cost < 0 or not math.isfinite(sell_cost) or not 0 <= sell_cost < 1:
        raise ValueError("invalid fixed costs")
    lower, upper, accounting = value_cash_window(raw, events, **tax_context)
    entry_cost = float(raw.open.iloc[0]) * (1 + buy_cost)
    # Only sellable equity incurs reference sale cost; cash, tax provisions and
    # receivables do not. Single-lot cash-only scope has one original share.
    sale_fee = float(raw.close.iloc[-1]) * sell_cost
    net_low = (float(lower.close.iloc[-1]) - sale_fee) / entry_cost - 1
    net_high = (float(upper.close.iloc[-1]) - sale_fee) / entry_cost - 1
    risk_possible = bool(lower.low.min() < entry_cost * .95)
    risk_certain = bool(upper.low.min() < entry_cost * .95)
    up_values = [True] if net_low > 0 else [False] if net_high <= 0 else [False, True]
    risk_values = [True] if risk_certain else [False] if not risk_possible else [False, True]
    # Conservative outer set: marginal bounds alone do not prove every joint
    # class is attainable. Do not multiply these events as independent chances.
    possible = sorted({("B" if r else "A") if u else ("D" if r else "C")
                       for u in up_values for r in risk_values})
    return {"terminal_net_lower": net_low, "terminal_net_upper": net_high,
            "b5_certain": risk_certain, "b5_possible": risk_possible,
            "p_class": possible[0] if len(possible) == 1 else None,
            "class_outer_set": possible, "class_outer_set_is_probability": False,
            "accounting": accounting, "formal_training_eligible": False,
            "execution_basis": "reference_not_verified_fill"}
