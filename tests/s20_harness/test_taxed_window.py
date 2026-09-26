import pandas as pd
import pytest

from research.s20_harness.corporate_actions import Distribution
from research.s20_harness.taxed_window import value_cash_window, safe_profit_bounds


def inputs():
    raw = pd.DataFrame({k: [10., 9., 9.] for k in ("open", "high", "low", "close")},
                       index=["20260505", "20260506", "20260507"])
    events = [Distribution("cash", "20260505", "20260506", "20260504", cash_per_share=1.,
                           pay_date="20260507", beneficiary_scope="existing_shareholders_verified")]
    context = dict(acquisition_date="20260505", transfer_settlement_date="20260508",
                   investor_scope="personal_public_market_unrestricted_single_lot",
                   market="SH", cash_rate_basis="gross_per_share")
    return raw, events, context


def test_gross_payment_preserved_and_tax_subtracted_once():
    raw, events, context = inputs()
    lower, upper, report = value_cash_window(raw, events, **context)
    assert lower.close.tolist() == pytest.approx([10., 9.8, 9.8])
    assert lower.cash.tolist() == [0., 0., 1.]
    assert lower.receivable_cash.tolist() == [0., 1., 0.]
    assert upper.close.tolist() == lower.close.tolist()
    assert report["terminal_tax_liability_upper"] == .2
    result = safe_profit_bounds(raw, events, buy_cost=0, sell_cost=.01, **context)
    assert result["terminal_net_lower"] == pytest.approx(-.029)
    assert result["p_class"] == "C"


def test_unknown_exit_retains_profit_class_ambiguity():
    raw, events, context = inputs()
    raw.loc["20260507", ["high", "close"]] = 9.1
    context["transfer_settlement_date"] = None
    result = safe_profit_bounds(raw, events, buy_cost=0, sell_cost=0, **context)
    assert result["class_outer_set"] == ["A", "C"]
    assert result["p_class"] is None
    context["cash_rate_basis"] = "net_per_share"
    with pytest.raises(ValueError, match="double-deduct"):
        value_cash_window(raw, events, **context)


def test_cdr_cannot_silently_use_ordinary_share_tax():
    raw, events, context = inputs()
    context["instrument_type"] = "CDR"
    with pytest.raises(ValueError, match="separate unit"):
        value_cash_window(raw, events, **context)
