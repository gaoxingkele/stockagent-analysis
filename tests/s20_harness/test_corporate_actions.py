import pytest

from research.s20_harness.corporate_actions import Distribution, EconomicPosition


def test_cash_receivable_is_not_available_cash_and_no_double_credit():
    event = Distribution("dividend", "20240102", "20240103", "20231229", 1., pay_date="20240105", beneficiary_scope="existing_shareholders_verified")
    position = EconomicPosition(100.)
    position.record_close("20240102", event)
    assert position.economic_value(10.) == 1000.
    position.advance("20240103")
    assert position.economic_value(9.) == 1000.
    assert position.cash == 0.
    assert position.receivable_cash == 100.
    position.advance("20240105")
    position.advance("20240105")
    assert position.cash == 100.
    assert position.receivable_cash == 0.
    assert position.economic_value(9.) == 1000.


def test_bonus_not_sellable_before_listing_and_record_holdings_fixed():
    event = Distribution("bonus", "20240102", "20240103", "20231229", bonus_per_share=1., bonus_list_date="20240108", beneficiary_scope="existing_shareholders_verified")
    position = EconomicPosition(100.)
    position.record_close("20240102", event)
    position.advance("20240103")
    assert position.shares == 100.
    assert position.pending_bonus_shares == 100.
    assert position.economic_value(5.) == 1000.
    # Selling old shares after ex-date does not cancel earned entitlement.
    position.shares = 0.
    position.cash = 500.
    position.advance("20240108")
    assert position.shares == 100.
    assert position.pending_bonus_shares == 0.
    assert position.economic_value(5.) == 1000.


def test_ex_date_buyer_has_no_old_entitlement():
    position = EconomicPosition(100.)
    position.advance("20240103")
    assert position.economic_value(9.) == 900.
    event = Distribution("old", "20240102", "20240103", "20231229", 1., pay_date="20240105", beneficiary_scope="existing_shareholders_verified")
    with pytest.raises(ValueError, match="backdate"):
        position.record_close("20240102", event)


def test_missing_payment_or_future_announcement_rejected():
    with pytest.raises(ValueError):
        Distribution("missing", "20240102", "20240103", "20231229", 1., beneficiary_scope="existing_shareholders_verified")
    with pytest.raises(ValueError):
        Distribution("late", "20240102", "20240103", "20240104", 1., pay_date="20240105", beneficiary_scope="existing_shareholders_verified")


def test_restructuring_rates_cannot_credit_ordinary_holder():
    with pytest.raises(ValueError, match="beneficiary scope"):
        Distribution("restructuring", "20251127", "20251128", "20251125", bonus_per_share=.599,
                     bonus_list_date="20251128", beneficiary_scope="investors_and_creditors")
    with pytest.raises(ValueError, match="beneficiary scope"):
        Distribution("unknown", "20251127", "20251128", "20251125", bonus_per_share=.599, bonus_list_date="20251128")


def test_cash_entitlement_during_suspension_does_not_require_invented_quote():
    # 603221 reviewed 2026 annual distribution example; supplied holdings are
    # a precondition, not evidence of an executable purchase during suspension.
    event = Distribution("603221_20260812", "20260811", "20260812", "20260807",
                         cash_per_share=.03, pay_date="20260812",
                         beneficiary_scope="existing_shareholders_verified")
    position = EconomicPosition(100.)
    position.record_close("20260811", event)
    position.advance("20260818")
    assert position.cash == pytest.approx(3.)
    assert position.economic_value(27.27) == pytest.approx(2730.)
    assert [entry["date"] for entry in position.ledger] == ["20260811", "20260812", "20260812"]
    position.advance("20260818")
    assert position.cash == pytest.approx(3.)
    assert len(position.ledger) == 3
    # A buyer on resumption has no entitlement to the earlier distribution.
    buyer = EconomicPosition(100.)
    buyer.advance("20260818")
    assert buyer.cash == 0
    with pytest.raises(ValueError, match="backdate"):
        buyer.record_close("20260811", event)
