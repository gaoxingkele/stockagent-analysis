import pytest

from research.s20_harness.cash_tax import tax_bounds


def tax(entry, exit, record="20260515"):
    return tax_bounds(1., acquisition_date=entry, transfer_settlement_date=exit,
                      record_date=record, market="SH",
                      investor_scope="personal_public_market_unrestricted_single_lot")


def test_natural_month_and_year_boundaries():
    assert tax("20260501", "20260601")["tax_rates"] == [.2]
    assert tax("20260501", "20260602")["tax_rates"] == [.1]
    assert tax("20250501", "20260501", "20260401")["tax_rates"] == [.1]
    assert tax("20250501", "20260502", "20260401")["tax_rates"] == [0.]


def test_unknown_transfer_and_month_end_are_not_false_exact_tax():
    assert tax("20260501", None)["net_entitlement_lower"] == .8
    result = tax("20260131", "20260228", "20260205")
    assert result["tax_rates"] == [.1, .2]
    assert not result["tax_rate_resolved"]
    assert tax("20260131", "20260305", "20260205")["tax_rates"] == [.1]


def test_invalid_entitlement_or_scope_rejected():
    with pytest.raises(ValueError, match="record-close"):
        tax("20260501", "20260515")
    with pytest.raises(ValueError, match="scope"):
        tax_bounds(1., acquisition_date="20260501", transfer_settlement_date=None,
                   record_date="20260515", investor_scope="institution", market="SH")
