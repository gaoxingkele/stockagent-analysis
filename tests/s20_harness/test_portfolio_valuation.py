from dataclasses import replace

import pytest

from research.s20_harness.portfolio_book import create, reserve_exit
from research.s20_harness.exit_policy import ExitPolicy
from research.s20_harness.portfolio_valuation import PriceMark, value_portfolio
from tests.s20_harness.test_portfolio_book import buy, fill, CAL


AT = "2024-07-29T15:00:00+08:00"
KNOWN = "2024-07-29T15:02:00+08:00"


def mark(**kwargs):
    m = PriceMark("stock", AT, "2024-07-29T15:01:00+08:00", "observed", 6,
                  "synthetic", "a"*64)
    return replace(m, **kwargs)


def value(book, marks):
    return value_portfolio(book, marks, valuation_at=AT, knowledge_at=KNOWN)


def test_reference_nav_uses_market_price_not_cost_and_does_not_mutate_cash():
    b = fill(buy(create(1000, CAL)))
    r = value(b, [mark()])
    assert r["nav"] == 899 and r["positions"][0]["market_value"] == 600
    assert r["cumulative_reference_return"] == pytest.approx(-0.101)
    assert b.cash == 299 and r["positions"][0]["remaining_cost_basis"] == 701
    assert not r["source_bytes_verified"] and not r["formal_training_authorized"]


@pytest.mark.parametrize("marks,status", [([], "missing_mark"),
    ([mark(status="suspended", price=None)], "suspended"),
    ([mark(price_at="2024-07-26T15:00:00+08:00")], "stale_mark")])
def test_missing_halted_and_stale_prices_never_turn_into_zero_or_cost(marks, status):
    b = fill(buy(create(1000, CAL)))
    r = value(b, marks)
    assert r["nav"] is None and r["cumulative_reference_return"] is None
    assert r["known_component_value"] == 299 and not r["known_component_is_total_nav"]
    assert r["positions"][0]["valuation_status"] == status
    assert r["positions"][0]["market_value"] is None


def test_reservations_are_cash_assets_not_positions():
    b = buy(create(1000, CAL))
    r = value_portfolio(b, [], valuation_at="2024-07-02T09:25:00+08:00",
                        knowledge_at="2024-07-02T09:26:00+08:00")
    assert r["nav"] == 1000 and r["reserved_buy_cash"] == 801 and not r["held_positions"]


def test_unreported_buy_after_slot_cannot_masquerade_as_all_cash_nav():
    r = value(buy(create(1000, CAL)), [])
    assert r["nav"] is None and r["unresolved_buy_order_ids"] == ["buy1"]
    assert r["unresolved_buy_reserved_cash"] == 801 and r["known_component_value"] == 199


def test_missing_exit_receipt_cannot_assume_original_shares_still_held():
    b = fill(buy(create(1000, CAL)))
    b = reserve_exit(b, order_id="sell", position_id="buy1",
                     policy=ExitPolicy("test", "2024-07-01T10:00:00+08:00"),
                     submitted_at="2024-07-29T14:50:00+08:00")
    r = value(b, [mark()])
    assert r["nav"] is None and r["unresolved_exit_order_ids"] == ["sell"]
    assert r["positions"][0]["valuation_status"] == "exit_outcome_pending"


@pytest.mark.parametrize("kwargs", [dict(price_at="2024-07-30T15:00:00+08:00"),
    dict(available_at="2024-07-29T15:03:00+08:00"), dict(available_at="2024-07-29T14:59:00+08:00"),
    dict(price=0), dict(price=float("inf")), dict(source_sha256="bad"), dict(source_id=""),
    dict(status="suspended"), dict(basis="adjusted_close"), dict(ts_code="other")])
def test_bad_provenance_units_timestamps_and_values_rejected(kwargs):
    with pytest.raises(ValueError): value(fill(buy(create(1000, CAL))), [mark(**kwargs)])


def test_duplicate_marks_and_later_book_snapshot_rejected():
    b = fill(buy(create(1000, CAL)))
    with pytest.raises(ValueError): value(b, [mark(), mark()])
    with pytest.raises(ValueError, match="later processing"):
        value(replace(b, last_processed_at="2024-07-30T09:00:00+08:00"), [mark()])
