import pytest

from research.s20_harness.exit_book import open_lot, apply_exit


CAL = ["20240702", "20240729", "20240730", "20240802"]


def lot():
    return open_lot("p", "600027.SH", CAL[0], CAL[1], 200, 1402.0, CAL)


def attempt(state, **kwargs):
    args = dict(attempt_id="a", position_id="p", ts_code="600027.SH", trade_date=CAL[1],
                calendar=CAL, filled_quantity=0, reason="suspended", evidence_id="fixture")
    args.update(kwargs)
    return apply_exit(state, **args)


def test_suspension_retains_position_and_no_fictitious_cash_or_value():
    original = lot()
    state, record = attempt(original)
    assert state.remaining_quantity == 200 and state.net_exit_cash == 0
    assert state.status == "EXIT_PENDING" and original.status == "OPEN"
    assert record["remaining_cost_basis"] == 1402 and record["remaining_market_value"] is None
    assert record["risk_exposure_continues"] and record["original_horizon_label_unchanged"]


def test_partial_and_delayed_exit_conserve_quantity_cash_cost_and_pnl():
    state, _ = attempt(lot())
    state, first = attempt(state, attempt_id="b", trade_date=CAL[2], filled_quantity=80,
                           price=6.0, fee=5.0, reason="partial_reference_fill")
    assert state.remaining_quantity == 120 and state.net_exit_cash == 475
    assert first["remaining_cost_basis"] == pytest.approx(841.2)
    state, last = attempt(state, attempt_id="c", trade_date=CAL[3], filled_quantity=120,
                          price=5.0, fee=5.0, reason="reference_fill")
    assert state.status == "CLOSED" and state.remaining_quantity == 0
    assert state.net_exit_cash == 1070 and state.exit_fees == 10
    assert state.realized_cost == 1402 and state.realized_pnl == -332
    assert not last["exit_pending"] and last["remaining_market_value"] == 0
    with pytest.raises(ValueError, match="closed"): attempt(state, attempt_id="d", trade_date=CAL[3])


@pytest.mark.parametrize("kwargs", [dict(price=7), dict(fee=1), dict(filled_quantity=201, price=7),
    dict(filled_quantity=-1), dict(filled_quantity=True), dict(position_id="other"),
    dict(ts_code="other"), dict(trade_date=CAL[0]), dict(trade_date="20240728"),
    dict(evidence_id=""), dict(reason=""), dict(filled_quantity=1, price=float("nan")),
    dict(filled_quantity=1, price=7, fee=8), dict(fee=-1)])
def test_invalid_attempts_rejected(kwargs):
    with pytest.raises(ValueError): attempt(lot(), **kwargs)


def test_duplicate_out_of_order_and_t_plus_one_rejected():
    state, _ = attempt(lot(), trade_date=CAL[2])
    with pytest.raises(ValueError, match="duplicate"): attempt(state, trade_date=CAL[2])
    with pytest.raises(ValueError, match="out-of-order"): attempt(state, attempt_id="b")
    same_day = open_lot("p", "600027.SH", CAL[0], CAL[0], 100, 701, CAL)
    with pytest.raises(ValueError, match="T\\+1"):
        attempt(same_day, trade_date=CAL[0], filled_quantity=100, price=7)


def test_invalid_calendar_and_entry_cost():
    with pytest.raises(ValueError): open_lot("p", "s", CAL[0], CAL[1], 100, 10, CAL[::-1])
    with pytest.raises(ValueError): open_lot("p", "s", CAL[0], CAL[1], 100, float("inf"), CAL)
