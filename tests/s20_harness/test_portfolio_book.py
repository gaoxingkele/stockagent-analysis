import pytest

from research.s20_harness.portfolio_book import (create, reserve_buy, settle_buy,
    reserve_exit, settle_portfolio_exit, accounting_snapshot)
from research.s20_harness.exit_policy import ExitPolicy


CAL = ["20240702", "20240729", "20240730", "20240731"]
POLICY = ExitPolicy("synthetic-policy", "2024-07-01T10:00:00+08:00")


def buy(book, **kwargs):
    args = dict(order_id="buy1", ts_code="stock", buy_date=CAL[0], due_date=CAL[1],
                quantity=100, price_cap=8, fee_cap=1, submitted_at="2024-07-02T09:20:00+08:00")
    args.update(kwargs)
    return reserve_buy(book, **args)


def fill(book, **kwargs):
    args = dict(event_id="buy-fill", order_id="buy1", outcome_at="2024-07-02T09:30:00+08:00",
        received_at="2024-07-02T09:31:00+08:00", processing_at="2024-07-02T09:32:00+08:00",
        filled_quantity=100, price=7, fee=1, reason="synthetic", evidence_id="fixture")
    args.update(kwargs)
    return settle_buy(book, **args)


def test_reservations_prevent_overspending_and_keep_rejected_signals():
    b = buy(create(1000, CAL))
    assert b.cash == 199 and accounting_snapshot(b)["reserved_buy_cash"] == 801
    b = buy(b, order_id="buy2", ts_code="other")
    assert b.journal[-1]["reason"] == "insufficient_cash" and len(b.buys) == 1
    b = buy(b, order_id="repeat")
    assert b.journal[-1]["reason"] == "buy_pending"
    b = fill(b)
    assert b.cash == 299 and len(b.positions) == 1 and not b.buys
    assert accounting_snapshot(b)["nav"] is None


def test_no_fill_and_partial_buy_release_only_unused_reservation():
    b = fill(buy(create(1000, CAL)), filled_quantity=0, price=None, fee=0)
    assert b.cash == 1000 and not b.positions and not b.buys
    b = fill(buy(create(1000, CAL)), filled_quantity=40)
    assert b.cash == 719 and b.positions[0].remaining_quantity == 40
    assert b.journal[-1]["unfilled_cancelled"] == 60
    assert accounting_snapshot(b)["remaining_cost_basis"] == 281


def test_pending_exit_keeps_capital_and_repeat_stock_blocked_until_closed():
    b = fill(buy(create(1000, CAL)))
    b = reserve_exit(b, order_id="exit1", position_id="buy1", policy=POLICY,
                     submitted_at="2024-07-29T14:50:00+08:00")
    b = settle_portfolio_exit(b, event_id="exit-result1", order_id="exit1",
        outcome_at="2024-07-29T15:00:00+08:00", received_at="2024-07-29T15:01:00+08:00",
        processing_at="2024-07-29T15:02:00+08:00", filled_quantity=0,
        reason="suspended", evidence_id="fixture")
    snap = accounting_snapshot(b)
    assert snap["cash"] == 299 and snap["remaining_cost_basis"] == 701 and snap["pending_exits"] == 1
    b = buy(b, order_id="again", buy_date=CAL[2], due_date=CAL[3], submitted_at="2024-07-30T09:10:00+08:00")
    assert b.journal[-1]["reason"] == "already_held"
    b = reserve_exit(b, order_id="exit2", position_id="buy1", policy=POLICY,
                     submitted_at="2024-07-30T09:20:00+08:00")
    b = settle_portfolio_exit(b, event_id="exit-result2", order_id="exit2",
        outcome_at="2024-07-30T09:30:00+08:00", received_at="2024-07-30T09:31:00+08:00",
        processing_at="2024-07-30T09:32:00+08:00", filled_quantity=100, price=6, fee=1,
        reason="synthetic_fill", evidence_id="fixture")
    snap = accounting_snapshot(b)
    assert snap["cash"] == 898 and snap["realized_pnl"] == -102 and snap["nav"] == 898
    assert snap["conservation_error"] == 0 and snap["held_positions"] == 0
    b = buy(b, order_id="new-episode", buy_date=CAL[3], due_date=CAL[3], submitted_at="2024-07-31T09:20:00+08:00")
    assert b.journal[-1]["reason"] == "reserved" and len(b.positions) == 1
    accounting_snapshot(b)


@pytest.mark.parametrize("kwargs", [dict(price=9), dict(fee=2), dict(filled_quantity=101),
    dict(filled_quantity=True), dict(evidence_id=""), dict(outcome_at="2024-07-02T15:00:00+08:00"),
    dict(received_at="2024-07-02T09:29:00+08:00"), dict(filled_quantity=0, price=None, fee=1)])
def test_buy_result_cannot_exceed_caps_or_use_invalid_evidence(kwargs):
    with pytest.raises(ValueError): fill(buy(create(1000, CAL)), **kwargs)


def test_duplicate_and_out_of_order_events_rejected():
    b = buy(create(1000, CAL))
    with pytest.raises(ValueError, match="duplicate"): buy(b)
    b = fill(b)
    with pytest.raises(ValueError): fill(b, event_id="replay")
    with pytest.raises(ValueError, match="out-of-order"): buy(b, order_id="late-backfill")
