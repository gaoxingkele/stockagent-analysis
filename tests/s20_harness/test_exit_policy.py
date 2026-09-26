from dataclasses import replace

import pytest

from research.s20_harness.exit_book import open_lot
from research.s20_harness.exit_policy import ExitPolicy, request_exit, settle_exit


CAL = ["20240702", "20240729", "20240730", "20240731"]
POLICY = ExitPolicy("test-policy", "2024-07-01T12:00:00+08:00")


def setup():
    lot = open_lot("p", "s", CAL[0], CAL[1], 100, 701, CAL)
    req = request_exit(lot, POLICY, order_id="a", submitted_at="2024-07-29T14:50:00+08:00", calendar=CAL)
    return lot, req


def settle(lot, req, **kwargs):
    args = dict(outcome_at="2024-07-29T15:00:00+08:00", received_at="2024-07-29T15:01:00+08:00",
                processing_at="2024-07-29T15:02:00+08:00", filled_quantity=0,
                reason="suspended", evidence_id="synthetic-test")
    args.update(kwargs)
    return settle_exit(lot, req, **args)


def test_due_close_failure_then_next_open_partial_preserves_pending():
    lot, req = setup()
    lot, e = settle(lot, req)
    assert e["phase"] == "due_close" and lot.net_exit_cash == 0
    req = request_exit(lot, POLICY, order_id="b", submitted_at="2024-07-30T09:20:00+08:00", calendar=CAL)
    assert req.phase == "retry_open"
    lot, e = settle(lot, req, outcome_at="2024-07-30T09:30:00+08:00",
        received_at="2024-07-30T09:31:00+08:00", processing_at="2024-07-30T09:32:00+08:00",
        filled_quantity=40, price=6, fee=5, reason="partial_test_fill")
    assert lot.remaining_quantity == 60 and lot.net_exit_cash == 235
    assert e["exit_pending"] and not e["fill_independently_verified"]


@pytest.mark.parametrize("stamp", ["2024-07-29T15:00:00+08:00", "2024-07-30T09:20:00+08:00", "2024-07-29T14:50:00"])
def test_late_skipped_or_naive_submission_rejected(stamp):
    lot, _ = setup()
    with pytest.raises(ValueError): request_exit(lot, POLICY, order_id="a", submitted_at=stamp, calendar=CAL)


@pytest.mark.parametrize("kwargs", [dict(outcome_at="2024-07-29T14:59:00+08:00"),
    dict(received_at="2024-07-29T14:59:00+08:00"), dict(processing_at="2024-07-29T15:00:00+08:00")])
def test_future_or_wrong_slot_result_rejected(kwargs):
    lot, req = setup()
    with pytest.raises(ValueError): settle(lot, req, **kwargs)


def test_stale_order_tampered_slot_and_policy_lookahead_rejected():
    lot, req = setup()
    changed, _ = settle(lot, req)
    with pytest.raises(ValueError, match="stale"): settle(changed, req)
    with pytest.raises(ValueError, match="schedule"): settle(lot, replace(req, phase="retry_open"))
    with pytest.raises(ValueError, match="frozen"):
        request_exit(lot, replace(POLICY, frozen_at="2024-07-30T12:00:00+08:00"), order_id="a",
                     submitted_at=req.submitted_at, calendar=CAL)


def test_retry_cannot_jump_over_unrecorded_market_session():
    lot, req = setup(); lot, _ = settle(lot, req)
    with pytest.raises(ValueError, match="next scheduled"):
        request_exit(lot, POLICY, order_id="b", submitted_at="2024-07-31T09:20:00+08:00", calendar=CAL)


def test_pending_lifecycle_cannot_switch_policy():
    lot, req = setup(); lot, _ = settle(lot, req)
    with pytest.raises(ValueError, match="policy changed"):
        request_exit(lot, replace(POLICY, policy_id="new"), order_id="b",
                     submitted_at="2024-07-30T09:20:00+08:00", calendar=CAL)


def test_retry_cannot_delete_a_session_from_calendar():
    lot, req = setup(); lot, _ = settle(lot, req)
    with pytest.raises(ValueError, match="calendar changed"):
        request_exit(lot, POLICY, order_id="b", submitted_at="2024-07-31T09:20:00+08:00",
                     calendar=[CAL[0], CAL[1], CAL[3]])
