import pandas as pd
import pytest

from research.s20_harness.tradability import LimitState, reference_open_buy, resolve_limits


def frame(up=11, down=9):
    return pd.DataFrame({"ts_code": ["A"], "trade_date": ["20240102"], "up_limit": [up], "down_limit": [down]})


def fill(state, **kwargs):
    args = dict(ts_code="A", trade_date="20240102", decision_at="2024-01-02T09:30:00+08:00", suspended=False)
    args.update(kwargs)
    return reference_open_buy(10, state, **args)


def test_strict_state_requires_known_prior_availability():
    unknown = resolve_limits(frame(), "A", "20240102")
    assert fill(unknown).reason == "limit_availability_unknown"
    prior = resolve_limits(frame(), "A", "20240102", available_at="2024-01-02T09:00:00+08:00")
    assert fill(prior).filled
    assert fill(prior, ts_code="B").reason == "limit_identity_mismatch"
    assert fill(prior, suspended=None).reason == "halt_status_unknown"
    assert fill(prior, decision_at="2024-01-03T09:30:00+08:00").reason == "execution_date_mismatch"


@pytest.mark.parametrize("timestamp", ["2024-01-02T09:30:00+08:00", "2026-09-14T01:00:00+08:00"])
def test_equal_or_retrospective_receipt_cannot_authorize_historical_fill(timestamp):
    state = resolve_limits(frame(), "A", "20240102", available_at=timestamp)
    assert fill(state).reason == "limit_not_available_before_execution"


def test_sentinel_missing_duplicates_and_forged_bounded_are_rejected():
    sentinel = resolve_limits(frame(99999.99, 0), "A", "20240102")
    assert sentinel.status == "sentinel_requires_rule_evidence" and not fill(sentinel).filled
    assert resolve_limits(frame(), "B", "20240102").status == "missing"
    assert resolve_limits(pd.concat([frame(), frame()]), "A", "20240102").status == "duplicate_or_conflicting"
    assert not fill(LimitState("A", "20240102", "bounded")).filled


def test_label_adapter_preserves_recommendation_when_limits_unavailable():
    from research.s20_harness.labels import label_p_track
    daily = pd.DataFrame({"ts_code": ["A", "A"], "trade_date": ["20240102", "20240103"],
                          "open": [10., 10.], "high": [11., 11.], "low": [10., 10.], "close": [11., 11.]})
    state = resolve_limits(frame(), "A", "20240102")
    result = label_p_track(daily, ["20240101", "20240102", "20240103"], "20240101", "A", horizon=2,
                          limit_state=state, execution_at="2024-01-02T09:30:00+08:00", execution_halt_status=False)
    assert result["fill_reason"] == "limit_availability_unknown"
    assert result["p_class"] is None and result["recommendation_kept"]
    assert result["execution_basis"] == "availability_gated_reference"
