import pytest

from research.s20_harness.label_availability import available_before, label_maturity


def test_late_data_controls_maturity_not_price_date():
    result = label_maturity(horizon_close_at="2024-01-30T15:00:00+08:00",
                            input_available_at=["2024-02-01T21:00:00+08:00"])
    assert not available_before(result, "2024-01-31T21:00:00+08:00")
    assert not available_before(result, "2024-02-01T21:00:00+08:00")
    assert available_before(result, "2024-02-02T21:00:00+08:00")


def test_early_failure_does_not_skip_horizon_and_pending_exit_waits():
    args = dict(horizon_close_at="2024-01-30T15:00:00+08:00",
                input_available_at=["2024-01-03T21:00:00+08:00"])
    assert not available_before(label_maturity(**args), "2024-01-04T21:00:00+08:00")
    assert label_maturity(**args, trading_exit_required=True)["reason"] == "exit_pending"
    result = label_maturity(**args, trading_exit_required=True, exit_confirmed_at="2024-02-05T10:00:00+08:00")
    assert not available_before(result, "2024-02-01T21:00:00+08:00")


def test_missing_or_unresolved_never_matures():
    args = dict(horizon_close_at="2024-01-30T15:00:00+08:00")
    for values in ([], [None]):
        assert not label_maturity(**args, input_available_at=values)["mature"]
    assert not label_maturity(**args, input_available_at=["2024-02-01T21:00:00+08:00"], unresolved=True)["mature"]
    with pytest.raises(ValueError, match="timezone-aware"):
        label_maturity(**args, input_available_at=["2024-02-01"])
