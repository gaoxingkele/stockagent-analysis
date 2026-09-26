import pandas as pd
import pytest

from research.s20_harness.paired_comparison import compare


def fixture(days=4):
    calendar = pd.bdate_range("2024-01-01", periods=days).strftime("%Y%m%d").tolist()
    rows = [dict(sample_id=f"{d}-{s}", entity_id=s, signal_date=d, selected=s == "A",
                 safe_profit=s == "A", risk10=s == "B") for d in calendar for s in ("A", "B")]
    candidate = pd.DataFrame(rows)
    baseline = candidate.copy()
    baseline["selected"] = ~baseline.selected
    return candidate, baseline, calendar


def test_daily_pair_not_stock_row_independence():
    report = compare(*fixture(), draws=200)
    bounds = report["equal_date_weight_unknown_delta_bounds"]
    assert bounds["safe_profit_delta_lower"] == 1 and bounds["risk10_delta_upper"] == -1
    assert all(r["intervals"] is None for r in report["block_sensitivity"])
    assert not report["formal_G3_passed"]


def test_unknown_conservative_direction():
    c, b, calendar = fixture(1)
    for frame in (c, b):
        frame["safe_profit"] = None
        frame["risk10"] = None
    report = compare(c, b, calendar, draws=200)
    bounds = report["equal_date_weight_unknown_delta_bounds"]
    assert bounds["safe_profit_delta_lower"] == -1 and bounds["risk10_delta_upper"] == 1


def test_coverage_change_not_paired_gain():
    c, b, calendar = fixture()
    c.loc[0, "selected"] = False
    report = compare(c, b, calendar, draws=200)
    assert not report["all_dates_same_recommendation_count"]
    assert all(r["reason"] == "coverage_mismatch" for r in report["block_sensitivity"])


def test_reproducible_block_sensitivity_and_zero_risk_guard():
    args = fixture(300)
    first = compare(*args, draws=200)
    assert first == compare(*args, draws=200)
    assert all(r["intervals"] is not None for r in first["block_sensitivity"])
    for frame in args[:2]:
        frame["risk10"] = False
    report = compare(*args, draws=200)
    assert all(r["reason"] == "no_observed_risk_events" for r in report["block_sensitivity"])


@pytest.mark.parametrize("kind", ["outcome", "universe", "duplicate", "calendar"])
def test_mismatch_rejected(kind):
    c, b, calendar = fixture()
    if kind == "outcome":
        b.loc[0, "risk10"] = True
    elif kind == "universe":
        b = b.iloc[1:]
    elif kind == "duplicate":
        c.loc[1, "entity_id"] = "A"
    else:
        calendar.reverse()
    with pytest.raises(ValueError):
        compare(c, b, calendar, draws=200)
