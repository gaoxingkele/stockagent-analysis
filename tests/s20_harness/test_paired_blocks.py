import pytest

from research.s20_harness.paired_blocks import paired_date_blocks


def report(values):
    return {"daily": [{"date": i, "recommendations": n,
                       "events": {"risk": {"positive": p, "unknown": u, "denominator": n}}}
                      for i, (p, u, n) in enumerate(values)]}


def test_identical_paired_series_cannot_create_spurious_difference():
    data = report([(i % 3, 0, 3) for i in range(180)])
    result = paired_date_blocks(data, data, event="risk", replicates=200)
    assert result["identification_bounds"] == [0, 0]
    assert all(b["reason"] == "degenerate_resampling_distribution" for b in result["blocks"])


def test_unknown_and_empty_dates_preserved():
    a = report([(1, 1, 2), (0, 0, 0)])
    b = report([(0, 0, 2), (0, 0, 0)])
    result = paired_date_blocks(a, b, event="risk", replicates=200)
    assert result["evaluation_days"] == 2
    assert result["identification_bounds"] == [.5, 1]
    assert result["same_daily_recommendation_counts"]
    assert all(x["ci95"] is None for x in result["blocks"])


def test_reproducible_intervals_and_denominator_weighting():
    a = report([(i % 5, 0, 7) for i in range(180)])
    b = report([(i % 7, 0, 8) for i in range(180)])
    first = paired_date_blocks(a, b, event="risk", replicates=200, seed=19)
    assert first == paired_date_blocks(a, b, event="risk", replicates=200, seed=19)
    assert all(x["status"] == "DIAGNOSTIC_INTERVAL" for x in first["blocks"])
    assert not first["same_daily_recommendation_counts"]
    assert not first["formal_promotion_authorized"]


def test_calendar_counts_and_rare_event_guards():
    a = report([(0, 0, 3)] * 180)
    b = report([(1, 0, 3)] * 180)
    assert paired_date_blocks(a, b, event="risk", replicates=200)["blocks"][0]["reason"].startswith("zero_or_all")
    with pytest.raises(ValueError, match="calendars"):
        paired_date_blocks(a, report([(1, 0, 3)]), event="risk")
    with pytest.raises(ValueError, match="conserve"):
        paired_date_blocks(report([(4, 0, 3)]), b, event="risk")
    with pytest.raises(ValueError, match="integer"):
        paired_date_blocks(report([(1.5, 0, 3)]), b, event="risk")
