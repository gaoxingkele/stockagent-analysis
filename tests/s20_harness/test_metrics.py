import pandas as pd
import pytest

from research.s20_harness.metrics import recommendation_report, binary_bounds


def test_unknown_outcomes_and_inactive_days_remain_in_denominators():
    recs = pd.DataFrame([{"recommendation_id": str(i), "sample_id": str(i), "entity_id": "same",
        "signal_date": "20260529", "fill_status": "filled" if i == 0 else "unfilled", "episode_id": None}
        for i in range(3)])
    outcomes = pd.DataFrame({"sample_id": ["0", "1"], "profit": [True, False]})
    report = recommendation_report(recs, outcomes, ["20260529", "20260601"], event_columns=["profit"])
    bounds = report["all_recommendation_events"]["profit"]
    assert bounds["rate_lower"] == 1/3 and bounds["rate_upper"] == 2/3
    assert bounds["known_only_rate"] == .5
    assert report["filled_only_events"]["profit"]["rate_lower"] == 1
    assert report["daily"][1]["events"]["profit"]["rate_lower"] is None
    assert report["active_day_coverage"] == .5
    assert report["unique_entities"] == 1 and report["missing_episode_rows"] == 3
    with pytest.raises(ValueError, match="recommendation identity"):
        recommendation_report(pd.concat([recs, recs]), outcomes, ["20260529"], event_columns=["profit"])


def test_empty_and_nonboolean_outcomes():
    assert binary_bounds([])["rate_upper"] is None
    assert binary_bounds([None])["rate_upper"] == 1
    with pytest.raises(ValueError, match="bool or unknown"):
        binary_bounds([.9])
