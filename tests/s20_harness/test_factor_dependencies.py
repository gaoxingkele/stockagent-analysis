import pandas as pd
import pytest

from research.s20_harness.factor_dependencies import assess


def test_factor_conflict_window_and_raw_label_independence():
    rows = [dict(sample_id="s", dependency_id=str(i), event_codes=["old", "new"], window_start=start,
                 window_end="20240626", factor_usage=usage)
            for i, (start, usage) in enumerate([("20240624", "within_window_ratios"),
                                                ("20240625", "within_window_ratios"),
                                                ("20240626", "absolute_factor_levels"),
                                                ("20240624", "none_raw_economic")])]
    anomalies = pd.DataFrame([dict(ts_code="old", event_date="20240625", evidence_id="a")])
    source = pd.DataFrame(rows)
    result = assess(source, anomalies)
    assert result.factor_conflict_evidence_ids.tolist() == [["a"], [], ["a"], []]
    assert result.recommendation_kept.all()
    assert not result.formal_training_eligible.any()
    pd.testing.assert_frame_equal(source, result[source.columns])
    source.loc[0, "window_end"] = "20240623"
    with pytest.raises(ValueError, match="reversed"):
        assess(source, anomalies)
