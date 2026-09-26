import pandas as pd
import pytest

from research.s20_harness.availability_partition import attach


def test_observation_is_not_backdated_and_unknown_retained():
    labels = pd.DataFrame([{"sample_id": str(i), "entity_id": "issuer", "signal_date": "20260529",
        "horizon_end": "20260629", "label_realized": i == 0,
        "taxed_status": "tax_bounds_diagnostic", "taxed_p_class": "A" if i == 0 else None} for i in range(2)])
    result = attach(labels, "2026-09-14T02:00:00+08:00")
    assert len(result) == 2
    assert result.gross_diagnostic_label_available_at.iloc[0] == "2026-09-13T18:00:00+00:00"
    assert pd.isna(result.gross_diagnostic_label_available_at.iloc[1])
    assert not result.historical_replay_availability_proven.any()
    assert not result.formal_training_eligible.any()
    with pytest.raises(ValueError, match="timezone-aware"):
        attach(labels, "2026-09-14")
