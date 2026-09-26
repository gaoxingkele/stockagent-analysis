import pandas as pd
import pytest

from research.s20_harness.splits import SEGMENTS, assign_segments, training_ids


def boundaries():
    return [dict(name=name, start_at=f"2024-{i+1:02d}-01T00:00:00+08:00", end_at=f"2024-{i+2:02d}-01T00:00:00+08:00")
            for i, name in enumerate(SEGMENTS)]


def samples():
    return pd.DataFrame([
        dict(sample_id="fit_ok", prediction_at="2024-01-02T21:00:00+08:00", feature_available_at="2024-01-02T20:00:00+08:00",
             horizon_close_at="2024-01-30T15:00:00+08:00", label_available_at="2024-01-30T21:00:00+08:00"),
        dict(sample_id="fit_late", prediction_at="2024-01-02T21:00:00+08:00", feature_available_at="2024-01-02T20:00:00+08:00",
             horizon_close_at="2024-01-30T15:00:00+08:00", label_available_at="2024-02-01T00:00:00+08:00"),
        dict(sample_id="outer_unknown", prediction_at="2024-05-02T21:00:00+08:00", feature_available_at="2024-05-02T20:00:00+08:00",
             horizon_close_at="2024-05-30T15:00:00+08:00", label_available_at=None)])


def test_boundary_purge_keeps_all_rows_and_outer_unknowns():
    assigned, report = assign_segments(samples(), boundaries(), evaluation_at="2024-06-02T21:00:00+08:00")
    assert training_ids(assigned, "fit") == ["fit_ok"]
    assert len(assigned) == 3 and report["all_input_rows_retained"]
    assert assigned.segment.tolist() == ["fit", "fit", "outer-test"]
    assert not assigned.evaluation_eligible.any()
    with pytest.raises(ValueError, match="development"):
        training_ids(assigned, "outer-test")


def test_future_features_and_early_negative_feedback():
    data = samples()
    data.loc[0, "feature_available_at"] = data.loc[0, "prediction_at"]
    assigned, _ = assign_segments(data, boundaries())
    assert not assigned.supervised_eligible.any()
    data.loc[0, "label_available_at"] = "2024-01-10T21:00:00+08:00"
    with pytest.raises(ValueError, match="full-window"):
        assign_segments(data, boundaries())


def test_naive_times_overlap_and_duplicate_ids_rejected():
    data = samples()
    data.loc[0, "prediction_at"] = "2024-01-02"
    with pytest.raises(ValueError, match="timezone-aware"):
        assign_segments(data, boundaries())
    split = boundaries()
    split[1]["start_at"] = split[0]["start_at"]
    with pytest.raises(ValueError, match="overlapping"):
        assign_segments(samples(), split)
    with pytest.raises(ValueError, match="unique"):
        assign_segments(pd.concat([samples(), samples()]), boundaries())


def test_every_stage_has_separate_cutoff_and_unknown_features_stay_visible():
    rows = []
    for i, name in enumerate(SEGMENTS, start=1):
        rows.append(dict(sample_id=name, prediction_at=f"2024-{i:02d}-02T21:00:00+08:00",
                         feature_available_at=f"2024-{i:02d}-02T20:00:00+08:00",
                         horizon_close_at=f"2024-{i:02d}-28T15:00:00+08:00",
                         label_available_at=f"2024-{i:02d}-28T21:00:00+08:00"))
    data = pd.DataFrame(rows)
    assigned, _ = assign_segments(data, boundaries(), evaluation_at="2024-06-02T21:00:00+08:00")
    assert assigned.segment.tolist() == list(SEGMENTS)
    assert assigned.supervised_eligible.tolist() == [True, True, True, True, False]
    assert assigned.evaluation_eligible.tolist() == [False, False, False, False, True]
    data.loc[2, "feature_available_at"] = None
    assigned, _ = assign_segments(data, boundaries())
    assert assigned.iloc[2].segment == "calibration"
    assert training_ids(assigned, "calibration") == []
