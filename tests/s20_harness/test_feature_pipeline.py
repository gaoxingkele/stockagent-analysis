import pandas as pd
import pytest

from research.s20_harness.feature_pipeline import prepare
from research.s20_harness.splits import SEGMENTS


def fixture():
    boundaries = [dict(name=n, start_at=f"2024-{i+1:02d}-01T00:00:00Z",
                       end_at=f"2024-{i+2:02d}-01T00:00:00Z") for i, n in enumerate(SEGMENTS)]
    rows = []
    for i, name in enumerate(SEGMENTS, 1):
        for j in range(2):
            rows.append(dict(sample_id=f"{name}{j}", prediction_at=f"2024-{i:02d}-02T21:00:00Z",
                             feature_available_at=f"2024-{i:02d}-02T20:00:00Z",
                             horizon_close_at=f"2024-{i:02d}-28T15:00:00Z",
                             label_available_at=f"2024-{i:02d}-28T21:00:00Z"))
    samples = pd.DataFrame(rows)
    values = pd.DataFrame(dict(sample_id=samples.sample_id, x=[1., 3.] + [1000.] * 8))
    contract = dict(role="prediction_features", columns=["x"], source_provenance_verified=True)
    return samples, values, boundaries, contract


def test_fit_only_and_reordered_values():
    samples, values, boundaries, contract = fixture()
    matrix, assigned, report = prepare(samples, values.iloc[::-1], boundaries, contract)
    assert report["center"] == {"x": 2.}
    assert report["scale"] == {"x": 1.}
    assert matrix.x.tolist() == [-1., 1.] + [998.] * 8
    assert matrix.sample_id.tolist() == samples.sample_id.tolist()
    assert not assigned.iloc[-1].supervised_eligible
    assert not report["formal_training_authorized"]


def test_unavailable_not_imputed_and_late_fit_label_not_consumed():
    samples, values, boundaries, contract = fixture()
    samples.loc[1, "label_available_at"] = boundaries[1]["start_at"]
    samples.loc[4, "feature_available_at"] = None
    matrix, _, report = prepare(samples, values, boundaries, contract)
    assert report["fit_sample_ids"] == ["fit0"]
    assert report["center"]["x"] == 1.
    assert pd.isna(matrix.loc[4, "x"])
    assert len(matrix) == 10 and report["feature_unavailable_rows"] == 1


def test_missing_numeric_fit_uses_no_outer_statistics():
    samples, values, boundaries, contract = fixture()
    values.loc[:1, "x"] = float("nan")
    matrix, _, report = prepare(samples, values, boundaries, contract)
    assert report["all_missing_fit_columns"] == ["x"]
    assert matrix.x.tolist() == [0., 0.] + [1000.] * 8


@pytest.mark.parametrize("kind", ["extra", "missing_row", "duplicate", "text", "inf", "provenance", "no_fit"])
def test_invalid_inputs_fail(kind):
    samples, values, boundaries, contract = fixture()
    if kind == "extra":
        values["future_label"] = 1
    elif kind == "missing_row":
        values = values.iloc[1:]
    elif kind == "duplicate":
        values.loc[1, "sample_id"] = values.loc[0, "sample_id"]
    elif kind == "text":
        values["x"] = "1"
    elif kind == "inf":
        values.loc[0, "x"] = float("inf")
    elif kind == "provenance":
        contract["source_provenance_verified"] = False
    else:
        samples.loc[:1, "label_available_at"] = None
    with pytest.raises(ValueError):
        prepare(samples, values, boundaries, contract)
