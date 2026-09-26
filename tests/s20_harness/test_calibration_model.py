import pandas as pd
import pytest

from research.s20_harness.baseline_model import run as baseline
from research.s20_harness.calibration_model import run
from tests.s20_harness.test_baseline_model import inputs


def fixture():
    samples, features, fit_labels, boundaries, contract = inputs()
    features["x"] = [1., 3.] * 5
    predictions, card = baseline(samples, features, fit_labels, boundaries, contract, target_id="P.safe.v4")
    labels = pd.DataFrame({"sample_id": ["calibration1", "calibration0"], "target": [True, False]})
    return samples, predictions, labels, boundaries, card


def test_baseline_calibration_integration_forward_only():
    output, card = run(*fixture(), target_id="P.safe.v4")
    assert len(output) == 10 and card["calibrated_rows"] == 4
    assert output.calibrated_probability.iloc[:6].isna().all()
    assert output.calibrated_probability.iloc[6:].between(0, 1).all()
    assert output.calibrated_probability.iloc[6] < output.calibrated_probability.iloc[7]
    assert card["calibration_sample_ids"] == ["calibration0", "calibration1"]
    assert not card["selected_subset_reliability_verified"]


def test_later_scores_cannot_change_calibrator():
    args = fixture()
    _, first = run(*args, target_id="P.safe.v4")
    args[1].loc[6:, "raw_probability"] = 0.99
    _, second = run(*args, target_id="P.safe.v4")
    assert first["coefficients"] == second["coefficients"]
    assert first["intercept"] == second["intercept"]


def test_unknown_outer_labels_and_unavailable_features_retained():
    args = fixture()
    args[0].loc[8:, "label_available_at"] = None
    args[0].loc[8, "feature_available_at"] = None
    args[1].loc[8, ["feature_ready", "raw_probability", "prediction_status"]] = [False, float("nan"), "feature_unavailable"]
    output, card = run(*args, target_id="P.safe.v4")
    assert len(output) == 10 and card["calibrated_rows"] == 3
    assert output.loc[8, "calibration_status"] == "feature_unavailable"
    assert pd.notna(output.loc[9, "calibrated_probability"])


@pytest.mark.parametrize("kind", ["outer_label", "purged", "target", "split", "probability", "in_sample", "segment", "duplicate", "single_class", "status"])
def test_firewall(kind):
    args = list(fixture())
    if kind == "outer_label":
        args[2] = pd.concat([args[2], pd.DataFrame({"sample_id": ["outer-test0"], "target": [True]})])
    elif kind == "purged":
        args[0].loc[5, "label_available_at"] = args[3][3]["start_at"]
    elif kind == "target":
        args[4]["target_id"] = "O.other"
    elif kind == "split":
        args[4]["preprocessing"]["split"]["split_protocol_sha256"] = "bad"
    elif kind == "probability":
        args[1].loc[8, "raw_probability"] = float("inf")
    elif kind == "in_sample":
        args[1].loc[0, "raw_probability"] = 0.5
    elif kind == "segment":
        args[1].loc[4, "segment"] = "fit"
    elif kind == "duplicate":
        args[1].loc[4, "sample_id"] = "fit0"
    elif kind == "single_class":
        args[2]["target"] = True
    else:
        args[1].loc[8, "prediction_status"] = "unknown"
    with pytest.raises(ValueError):
        run(*args, target_id="P.safe.v4")
