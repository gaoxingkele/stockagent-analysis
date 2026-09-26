import pandas as pd
import pytest

from research.s20_harness.recommendation_policy import apply


def fixture():
    frame = pd.DataFrame(dict(sample_id=["b", "a", "c"], entity_id=["B", "A", "C"],
                              signal_date=["20240502"] * 3,
                              prediction_at=["2024-05-02T21:00:00+08:00"] * 3,
                              score=[0.8, 0.8, 0.99], risk=[0.05, 0.05, 0.9]))
    policy = dict(policy_id="fixed1", target_id="P.safe.v4", risk_target_id="P.B5.v4",
                  mode="risk_gated", frozen_at="2024-05-01T21:00:00+08:00",
                  n_cap=1, min_score=0.6, max_risk=0.1)
    return frame, policy, ["20240502", "20240503"]


def test_risk_not_offset_by_score_and_empty_calendar_days():
    rows, report = apply(*fixture())
    assert rows.loc[rows.selected, "sample_id"].tolist() == ["a"]
    assert rows.loc[2, "reject_reason"] == "risk_high"
    assert len(rows) == 3 and report["active_day_coverage"] == 0.5
    assert report["daily"][1]["precision"] is None
    assert report["daily"][1]["selected"] == 0


def test_unknowns_retained_and_abstain():
    frame, policy, calendar = fixture()
    frame.loc[0, "score"] = float("nan")
    frame.loc[1, "risk"] = float("nan")
    rows, report = apply(frame, policy, calendar)
    assert not rows.selected.any() and len(rows) == 3
    assert report["active_day_coverage"] == 0


def test_explicit_no_risk_control():
    frame, policy, calendar = fixture()
    policy.update(mode="score_only_control", max_risk=None, risk_target_id=None)
    rows, report = apply(frame, policy, calendar)
    assert rows.loc[rows.selected, "sample_id"].tolist() == ["c"]
    assert not report["risk_gate_applied"]


def test_baseline_calibration_selection_chain():
    from research.s20_harness.calibration_model import run
    from tests.s20_harness.test_calibration_model import fixture as calibration_fixture

    args = calibration_fixture()
    calibrated, _ = run(*args, target_id="P.safe.v4")
    outer = calibrated.loc[calibrated.segment.eq("outer-test")].copy()
    metadata = args[0].set_index("sample_id")
    candidates = pd.DataFrame({"sample_id": outer.sample_id, "entity_id": outer.sample_id,
                               "prediction_at": outer.sample_id.map(metadata.prediction_at),
                               "signal_date": "20240503", "score": outer.calibrated_probability,
                               "risk": float("nan")})
    # Synthetic prediction is 21:00 UTC, the following date in Shanghai.
    _, policy, _ = fixture()
    policy.update(mode="score_only_control", max_risk=None, risk_target_id=None,
                  min_score=0.0)
    rows, report = apply(candidates, policy, ["20240503", "20240506"])
    assert len(rows) == 2 and report["selected"] == 1
    assert rows.loc[rows.selected, "sample_id"].tolist() == ["outer-test1"]
    assert not report["risk_gate_applied"] and not report["formal_H05_accepted"]


@pytest.mark.parametrize("kind", ["late", "date", "duplicate", "outcome", "threshold", "bool_n", "infinity", "calendar"])
def test_invalid(kind):
    frame, policy, calendar = fixture()
    if kind == "late":
        policy["frozen_at"] = frame.prediction_at.iloc[0]
    elif kind == "date":
        frame.loc[0, "signal_date"] = "20240503"
    elif kind == "duplicate":
        frame.loc[1, "entity_id"] = "B"
    elif kind == "outcome":
        frame["target"] = True
    elif kind == "threshold":
        policy["max_risk"] = float("nan")
    elif kind == "bool_n":
        policy["n_cap"] = True
    elif kind == "infinity":
        frame.loc[0, "score"] = float("inf")
    else:
        calendar.reverse()
    with pytest.raises(ValueError):
        apply(frame, policy, calendar)
