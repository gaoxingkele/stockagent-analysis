import pandas as pd
import pytest

from research.s20_harness.fold_comparison import aggregate, load_pair
from tests.s20_harness.test_paired_comparison import fixture


def fold(name, days, offset):
    candidate, baseline, calendar = fixture(days)
    shifted = [(pd.Timestamp(d) + pd.Timedelta(days=offset)).strftime("%Y%m%d") for d in calendar]
    for frame in (candidate, baseline):
        frame["signal_date"] = frame.signal_date.map(dict(zip(calendar, shifted)))
        frame["sample_id"] = name + "-" + frame.sample_id
    return dict(fold_id=name, calendar=shifted, candidate=candidate, baseline=baseline)


def test_date_weighted_not_fold_weighted():
    a, b = fold("a", 2, 0), fold("b", 4, 40)
    b["candidate"], b["baseline"] = b["baseline"], b["candidate"]
    result = aggregate([a, b], draws=200)
    assert result["unique_calendar_days"] == 6
    assert result["equal_date_weight_unknown_delta_bounds"]["safe_profit_delta_lower"] == pytest.approx(-1 / 3)
    assert result["pooled_confidence_interval"] is None and not result["formal_G3_passed"]


def test_seed_repeat_cannot_add_dates():
    with pytest.raises(ValueError, match="overlapping"):
        aggregate([fold("seed1", 3, 0), fold("seed2", 3, 0)], draws=200)


def test_fold_order_and_duplicate_id_rejected():
    with pytest.raises(ValueError, match="chronologically"):
        aggregate([fold("later", 3, 40), fold("earlier", 3, 0)], draws=200)
    with pytest.raises(ValueError, match="unique fold"):
        aggregate([fold("same", 3, 0), fold("same", 3, 40)], draws=200)


def test_empty_selection_does_not_gain_precision():
    a = fold("empty", 3, 0)
    a["candidate"]["selected"] = False
    a["baseline"]["selected"] = False
    result = aggregate([a], draws=200)
    assert result["matched_active_days"] == 0
    assert all(v is None for v in result["equal_date_weight_unknown_delta_bounds"].values())


def test_b5_cannot_be_declared_as_reference_b10():
    with pytest.raises(ValueError, match="P.B10"):
        load_pair({}, {}, safe_target_id="P.safe.v4", risk10_target_id="P.B5.v4")


def test_bound_evaluation_pair_integration(tmp_path):
    from pathlib import Path
    from research.s20_harness.policy_run import build as policy_build
    from research.s20_harness.policy_evaluation import build as evaluation_build
    from research.s20_harness.runtime import atomic_json, digest
    from tests.s20_harness.test_policy_run import setup
    from tests.s20_harness.test_baseline_evaluation import fixture as outcome_fixture
    from research.s20_harness.fold_comparison import compare_bound

    score, config, risk = setup(tmp_path, risk=True, risk_target="P.B10.v4")
    policy = Path(policy_build(tmp_path, score, digest(score/"summary.json"), config, digest(config),
                   risk_directory=risk, risk_sha=digest(risk/"summary.json"))["directory"])
    refs = []
    for target in ("P.safe.v4", "P.B10.v4"):
        source = tmp_path/"outcome.json"
        records = outcome_fixture()[2].to_dict("records")
        if target == "P.B10.v4":
            for record in records:
                record["target"] = not record["target"]
        atomic_json(source, {"target_id": target, "evaluation_at": "2024-06-03T00:00:00Z",
                            "outcomes": records})
        out = Path(evaluation_build(tmp_path, policy, digest(policy/"summary.json"), source, digest(source))["directory"])
        refs.append({"directory": str(out), "summary_sha256": digest(out/"summary.json")})
    report = compare_bound(refs, refs, safe_target_id="P.safe.v4", risk10_target_id="P.B10.v4", draws=200)
    assert report["comparison"]["equal_date_weight_unknown_delta_bounds"]["safe_profit_delta_lower"] == 0
    assert report["candidate_binding"]["evidence_mode"] == "synthetic"
    assert not report["formal_G3_passed"]
