from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness.baseline_run import build as baseline
from research.s20_harness.policy_run import build
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_baseline_run import plan
from tests.s20_harness.test_policy_search import fixture


def setup(tmp_path, risk=False, floor=1, risk_target="P.B5.v4"):
    p = plan()
    source = tmp_path/"model.json"
    atomic_json(source, p)
    score = Path(baseline(tmp_path, source, digest(source))["directory"])
    args = fixture()
    registry = args[-1]
    registry["min_mature_selected"] = floor
    risk_dir = None
    if risk:
        p["target_id"] = p["policy"]["target_id"] = risk_target
        for candidate_policy in registry["policies"]:
            if candidate_policy["mode"] == "risk_gated":
                candidate_policy["risk_target_id"] = risk_target
        # Separate model fitted to a separately named synthetic risk label.
        for key in ("fit_labels", "calibration_labels"):
            for row in p[key]:
                row["target"] = not row["target"]
        atomic_json(source, p)
        risk_dir = Path(baseline(tmp_path, source, digest(source))["directory"])
    else:
        registry["policies"] = registry["policies"][:1]
    payload = dict(registry=registry, selection_outcomes=args[2].to_dict("records"),
                   selection_calendar=args[4], outer_calendar=p["calendar"])
    policy = tmp_path/"policy.json"
    atomic_json(policy, payload)
    return score, policy, risk_dir


def test_bound_score_only_search_and_outer(tmp_path):
    score, policy, _ = setup(tmp_path)
    report = build(tmp_path, score, digest(score/"summary.json"), policy, digest(policy))
    assert report["outer_candidates"] == 2 and report["outer_selected"] == 1
    assert not report["outer_outcomes_consumed"] and report["new_model_fits"] == 0


def test_independent_fitted_downside_binding(tmp_path):
    score, policy, risk = setup(tmp_path, risk=True)
    report = build(tmp_path, score, digest(score/"summary.json"), policy, digest(policy),
                   risk_directory=risk, risk_sha=digest(risk/"summary.json"))
    assert report["downside_model_bound"] and report["policy_evaluations"] == 2


def test_no_winner_keeps_all_outer_candidates(tmp_path):
    score, policy, _ = setup(tmp_path, floor=100)
    report = build(tmp_path, score, digest(score/"summary.json"), policy, digest(policy))
    rows = pd.read_parquet(Path(report["directory"])/"outer_candidates.parquet")
    assert len(rows) == 2 and not rows.selected.any()
    assert report["selected_policy_id"] is None


def test_missing_risk_binding_rejected(tmp_path):
    score, policy, _ = setup(tmp_path, risk=True)
    with pytest.raises(ValueError, match="pinned downside"):
        build(tmp_path, score, digest(score/"summary.json"), policy, digest(policy))
