import json
from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness.policy_evaluation import build, verify_evaluation
from research.s20_harness.policy_run import build as policy_build
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_policy_evaluation import saved
from tests.s20_harness.test_policy_run import setup
from tests.s20_harness.test_baseline_evaluation import fixture


def test_semantic_evaluation(tmp_path):
    policy, source = saved(tmp_path)
    report = build(tmp_path, policy, digest(policy/"summary.json"), source, digest(source))
    out = Path(report["directory"])
    assert verify_evaluation(out, digest(out/"summary.json"))["maturity_and_metrics_recomputed"]


def test_rehashed_fictional_metric_rejected(tmp_path):
    policy, source = saved(tmp_path)
    report = build(tmp_path, policy, digest(policy/"summary.json"), source, digest(source))
    out = Path(report["directory"])
    metrics = json.loads((out/"metrics.json").read_text())
    metrics["selected_candidates"]["event_bounds"]["known_only_rate"] = 1.0
    atomic_json(out/"metrics.json", metrics)
    report["artifacts"]["metrics.json"] = digest(out/"metrics.json")
    atomic_json(out/"summary.json", report)
    with pytest.raises(ValueError, match="semantic metric"):
        verify_evaluation(out, digest(out/"summary.json"))


def test_downside_evaluation_uses_risk_probability(tmp_path):
    score, config, risk = setup(tmp_path, risk=True)
    policy = Path(policy_build(tmp_path, score, digest(score/"summary.json"), config, digest(config),
                  risk_directory=risk, risk_sha=digest(risk/"summary.json"))["directory"])
    source = tmp_path/"risk_outcome.json"
    atomic_json(source, {"target_id": "P.B5.v4", "evaluation_at": "2024-06-03T00:00:00Z",
                        "outcomes": fixture()[2].to_dict("records")})
    report = build(tmp_path, policy, digest(policy/"summary.json"), source, digest(source))
    out = Path(report["directory"])
    rows = pd.read_parquet(out/"evaluated_outer.parquet")
    original = pd.read_parquet(policy/"outer_candidates.parquet")
    assert rows.score.tolist() == original.risk.tolist()
    assert rows.original_selection_score.tolist() == original.score.tolist()
    assert rows.selected.tolist() == original.selected.tolist()
    assert verify_evaluation(out, digest(out/"summary.json"))["evaluated_score_field"] == "risk"
