import json
from pathlib import Path

import pytest

from research.s20_harness.policy_run import build as policy_build, verify_policy
from research.s20_harness.policy_evaluation import build
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_policy_run import setup
from tests.s20_harness.test_baseline_evaluation import fixture


def saved(tmp_path, floor=1):
    score, policy, _ = setup(tmp_path, floor=floor)
    report = policy_build(tmp_path, score, digest(score/"summary.json"), policy, digest(policy))
    directory = Path(report["directory"])
    payload = {"target_id": "P.safe.v4", "evaluation_at": "2024-06-01T00:00:00Z",
               "outcomes": fixture()[2].to_dict("records")}
    source = tmp_path/"outer.json"
    atomic_json(source, payload)
    return directory, source


def test_policy_outer_evaluation_retains_unmatured(tmp_path):
    directory, source = saved(tmp_path)
    pin = digest(directory/"summary.json")
    report = build(tmp_path, directory, pin, source, digest(source))
    metrics = json.loads((Path(report["directory"])/"metrics.json").read_text())
    assert report["selected"] == 1 and report["rows"] == 2
    assert metrics["selected_candidates"]["event_bounds"]["unknown"] == 1
    assert digest(directory/"summary.json") == pin
    assert not report["policy_reselected"] and not report["models_refitted"]


def test_no_policy_empty_selection_is_null(tmp_path):
    directory, source = saved(tmp_path, floor=100)
    report = build(tmp_path, directory, digest(directory/"summary.json"), source, digest(source))
    metrics = json.loads((Path(report["directory"])/"metrics.json").read_text())
    assert report["selected"] == 0 and report["rows"] == 2
    assert metrics["selected_candidates"]["event_bounds"]["known_only_rate"] is None


def test_missing_artifact_not_accepted(tmp_path):
    directory, _ = saved(tmp_path)
    report = json.loads((directory/"summary.json").read_text())
    del report["artifacts"]["selection_trial_00.parquet"]
    atomic_json(directory/"summary.json", report)
    with pytest.raises(ValueError, match="artifact set"):
        verify_policy(directory, digest(directory/"summary.json"))


def test_selection_labels_cannot_enter_outer_evaluation(tmp_path):
    directory, source = saved(tmp_path)
    payload = json.loads(source.read_text())
    payload["outcomes"][0]["sample_id"] = "selection-policy0"
    atomic_json(source, payload)
    with pytest.raises(ValueError, match="outside evaluation"):
        build(tmp_path, directory, digest(directory/"summary.json"), source, digest(source))
