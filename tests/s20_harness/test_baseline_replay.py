import json
from pathlib import Path

import pytest

from research.s20_harness.baseline_run import build, verify
from research.s20_harness.baseline_replay import replay
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_baseline_run import plan


def saved(tmp_path):
    path = tmp_path/"plan.json"
    atomic_json(path, plan())
    return Path(build(tmp_path, path, digest(path))["directory"])


def test_exact_replay(tmp_path):
    out = saved(tmp_path)
    report = replay(tmp_path, out, digest(out/"summary.json"))
    assert report["status"] == "VERIFIED_DIAGNOSTIC" and len(report["compared"]) == 6
    assert report["replay_model_fits"] == 1 and not report["formal_stage_accepted"]


def test_rehashed_fictional_parameter_is_not_semantic_evidence(tmp_path):
    out = saved(tmp_path)
    name = "baseline_card.json"
    card = json.loads((out/name).read_text())
    card["intercept"] = [123.]
    atomic_json(out/name, card)
    checkpoint = json.loads((out/"checkpoint.json").read_text())
    checkpoint["artifacts"][name] = digest(out/name)
    atomic_json(out/"checkpoint.json", checkpoint)
    report = json.loads((out/"summary.json").read_text())
    report["artifacts"][name] = digest(out/name)
    report["artifacts"]["checkpoint.json"] = digest(out/"checkpoint.json")
    atomic_json(out/"summary.json", report)
    assert verify(out, digest(out/"summary.json"))["artifact_bytes_verified"]
    with pytest.raises(ValueError, match="semantic replay failed"):
        replay(tmp_path, out, digest(out/"summary.json"))


def test_missing_artifact_cannot_be_omitted_from_manifest(tmp_path):
    out = saved(tmp_path)
    report = json.loads((out/"summary.json").read_text())
    del report["artifacts"]["baseline_card.json"]
    atomic_json(out/"summary.json", report)
    with pytest.raises(ValueError, match="artifact set"):
        verify(out, digest(out/"summary.json"))
