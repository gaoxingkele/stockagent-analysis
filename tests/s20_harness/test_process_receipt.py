import json
from pathlib import Path

import pytest

from research.s20_harness.bounded_baseline import run, verify_process
from research.s20_harness.process_runner import Limits
from research.s20_harness.runtime import atomic_json, digest
from research.s20_harness.trial_budget import Budget
from tests.s20_harness.test_trial_budget import contract
from tests.s20_harness.test_baseline_run import plan


@pytest.fixture
def completed(tmp_path):
    source = tmp_path/"input.json"
    atomic_json(source, plan())
    pin = digest(source)
    limits = Limits(120, 2*1024**3, 1)
    budget = Budget(tmp_path/"budget.sqlite", contract(pin))
    result = run(tmp_path, source, pin, budget, "baseline", "one", limits)
    return tmp_path, result["artifact"], source, pin, limits


def test_recorded_completion(completed):
    assert verify_process(*completed)["limits_and_input_bound"]


@pytest.mark.parametrize("kind", ["unhashed", "limits", "command", "exit"])
def test_wrong_receipt_rejected(completed, kind):
    root, artifact, source, pin, limits = completed
    path = Path(artifact["process_directory"])/"process_result.json"
    data = json.loads(path.read_text())
    if kind in ("unhashed", "limits"):
        data["limits"]["wall_seconds"] = 10000
    elif kind == "command":
        data["command"][-3] = "wrong-input-hash"
    else:
        data["exit_code"] = 1
    atomic_json(path, data)
    if kind != "unhashed":
        artifact["process_summary_sha256"] = digest(path)
    with pytest.raises(ValueError):
        verify_process(root, artifact, source, pin, limits)
