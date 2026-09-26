import json
import sys

import pytest

from research.s20_harness.process_runner import Limits, run_job


def test_child_success_and_output(tmp_path):
    result = run_job([sys.executable, "-c", "print('ok')"], tmp_path, tmp_path / "job",
                     Limits(10, 1024**3, 1))
    assert result["exit_code"] == 0
    assert result["termination_reason"] is None
    assert (tmp_path / "job/stdout.log").read_text().strip() == "ok"
    assert json.loads((tmp_path / "job/process_result.json").read_text())["exit_code"] == 0


def test_timeout_terminates_owned_child(tmp_path):
    result = run_job([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path,
                     tmp_path / "timeout", Limits(.3, 1024**3, 1, .05))
    assert result["termination_reason"] == "wall_limit"
    assert result["exit_code"] is not None
    assert result["wall_seconds"] < 10


def test_memory_limit_and_invalid_limits(tmp_path):
    result = run_job([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path,
                     tmp_path / "memory", Limits(10, 1, 1, .05))
    assert result["termination_reason"] == "memory_limit"
    assert result["exit_code"] is not None
    with pytest.raises(ValueError):
        Limits(0, 1, 1)
