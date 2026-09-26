from pathlib import Path

from research.s20_harness.cli import main
from research.s20_harness.contracts import load_plan, validate_plan

PLAN = Path(__file__).resolve().parents[2] / "config" / "s20_v4_harness_plan.json"


def test_validate_plan_reports_caps_abstain_and_no_training(capsys):
    code = main(["validate-plan", "--plan", str(PLAN)])
    text = capsys.readouterr().out
    assert code == 0
    assert "DAG is valid" in text
    assert "TopN caps: 1/3/5/10/20" in text
    assert "abstain allowed" in text
    assert "production actions disabled" in text
    assert "real orders disabled" in text
    assert "H03+ training is not started" in text
    result = validate_plan(load_plan(PLAN))
    assert result["valid"] is True
    assert result["training_started"] is False


def test_run_stage_h03_is_refused_until_foundation_passes(capsys):
    code = main(["run-stage", "--plan", str(PLAN), "--stage", "H03"])
    text = capsys.readouterr().out
    assert code == 2
    assert "H03+ training is not started" in text
    assert "H00-H02" in text
