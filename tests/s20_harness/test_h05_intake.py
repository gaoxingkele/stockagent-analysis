from pathlib import Path
import pytest
from research.s20_harness.h05_intake import audit,OUTPUTS
from research.s20_harness.budgeted_joint_policy import run
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_budgeted_joint_policy import prepared


def test_budgeted_outputs_copied_without_formal_acceptance(tmp_path):
    d,s,ds,ss,b=prepared(tmp_path)
    completed=run(tmp_path,d,ds,s,ss,b,'a','1')
    source=Path(completed['verification']['directory'])
    binding=tmp_path/'config/s20_v4_h05_intake.json'
    binding.parent.mkdir(exist_ok=True)
    atomic_json(binding,dict(protocol_hash='frozen',budget_path=str(b.path),contract_sha256=b.sha,trial_id='a',attempt_id='1'))
    before=digest(b.path);out=tmp_path/'intake'
    result=audit(tmp_path,out,'frozen')
    assert not result['formal_gate_passed'] and len(result['acceptance_gaps'])>=5
    assert result['budget_validation']['policy_budget_and_result_verified']
    assert all(digest(out/n)==digest(source/n) for n in OUTPUTS)
    assert digest(b.path)==before
    with pytest.raises(ValueError,match='existing outputs'):audit(tmp_path,out,'frozen')
    with pytest.raises(ValueError,match='protocol'):audit(tmp_path,tmp_path/'other','changed')


def test_missing_binding_never_creates_acceptance(tmp_path):
    with pytest.raises(ValueError,match='missing pinned'):audit(tmp_path,tmp_path/'intake','frozen')
    assert not (tmp_path/'intake').exists()
