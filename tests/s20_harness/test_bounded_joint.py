from pathlib import Path
import pytest
from research.s20_harness.bounded_joint import run, verify_process
from research.s20_harness.process_runner import Limits
from research.s20_harness.trial_budget import Budget, verify_attempt
from research.s20_harness.runtime import atomic_json, digest, load_plan
from tests.s20_harness.test_joint_run import plan
from tests.s20_harness.test_trial_budget import contract


def setup(tmp_path,family='multinomial'):
    source=tmp_path/'joint.json';value=plan();value['model_family']=family
    atomic_json(source,value);pin=digest(source)
    return source,pin,Budget(tmp_path/'budget.sqlite',contract(pin))


@pytest.mark.parametrize('family',['multinomial','shallow_joint'])
def test_joint_owned_process_budget_identity_and_reuse(tmp_path,family):
    source,pin,budget=setup(tmp_path,family);limits=Limits(120,2*1024**3,1)
    result=run(tmp_path,source,pin,budget,'baseline','one',limits)
    artifact=result['artifact']
    assert result['executed'] and result['process']['exit_code']==0
    assert artifact['pipeline_kind']=='joint'
    assert verify_process(tmp_path,artifact,source,pin,limits)['recorded_completion_verified']
    checked=verify_attempt(tmp_path,budget.path,budget.sha,'baseline','one',source,pin,artifact,limits)
    assert checked['recorded_prelaunch_reservation_verified']
    assert checked['charged_costs']==dict(model_fits=1,underlying_fits=2,calibrator_fits=1,policy_evaluations=1)
    reused=run(tmp_path,source,pin,budget,'baseline','one',limits)
    assert not reused['executed'] and reused['reusable']
    from research.s20_harness.bounded_baseline import verify_process as verify_binary
    with pytest.raises(ValueError,match='pipeline identity'):
        verify_binary(tmp_path,artifact,source,pin,limits)
    with pytest.raises(ValueError,match='limits differ'):
        run(tmp_path,source,pin,budget,'baseline','one',Limits(119,2*1024**3,1))
    atomic_json(Path(artifact['directory'])/'joint_card.json',{})
    with pytest.raises(ValueError,match='artifact pin'):
        run(tmp_path,source,pin,budget,'baseline','one',limits)
    assert budget.status()['reserved_counts']['model_fits']==1


def test_joint_timeout_retains_charge_without_duplicate_launch(tmp_path):
    source,pin,budget=setup(tmp_path);limits=Limits(.001,2*1024**3,1,.001)
    with pytest.raises(RuntimeError,match='wall_limit'):
        run(tmp_path,source,pin,budget,'baseline','one',limits)
    assert budget.status()['attempts'][0]['state']=='FAILED'
    again=run(tmp_path,source,pin,budget,'baseline','one',limits)
    assert not again['executed'] and not again['reusable']
    assert again['budget']['reserved_counts']['model_fits']==1


def test_joint_calibration_failure_keeps_partial_artifacts_and_charge(tmp_path):
    source=tmp_path/'joint.json';value=plan()
    value['calibration_labels'][1]['target']='A'
    atomic_json(source,value);pin=digest(source)
    budget=Budget(tmp_path/'budget.sqlite',contract(pin))
    with pytest.raises(RuntimeError,match='process failed'):
        run(tmp_path,source,pin,budget,'baseline','one',Limits(120,2*1024**3,1))
    attempt=budget.status()['attempts'][0]
    assert attempt['state']=='FAILED'
    process=Path(attempt['result']['process_directory'])
    assert load_plan(process/'process_result.json')['exit_code']!=0
    out=next((tmp_path/'output/experiments/s20_safe_v4/sources').glob('joint-run-*'))
    checkpoint=load_plan(out/'checkpoint.json')
    assert checkpoint['status']=='FAILED' and checkpoint['completed_steps']==['joint_model']
    assert (out/'joint_card.json').exists() and not (out/'summary.json').exists()
    assert budget.status()['reserved_counts']['model_fits']==1
