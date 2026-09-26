from pathlib import Path
import pytest
from research.s20_harness.bounded_joint import run
from research.s20_harness.joint_replay import replay
from research.s20_harness.runtime import atomic_json,digest,load_plan
from research.s20_harness.trial_budget import Budget,verify_attempt
from research.s20_harness.process_runner import Limits
from tests.s20_harness.test_joint_binary_scope import joint_plan


def test_conditional_owned_chain_four_underlying_fits_and_reuse(tmp_path):
    value=joint_plan();value['model_family']='conditional_three'
    source=tmp_path/'conditional.json';atomic_json(source,value);sha=digest(source)
    def budget(path,underlying):
        costs=dict(model_fits=1,underlying_fits=underlying,calibrator_fits=1,policy_evaluations=1)
        return Budget(path,dict(budget_id='conditional',limits=costs,
            trials=[dict(trial_id='conditional',input_sha256=sha,costs=costs,max_attempts=1)]))
    wrong=budget(tmp_path/'wrong.sqlite',2);limits=Limits(120,2*1024**3,1)
    with pytest.raises(ValueError,match='cost/input'):run(tmp_path,source,sha,wrong,'conditional','one',limits)
    assert wrong.status()['attempts']==[]
    correct=budget(tmp_path/'correct.sqlite',4)
    result=run(tmp_path,source,sha,correct,'conditional','one',limits)
    artifact=result['artifact'];out=Path(artifact['directory'])
    assert load_plan(out/'summary.json')['underlying_fits']==4
    assert load_plan(out/'joint_card.json')['underlying_fits']==3
    assert replay(out,digest(out/'summary.json'))['model_fits']==0
    checked=verify_attempt(tmp_path,correct.path,correct.sha,'conditional','one',source,sha,artifact,limits)
    assert checked['charged_costs']['underlying_fits']==4
    reused=run(tmp_path,source,sha,correct,'conditional','one',limits)
    assert reused['reusable'] and not reused['executed']
    # Consistently rehashed artifact mutation must still fail semantic replay.
    card=load_plan(out/'joint_card.json');card['heads']['down_if_up']['fit_sample_ids']=[]
    atomic_json(out/'joint_card.json',card)
    cp=load_plan(out/'checkpoint.json');cp['artifacts']['joint_card.json']=digest(out/'joint_card.json')
    atomic_json(out/'checkpoint.json',cp)
    report=load_plan(out/'summary.json');report['artifacts']['joint_card.json']=digest(out/'joint_card.json')
    report['artifacts']['checkpoint.json']=digest(out/'checkpoint.json');atomic_json(out/'summary.json',report)
    with pytest.raises(ValueError,match='head membership'):replay(out,digest(out/'summary.json'))
