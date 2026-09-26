from pathlib import Path
import numpy as np
import pytest
from research.s20_harness.mature_frequency import run
from research.s20_harness.joint_run import pipeline_costs,build
from research.s20_harness.joint_replay import replay
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_joint_model import inputs
from tests.s20_harness.test_joint_binary_scope import joint_plan


def test_frequency_only_mature_fit_labels_and_matching_availability():
    args=inputs();args[0].loc[9,'feature_available_at']=None
    args[2].loc[args[2].target.eq('D'),'target']='A'
    out,card=run(*args,target_id='P.joint.v4')
    assert card['probabilities']==[.5,.25,.25,0.]
    assert len(out)==20 and out.p_A.notna().sum()==7
    np.testing.assert_allclose(out.loc[out.p_A.notna(),'p_A'],.5)
    assert card['model_level_fits']==card['underlying_fits']==0
    args[1]['x']=123.
    _,changed=run(*args,target_id='P.joint.v4')
    assert changed['probabilities']==card['probabilities']
    args[2].loc[0,'sample_id']='outer-test0'
    with pytest.raises(ValueError,match='eligible'):run(*args,target_id='P.joint.v4')


def test_saved_frequency_calibration_replay_and_costs(tmp_path):
    value=joint_plan();value['model_family']='mature_frequency'
    source=tmp_path/'frequency.json';atomic_json(source,value)
    from research.s20_harness.bounded_joint import run as bounded
    from research.s20_harness.trial_budget import Budget,verify_attempt
    from research.s20_harness.process_runner import Limits
    from research.s20_harness.runtime import load_plan
    costs=pipeline_costs(value);pin=digest(source);limits=Limits(120,2*1024**3,1)
    budget=Budget(tmp_path/'budget.sqlite',dict(budget_id='frequency',limits=costs,
        trials=[dict(trial_id='frequency',input_sha256=pin,costs=costs,max_attempts=1)]))
    result=bounded(tmp_path,source,pin,budget,'frequency','one',limits)
    out=Path(result['artifact']['directory']);report=load_plan(out/'summary.json')
    assert report['model_level_fits']==0 and report['underlying_fits']==1
    assert pipeline_costs(value)==dict(model_fits=0,underlying_fits=1,calibrator_fits=1,policy_evaluations=1)
    assert replay(out,digest(out/'summary.json'))['saved_inference_recomputed']
    checked=verify_attempt(tmp_path,budget.path,budget.sha,'frequency','one',source,pin,result['artifact'],limits)
    assert checked['charged_costs']['model_fits']==0
