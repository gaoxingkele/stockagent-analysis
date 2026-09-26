from pathlib import Path
import pytest
from research.s20_harness.runtime import atomic_json,digest,load_plan
from research.s20_harness.trial_budget import Budget
from research.s20_harness.process_runner import Limits
from research.s20_harness.bounded_joint import run as joint_run
from research.s20_harness.bounded_baseline import run as binary_run
from research.s20_harness.joint_binary_scope import project
from research.s20_harness.joint_binary_comparison import run_registered,verify
from research.s20_harness.comparison_parent_binding import audit,preflight
from tests.s20_harness.test_joint_binary_scope import joint_plan


@pytest.mark.parametrize('shared',[False,True])
def test_parent_training_bindings_survive_ledger_growth(tmp_path,shared):
    joint=joint_plan();refs={};models={};limits=Limits(120,2*1024**3,1)
    costs=dict(model_fits=1,underlying_fits=2,calibrator_fits=1,policy_evaluations=1)
    budgets=[]
    prepared=[]
    for name,plan,runner in [('joint',joint,joint_run),('binary',project(joint),binary_run)]:
        source=tmp_path/(name+'.json');atomic_json(source,plan);pin=digest(source)
        prepared.append((name,source,pin,runner))
    common=Budget(tmp_path/'shared.sqlite',dict(budget_id='shared',limits={k:v*4 for k,v in costs.items()},
        trials=[dict(trial_id=name,input_sha256=pin,costs=costs,max_attempts=2) for name,_,pin,_ in prepared])) if shared else None
    for name,source,pin,runner in prepared:
        budget=common or Budget(tmp_path/(name+'.sqlite'),dict(budget_id=name,limits={k:v*2 for k,v in costs.items()},
            trials=[dict(trial_id=name,input_sha256=pin,costs=costs,max_attempts=2)]))
        result=runner(tmp_path,source,pin,budget,name,'one',limits)
        artifact=result['artifact'];models[name]={k:artifact[k] for k in ['directory','summary_sha256']}
        refs[name]=dict(path=str(budget.path),contract_sha256=budget.sha,trial_id=name,attempt_id='one',
            input_path=str(source),input_sha256=pin,artifact=artifact,limits=limits.__dict__)
        budgets.append((budget,name,pin))
    source=tmp_path/'comparison.json'
    plan=dict(schema_version='2',**models,target_id='P.joint.v4',evaluation_at='2024-06-01T00:00:00Z',
        k=1,outcomes=[],parent_budget_references=refs)
    atomic_json(source,plan);pin=digest(source)
    policy_costs=dict(model_fits=0,underlying_fits=0,calibrator_fits=0,policy_evaluations=2)
    budget=Budget(tmp_path/'comparison.sqlite',dict(budget_id='comparison',limits=policy_costs,
        trials=[dict(trial_id='compare',input_sha256=pin,costs=policy_costs,max_attempts=1)]))
    first=run_registered(tmp_path,source,pin,budget,'compare','one')
    assert first['parent_training_budget_verified']
    inventory=first['parent_budget_inventory']
    assert len(inventory['referenced_ledgers'])==(1 if shared else 2)
    assert inventory['combined_capacity']==dict(model_fits=2,underlying_fits=4,calibrator_fits=2,policy_evaluations=4)
    excessive=Budget(tmp_path/'excess.sqlite',dict(budget_id='excess',limits=dict(policy_costs,model_fits=71),
        trials=[dict(trial_id='compare',input_sha256=pin,costs=policy_costs,max_attempts=1)]))
    with pytest.raises(ValueError,match='exceeds 72'):
        run_registered(tmp_path,source,pin,excessive,'compare','one')
    assert excessive.status()['attempts']==[]
    for parent,name,sha in budgets:
        parent.reserve(name,'later',sha);parent.finish(name,'later','FAILED',{'test':'retained'})
    second=run_registered(tmp_path,source,pin,budget,'compare','one')
    assert second['reusable'] and second['parent_training_budget_verified']
    assert second['parent_budget_inventory']['combined_capacity']['model_fits']==4
    if shared:
        assert preflight(tmp_path,plan,common)['combined_capacity']['model_fits']==4
    out=Path(first['artifact']['directory'])
    assert verify(tmp_path,out,digest(out/'summary.json'))['rankings_and_metrics_recomputed']
    refs['joint']['attempt_id']='later'
    with pytest.raises(ValueError,match='completion/artifact'):audit(tmp_path,plan)
    refs['joint']['attempt_id']='one'
    refs['joint']['artifact']['summary_sha256']='0'*64
    with pytest.raises(ValueError,match='another model'):audit(tmp_path,plan)
