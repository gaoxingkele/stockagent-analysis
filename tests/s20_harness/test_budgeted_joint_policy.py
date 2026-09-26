import pytest
from research.s20_harness.budgeted_joint_policy import binding,run,verify
from research.s20_harness.trial_budget import Budget
from research.s20_harness.runtime import digest
from tests.s20_harness.test_joint_policy_run import setup


def prepared(tmp_path,limit=2):
    directory,source=setup(tmp_path);run_sha=digest(directory/'summary.json');input_sha=digest(source)
    _,sha,costs=binding(directory,run_sha,source,input_sha)
    contract=dict(budget_id='shared-policies',limits=dict(costs,policy_evaluations=limit),trials=[
        dict(trial_id=t,input_sha256=sha,costs=costs,max_attempts=1) for t in ['a','b']])
    return directory,source,run_sha,input_sha,Budget(tmp_path/'shared.sqlite',contract)


def test_reserve_before_comparison_reuse_and_shared_cap(tmp_path,monkeypatch):
    d,s,ds,ss,b=prepared(tmp_path)
    from research.s20_harness import joint_policy_run
    original=joint_policy_run.build
    def guarded(*args):
        assert b.status()['reserved_counts']['policy_evaluations']==2
        assert b.status()['attempts'][0]['state']=='RESERVED'
        return original(*args)
    monkeypatch.setattr(joint_policy_run,'build',guarded)
    result=run(tmp_path,d,ds,s,ss,b,'a','1')
    assert result['executed'] and result['verification']['policy_budget_and_result_verified']
    before=digest(b.path)
    assert verify(tmp_path,b.path,b.sha,'a','1')['charged_costs']['model_fits']==0
    assert digest(b.path)==before
    monkeypatch.setattr(joint_policy_run,'build',lambda *args:pytest.fail('duplicate comparison execution'))
    assert not run(tmp_path,d,ds,s,ss,b,'a','1')['executed']
    with pytest.raises(ValueError,match='exhausted'):run(tmp_path,d,ds,s,ss,b,'b','1')


def test_failure_retains_charge_no_automatic_retry(tmp_path,monkeypatch):
    d,s,ds,ss,b=prepared(tmp_path)
    from research.s20_harness import joint_policy_run
    def fail(*args):raise RuntimeError('injected')
    monkeypatch.setattr(joint_policy_run,'build',fail)
    with pytest.raises(RuntimeError,match='injected'):run(tmp_path,d,ds,s,ss,b,'a','1')
    assert b.status()['attempts'][0]['state']=='FAILED'
    assert b.status()['reserved_counts']['policy_evaluations']==2
    with pytest.raises(ValueError,match='reconciliation'):run(tmp_path,d,ds,s,ss,b,'a','1')
    with pytest.raises(ValueError,match='exhausted'):run(tmp_path,d,ds,s,ss,b,'b','1')


def test_cli_existing_budget_execute_verify_resume_and_invalid_pin(tmp_path,monkeypatch,capsys):
    import json
    from research.s20_harness import cli
    monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    d,s,ds,ss,b=prepared(tmp_path)
    budget_args=['--budget-path',str(b.path),'--contract-sha256',b.sha,'--trial-id','a','--attempt-id','1']
    run_args=['run-budgeted-joint-policy',*budget_args,'--run-directory',str(d),'--run-sha256',ds,
              '--input',str(s),'--input-sha256',ss]
    assert cli.main(run_args)==0
    assert json.loads(capsys.readouterr().out)['executed']
    before=digest(b.path)
    assert cli.main(['verify-budgeted-joint-policy',*budget_args])==0
    assert json.loads(capsys.readouterr().out)['policy_budget_and_result_verified']
    assert digest(b.path)==before
    assert cli.main(run_args)==0
    assert not json.loads(capsys.readouterr().out)['executed']
    bad=list(run_args);bad[bad.index('--contract-sha256')+1]='0'*64
    assert cli.main(bad)==2
    assert not json.loads(capsys.readouterr().out)['valid']
    assert b.status()['reserved_counts']['policy_evaluations']==2


def test_missing_budget_not_created(tmp_path):
    from research.s20_harness.budgeted_joint_policy import run_existing
    missing=tmp_path/'not-created.sqlite'
    with pytest.raises(ValueError,match='existing workspace'):
        run_existing(tmp_path,tmp_path,'0'*64,tmp_path/'input.json','0'*64,missing,'0'*64,'a','1')
    assert not missing.exists()


def test_same_model_raw_and_calibrated_policy_paths_charged_separately(tmp_path):
    from pathlib import Path
    import pandas as pd
    from research.s20_harness.runtime import atomic_json,load_plan
    d,cal_source=setup(tmp_path);ds=digest(d/'summary.json')
    payload=load_plan(cal_source);payload['probability_source']='raw'
    raw_source=tmp_path/'raw-policies.json';atomic_json(raw_source,payload)
    trials=[]
    for label,source in [('raw',raw_source),('calibrated',cal_source)]:
        _,sha,costs=binding(d,ds,source,digest(source))
        trials.append(dict(trial_id=label,input_sha256=sha,costs=costs,max_attempts=1))
    assert trials[0]['input_sha256']!=trials[1]['input_sha256']
    b=Budget(tmp_path/'both.sqlite',dict(budget_id='both',limits=dict(costs,policy_evaluations=4),trials=trials))
    before={p.name:digest(p) for p in d.iterdir() if p.is_file()}
    predictions=pd.read_parquet(d/'calibrated_predictions.parquet').set_index('sample_id')
    outputs={}
    for label,source in [('raw',raw_source),('calibrated',cal_source)]:
        result=run(tmp_path,d,ds,source,digest(source),b,label,'1')
        assert result['executed']
        artifact=result['budget']['attempts'][-1]['result'];out=Path(artifact['directory'])
        outputs[label]=out
        comparison=load_plan(out/'comparison.json')
        assert comparison['probability_source']==label
        assert comparison['fixed_selection_calibration']['selection_reference']=='saved_'+label+'_policy_selection'
        rows=pd.read_parquet(out/'policy_00.parquet')
        assert rows.p_A.tolist()==predictions.loc[rows.sample_id,('cal_' if label=='calibrated' else '')+'p_A'].tolist()
    assert b.status()['reserved_counts']==dict(model_fits=0,underlying_fits=0,calibrator_fits=0,policy_evaluations=4)
    assert before=={p.name:digest(p) for p in d.iterdir() if p.is_file()}
    from research.s20_harness.calibration_policy_contrast import inspect
    budget_before=digest(b.path)
    contrast=inspect(tmp_path,outputs['raw'],digest(outputs['raw']/'summary.json'),
                     outputs['calibrated'],digest(outputs['calibrated']/'summary.json'))
    assert contrast['same_model_and_evaluation_inputs_verified']
    assert len(contrast['policies'])==2 and not contrast['automatic_winner_selected']
    assert digest(b.path)==budget_before
