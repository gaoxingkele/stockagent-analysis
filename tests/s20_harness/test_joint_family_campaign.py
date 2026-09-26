from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.multifold_run import run
from research.s20_harness.campaign_evaluation import build as evaluate,verify
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_joint_multifold import setup as two_folds


def setup(tmp_path,control_family='conditional_three'):
    source=two_folds(tmp_path);manifest=load_plan(source)
    common=load_plan(Path(manifest['jobs'][0]['input_path']))
    for i,job in enumerate(manifest['jobs']):
        value=dict(common,model_family='multinomial' if i==0 else control_family)
        path=Path(job['input_path']);atomic_json(path,value)
        job['fold_id']='common';job['input_sha256']=digest(path)
        trial=manifest['budget']['trials'][i];trial['input_sha256']=digest(path)
        from research.s20_harness.joint_run import pipeline_costs
        trial['costs']=pipeline_costs(value)
    manifest['budget']['limits']={k:sum(t['costs'][k] for t in manifest['budget']['trials'])
                                  for k in manifest['budget']['limits']}
    atomic_json(source,manifest)
    return source


@pytest.mark.parametrize('control_family',['conditional_three','mature_frequency'])
def test_same_fold_joint_family_competition_evaluates_and_reuses(tmp_path,control_family):
    source=setup(tmp_path,control_family);result=run(tmp_path,source,digest(source))
    assert result['distinct_outer_folds']==1
    is_frequency=control_family=='mature_frequency'
    assert result['budget']['reserved_counts']==dict(model_fits=1 if is_frequency else 2,
        underlying_fits=3 if is_frequency else 6,calibrator_fits=2,policy_evaluations=2)
    rows=pd.read_parquet(result['outer_prediction_ledger']['path'])
    assert len(rows)==4 and rows.p_A.isna().sum()==2
    assert set(rows.model_family)=={'fixed_multinomial_logistic',
                                  'mature_empirical_frequency' if is_frequency else 'fixed_conditional_three_logistic'}
    if is_frequency:
        frequency=rows.loc[rows.model_family.eq('mature_empirical_frequency')&rows.raw_p_A.notna()]
        assert frequency.raw_p_A.eq(.25).all() and len(frequency)==1
    ids=[rows.loc[rows.job_id.eq(j),'sample_id'].tolist() for j in rows.job_id.unique()]
    assert ids[0]==ids[1]
    reused=run(tmp_path,source,digest(source),directory=result['directory'])
    assert not any(j['executed_this_call'] for j in reused['jobs'])
    receipt=next(Path(result['directory']).glob('receipt-*.json'))
    outcome=tmp_path/'outcome.json'
    atomic_json(outcome,dict(target_id='P.joint.v4',evaluation_at='2024-06-01T00:00:00Z',outcomes=[
        dict(sample_id=sid,target=c,label_available_at='2024-05-29T00:00:00Z') for sid,c in zip(ids[0],['A','D'])]))
    report=evaluate(tmp_path,receipt,digest(receipt),[dict(fold_id='common',path=str(outcome),sha256=digest(outcome))])
    out=Path(report['directory'])
    assert verify(tmp_path,out,digest(out/'summary.json'))['metrics_recomputed']
    assert report['outer_folds']==1 and report['rows']==4 and not report['model_rows_are_independent_observations']


@pytest.mark.parametrize('bad',['cost','policy','features'])
def test_joint_family_campaign_invalid_second_job_rejected_prelaunch(tmp_path,bad):
    source=setup(tmp_path);manifest=load_plan(source)
    if bad=='cost':manifest['budget']['trials'][1]['costs']['underlying_fits']=2
    else:
        job=manifest['jobs'][1];path=Path(job['input_path']);plan=load_plan(path)
        if bad=='policy':plan['policy']['selection']['n_cap']+=1
        else:plan['features'][0]['x']+=.1
        atomic_json(path,plan);job['input_sha256']=digest(path)
        manifest['budget']['trials'][1]['input_sha256']=digest(path)
    atomic_json(source,manifest)
    with pytest.raises(ValueError,match='costs differ|comparison scope mismatch'):run(tmp_path,source,digest(source))
    assert not (tmp_path/'output').exists()
