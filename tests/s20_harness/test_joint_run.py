from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.joint_run import build,verify
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_joint_model import inputs
from tests.s20_harness.test_joint_policy import fixture


def plan():
    samples,features,labels,bounds,contract=inputs()
    # Non-saturated forward probabilities make parameter/calibration mutations
    # observable; the feature-pipeline fixture deliberately uses extreme values.
    forward=~features.sample_id.str.startswith('fit')
    features.loc[forward,'x']=[.3+i*.2 for i in range(int(forward.sum()))]
    samples['entity_id']=samples.sample_id
    samples['signal_date']=pd.to_datetime(samples.prediction_at,utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y%m%d')
    samples.loc[9,'feature_available_at']=None
    _,policy,_=fixture();policy['selection']['frozen_at']='2024-05-01T00:00:00Z'
    return dict(schema_version='joint-1',evidence_mode='synthetic',target_id='P.joint.v4',
        samples=samples.to_dict('records'),features=features.to_dict('records'),fit_labels=labels.to_dict('records'),
        calibration_labels=[dict(sample_id='calibration0',target='A'),dict(sample_id='calibration1',target='D')],
        boundaries=bounds,feature_contract=contract,policy=policy,calendar=['20240502','20240503'])


@pytest.mark.parametrize('family', ['multinomial','conditional_three','mature_frequency','cost_sensitive_joint','shallow_joint'])
def test_explicit_seed_saved_and_replayed(tmp_path, family):
    from research.s20_harness.joint_replay import replay
    value=plan();value.update(model_family=family,random_seed=71)
    if family=='cost_sensitive_joint':value['class_weights']={'A':1.,'B':2.,'C':1.,'D':3.}
    path=tmp_path/'seed.json';atomic_json(path,value)
    report=build(tmp_path,path,digest(path));out=Path(report['directory'])
    card=load_plan(out/'joint_card.json')
    assert card['randomness']['requested_seed']==71
    assert card['randomness']['independent_market_evidence'] is False
    if family!='mature_frequency':assert card['parameters']['random_state']==71
    assert replay(out,digest(out/'summary.json'))['model_fits']==0


@pytest.mark.parametrize('seed', [True,-1,2**32,1.5,'20',None])
def test_joint_invalid_seed_rejected(tmp_path,seed):
    from research.s20_harness.joint_run import pipeline_costs
    value=plan();value['random_seed']=seed
    path=tmp_path/'seed.json';atomic_json(path,value)
    with pytest.raises(ValueError,match='random_seed'):build(tmp_path,path,digest(path))
    with pytest.raises(ValueError,match='random_seed'):pipeline_costs(value)


def test_saved_joint_chain_retains_missing_and_checks_artifacts(tmp_path):
    path=tmp_path/'joint.json';atomic_json(path,plan())
    report=build(tmp_path,path,digest(path));out=Path(report['directory'])
    assert verify(out,digest(out/'summary.json'))['artifact_bytes_verified']
    rows=pd.read_parquet(out/'candidate_ledger.parquet')
    assert len(rows)==2 and rows.p_A.isna().sum()==1 and 'utility_rank' in rows
    assert report['underlying_fits']==2 and not report['absolute_probability_validated']
    from research.s20_harness.joint_replay import replay
    assert replay(out,digest(out/'summary.json'))['model_fits']==0
    original=load_plan(out/'joint_card.json')
    original['coefficients'][0][0]+=.5
    atomic_json(out/'joint_card.json',original)
    cp=load_plan(out/'checkpoint.json')
    cp['artifacts']['joint_card.json']=digest(out/'joint_card.json')
    atomic_json(out/'checkpoint.json',cp)
    report['artifacts']['joint_card.json']=digest(out/'joint_card.json')
    report['artifacts']['checkpoint.json']=digest(out/'checkpoint.json')
    atomic_json(out/'summary.json',report)
    assert verify(out,digest(out/'summary.json'))['artifact_bytes_verified']
    with pytest.raises(AssertionError):replay(out,digest(out/'summary.json'))
    atomic_json(out/'joint_card.json',{})
    with pytest.raises(ValueError,match='artifact pin'):verify(out,digest(out/'summary.json'))


def test_calibration_failure_retains_completed_model(tmp_path):
    value=plan();value['calibration_labels'][1]['target']='A'
    path=tmp_path/'joint.json';atomic_json(path,value)
    with pytest.raises(ValueError,match='diverse'):build(tmp_path,path,digest(path))
    out=next((tmp_path/'output/experiments/s20_safe_v4/sources').iterdir())
    cp=load_plan(out/'checkpoint.json')
    assert cp['status']=='FAILED' and cp['completed_steps']==['joint_model']
    assert (out/'joint_card.json').exists() and not (out/'summary.json').exists()


def test_direct_probability_policy_saved_and_replayed(tmp_path):
    from research.s20_harness.joint_replay import replay
    value=plan();value['policy']=dict(selection=value['policy']['selection'],ranking='safe_probability')
    path=tmp_path/'direct.json';atomic_json(path,value)
    report=build(tmp_path,path,digest(path));out=Path(report['directory'])
    assert report['policy_evaluations']==1
    assert pd.read_parquet(out/'candidate_ledger.parquet').utility_rank.isna().all()
    assert replay(out,digest(out/'summary.json'))['model_fits']==0


@pytest.mark.parametrize('family,underlying',[('multinomial',1),('conditional_three',3),('mature_frequency',0)])
def test_uncalibrated_control_preserves_raw_and_actual_costs(tmp_path,family,underlying):
    from research.s20_harness.joint_replay import replay
    value=plan();value.update(model_family=family,calibration_method='identity_raw',calibration_labels=[])
    path=tmp_path/'raw.json';atomic_json(path,value)
    report=build(tmp_path,path,digest(path));out=Path(report['directory'])
    assert report['underlying_fits']==underlying and report['calibrator_fits']==0
    assert replay(out,digest(out/'summary.json'))['saved_inference_recomputed']
    predictions=pd.read_parquet(out/'calibrated_predictions.parquet')
    later=predictions.segment.isin(['selection-policy','outer-test'])
    for name in ['p_A','p_B','p_C','p_D']:
        pd.testing.assert_series_equal(predictions.loc[later,name],predictions.loc[later,'cal_'+name],check_names=False)


@pytest.mark.parametrize('artifact',['calibration_card.json','candidate_ledger.parquet'])
def test_semantic_replay_rejects_rehashed_inconsistent_outputs(tmp_path,artifact):
    from research.s20_harness.joint_replay import replay
    path=tmp_path/'joint.json';atomic_json(path,plan())
    report=build(tmp_path,path,digest(path));out=Path(report['directory'])
    if artifact.endswith('.json'):
        value=load_plan(out/artifact)
        value['temperature']=2. if value['temperature']<1.5 else .5
        atomic_json(out/artifact,value)
    else:
        value=pd.read_parquet(out/artifact)
        value.loc[value.p_A.notna(),'utility_rank']+=.1
        value.to_parquet(out/artifact,index=False)
    cp=load_plan(out/'checkpoint.json');cp['artifacts'][artifact]=digest(out/artifact)
    atomic_json(out/'checkpoint.json',cp)
    report['artifacts'][artifact]=digest(out/artifact)
    report['artifacts']['checkpoint.json']=digest(out/'checkpoint.json')
    atomic_json(out/'summary.json',report)
    assert verify(out,digest(out/'summary.json'))['artifact_bytes_verified']
    with pytest.raises(AssertionError):replay(out,digest(out/'summary.json'))
