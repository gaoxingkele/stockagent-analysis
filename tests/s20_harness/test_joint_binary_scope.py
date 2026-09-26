import json
from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.joint_binary_scope import project,validate,bind
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_joint_run import plan


def joint_plan():
    return json.loads(json.dumps(plan()),parse_constant=lambda _:None)


def test_projection_keeps_dangerous_upside_negative_and_input_unchanged():
    joint=joint_plan();before=json.dumps(joint,sort_keys=True)
    binary=project(joint)
    assert json.dumps(joint,sort_keys=True)==before
    assert [r['target'] for r in binary['fit_labels']]==[r['target']=='A' for r in joint['fit_labels']]
    assert sum(r['target'] for r in binary['fit_labels'])==3
    report=validate(joint,binary)
    assert report['same_samples_features_splits'] and not report['pure_model_effect_identified']


@pytest.mark.parametrize('field',['fit_labels','calibration_labels','features','samples','boundaries','policy'])
def test_scope_changes_rejected(field):
    joint=joint_plan();binary=project(joint)
    if field in ['fit_labels','calibration_labels']:binary[field][0]['target']=not binary[field][0]['target']
    elif field=='features':binary[field][0]['x']+=.1
    elif field=='samples':binary[field]=binary[field][:-1]
    elif field=='boundaries':binary[field][0]['start_at']='2023-01-01T00:00:00Z'
    else:binary[field]['n_cap']+=1
    with pytest.raises(ValueError,match='scope or A-label'):validate(joint,binary)


def test_actual_saved_binary_joint_share_consumed_membership(tmp_path):
    from research.s20_harness.joint_run import build as joint_train
    from research.s20_harness.baseline_run import build as binary_train
    joint=joint_plan();binary=project(joint)
    jp=tmp_path/'joint.json';bp=tmp_path/'binary.json'
    atomic_json(jp,joint);atomic_json(bp,binary)
    jr=joint_train(tmp_path,jp,digest(jp));br=binary_train(tmp_path,bp,digest(bp))
    jd=Path(jr['directory']);bd=Path(br['directory'])
    result=bind(jd,digest(jd/'summary.json'),bd,digest(bd/'summary.json'))
    assert result['consumed_membership_verified'] and result['models_refit']==0
    left=pd.read_parquet(jd/'candidate_ledger.parquet');right=pd.read_parquet(bd/'candidate_ledger.parquet')
    assert left.sample_id.tolist()==right.sample_id.tolist()
    assert left.p_A.isna().tolist()==right.score.isna().tolist()
    from research.s20_harness.joint_binary_scope import compare_ranking
    outcomes=pd.DataFrame(dict(sample_id=left.sample_id.tolist(),target=['A','D'],
        label_available_at=['2024-05-29T00:00:00Z']*2))
    rows,report=compare_ranking(jd,digest(jd/'summary.json'),bd,digest(bd/'summary.json'),outcomes,
        evaluation_at='2024-06-01T00:00:00Z',k=1)
    assert len(rows)==4 and report['same_day_same_count']
    assert report['daily_selected_counts']=={'20240503':1}
    assert report['scored_shortfall_days']==['20240502']
    assert not report['original_policies_evaluated'] and report['models_refit']==0
    assert report['metrics']['binary_safe']['selected_candidates']['event_bounds']['rate_lower']==1
    outcomes['target']=['D','A']
    changed,_=compare_ranking(jd,digest(jd/'summary.json'),bd,digest(bd/'summary.json'),outcomes,
        evaluation_at='2024-06-01T00:00:00Z',k=1)
    assert changed.selected.tolist()==rows.selected.tolist()
    from research.s20_harness.joint_binary_comparison import build as save_comparison,verify as verify_comparison
    from research.s20_harness.runtime import load_plan
    source=tmp_path/'comparison.json'
    atomic_json(source,dict(schema_version='1',target_id='P.joint.v4',
        joint=dict(directory=str(jd),summary_sha256=digest(jd/'summary.json')),
        binary=dict(directory=str(bd),summary_sha256=digest(bd/'summary.json')),
        evaluation_at='2024-06-01T00:00:00Z',k=1,outcomes=outcomes.to_dict('records')))
    saved=save_comparison(tmp_path,source,digest(source));out=Path(saved['directory'])
    assert verify_comparison(tmp_path,out,digest(out/'summary.json'))['rankings_and_metrics_recomputed']
    from research.s20_harness.joint_binary_comparison import run_registered
    from research.s20_harness.trial_budget import Budget
    costs=dict(model_fits=0,underlying_fits=0,calibrator_fits=0,policy_evaluations=2)
    budget=Budget(tmp_path/'comparison-budget.sqlite',dict(budget_id='comparison',limits=costs,
        trials=[dict(trial_id='compare',input_sha256=digest(source),costs=costs,max_attempts=2)]))
    registered=run_registered(tmp_path,source,digest(source),budget,'compare','one')
    assert registered['executed'] and registered['local_comparison_reserved']
    again=run_registered(tmp_path,source,digest(source),budget,'compare','one')
    assert again['reusable'] and not again['executed']
    assert again['budget']['reserved_counts']==costs
    assert not again['parent_training_budget_verified']
    with pytest.raises(ValueError,match='exhausted'):
        run_registered(tmp_path,source,digest(source),budget,'compare','two')
    metrics=load_plan(out/'metrics.json');metrics['same_day_same_count']=False
    atomic_json(out/'metrics.json',metrics)
    saved['artifacts']['metrics.json']=digest(out/'metrics.json');atomic_json(out/'summary.json',saved)
    with pytest.raises(ValueError,match='metrics reconstruction'):
        verify_comparison(tmp_path,out,digest(out/'summary.json'))


def test_registered_comparison_failure_is_charged_and_not_repeated(tmp_path):
    from research.s20_harness.joint_binary_comparison import run_registered
    from research.s20_harness.trial_budget import Budget
    source=tmp_path/'invalid-comparison.json';atomic_json(source,{})
    costs=dict(model_fits=0,underlying_fits=0,calibrator_fits=0,policy_evaluations=2)
    budget=Budget(tmp_path/'budget.sqlite',dict(budget_id='comparison-failure',limits=costs,
        trials=[dict(trial_id='compare',input_sha256=digest(source),costs=costs,max_attempts=1)]))
    with pytest.raises(ValueError,match='exact ranking comparison'):
        run_registered(tmp_path,source,digest(source),budget,'compare','one')
    assert budget.status()['attempts'][0]['state']=='FAILED'
    result=run_registered(tmp_path,source,digest(source),budget,'compare','one')
    assert not result['executed'] and not result['reusable']
    assert result['budget']['reserved_counts']==costs
