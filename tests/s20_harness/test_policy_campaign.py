import json
from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness.multifold_run import run, comparison_scope, check_control_scope
from research.s20_harness.baseline_outer_ledger import verify_receipt
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_multifold_run import same_fold_models


def setup(tmp_path):
    path=same_fold_models(tmp_path);manifest=json.loads(path.read_text())
    manifest.update(schema_version='5',comparison_contract='fixed_atr_policy_control')
    for i,job in enumerate(manifest['jobs']):
        job['pipeline_kind']='baseline'
        source=Path(job['input_path']);plan=json.loads(source.read_text())
        plan['model_family']='logistic'
        if i:
            plan['policy'].update(policy_id='atr-control',mode='atr_liquidity_control',
                max_atr_fraction=.04,min_traded_value_cny=1000000.)
            plan['policy_controls']=[dict(sample_id='outer-test'+str(k),
                available_at='2024-05-02T20:00:00Z',atr_fraction=.02 if k==0 else .1,
                traded_value_cny=2000000.) for k in range(2)]
        atomic_json(source,plan);job['input_sha256']=digest(source)
        manifest['budget']['trials'][i]['input_sha256']=digest(source)
    atomic_json(path,manifest)
    return path


def test_owned_policy_comparison_preserves_predictions_budget_and_receipt(tmp_path):
    path=setup(tmp_path);report=run(tmp_path,path,digest(path))
    assert report['same_fold_variation_allowed']==['policy','policy_controls']
    assert report['budget']['reserved_counts']['policy_evaluations']==2
    assert report['budget']['reserved_counts']['model_fits']==2
    tables=[pd.read_parquet(Path(j['directory'])/'candidate_ledger.parquet') for j in report['jobs']]
    pd.testing.assert_series_equal(tables[0].score,tables[1].score,check_exact=True)
    assert tables[0].loc[tables[0].selected,'sample_id'].tolist()==['outer-test1']
    assert tables[1].loc[tables[1].selected,'sample_id'].tolist()==['outer-test0']
    receipt=next(Path(report['directory']).glob('receipt-*.json'))
    checked=verify_receipt(tmp_path,receipt,digest(receipt))
    assert checked['rows']==4 and checked['models_refit']==0
    reused=run(tmp_path,path,digest(path),directory=report['directory'])
    assert not any(j['executed_this_call'] for j in reused['jobs'])
    from research.s20_harness.campaign_evaluation import build,verify
    outcomes=tmp_path/'outcomes.json'
    atomic_json(outcomes,dict(target_id='P.safe.v4',evaluation_at='2024-06-03T00:00:00Z',
        outcomes=[dict(sample_id='outer-test'+str(k),target=bool(k==0),
            label_available_at='2024-05-28T21:00:00Z') for k in range(2)]))
    result=build(tmp_path,receipt,digest(receipt),
        [dict(fold_id='common',path=str(outcomes),sha256=digest(outcomes))])
    evaluated=Path(result['directory'])
    assert verify(tmp_path,evaluated,digest(evaluated/'summary.json'))['metrics_recomputed']
    assert len(pd.read_parquet(evaluated/'evaluated_predictions.parquet'))==4


@pytest.mark.parametrize('field',['model_family','features','fit_labels','n_cap','min_score','frozen_at'])
def test_policy_contract_rejects_hidden_changes_before_fit(tmp_path,field):
    path=setup(tmp_path);manifest=json.loads(path.read_text());job=manifest['jobs'][1]
    source=Path(job['input_path']);plan=json.loads(source.read_text())
    if field=='model_family':plan[field]='shallow_tree'
    elif field in ['features','fit_labels']:plan[field]=plan[field][:-1]
    elif field=='n_cap':plan['policy'][field]=3
    elif field=='min_score':plan['policy'][field]=.3
    else:plan['policy'][field]='2024-05-01T01:00:00Z'
    atomic_json(source,plan);job['input_sha256']=digest(source)
    manifest['budget']['trials'][1]['input_sha256']=digest(source);atomic_json(path,manifest)
    with pytest.raises(ValueError,match='comparison scope mismatch'):run(tmp_path,path,digest(path))
    assert not (tmp_path/'output').exists()


def test_different_atr_thresholds_cannot_change_control_data(tmp_path):
    path=setup(tmp_path);manifest=json.loads(path.read_text())
    plan=json.loads(Path(manifest['jobs'][1]['input_path']).read_text())
    variant=json.loads(json.dumps(plan));variant['policy']['max_atr_fraction']=.03
    variation=['policy','policy_controls']
    assert comparison_scope(plan,variation)==comparison_scope(variant,variation)
    scopes={};check_control_scope(plan,'f',variation,scopes)
    check_control_scope(variant,'f',variation,scopes)
    variant['policy_controls'][0]['atr_fraction']=.01
    with pytest.raises(ValueError,match='control data mismatch'):
        check_control_scope(variant,'f',variation,scopes)


def test_policy_count_cap_rejected_before_execution(tmp_path):
    path=setup(tmp_path);manifest=json.loads(path.read_text())
    manifest['jobs']=manifest['jobs'][:1]*7
    atomic_json(path,manifest)
    with pytest.raises(ValueError,match='six policies per fold'):run(tmp_path,path,digest(path))
    assert not (tmp_path/'output').exists()
