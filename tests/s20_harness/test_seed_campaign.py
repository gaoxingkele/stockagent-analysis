import json
from pathlib import Path

import pytest

from research.s20_harness.multifold_run import run, comparison_variation, check_seed_grid
from research.s20_harness.runtime import atomic_json, digest, load_plan
from tests.s20_harness.test_multifold_run import same_fold_models


def setup(tmp_path,pipeline='baseline'):
    source=same_fold_models(tmp_path);manifest=load_plan(source)
    manifest.update(schema_version='7',comparison_contract='fixed_model_seed_control',
                    seed_grid=[20,71],fold_grid=['common'])
    for i,job in enumerate(manifest['jobs']):
        path=Path(job['input_path']);plan=load_plan(path)
        if pipeline=='joint':
            from tests.s20_harness.test_joint_run import plan as joint_plan
            plan=joint_plan()
            # The direct trainer fixture contains pandas NaN for this missing
            # timestamp; campaign inputs require JSON null, not nonstandard NaN.
            plan['samples'][9]['feature_available_at']=None
        plan.update(model_family='multinomial' if pipeline=='joint' else 'shallow_tree',random_seed=manifest['seed_grid'][i])
        atomic_json(path,plan);job.update(pipeline_kind=pipeline,input_sha256=digest(path))
        manifest['budget']['trials'][i]['input_sha256']=digest(path)
    atomic_json(source,manifest)
    return source


@pytest.mark.parametrize('pipeline',['baseline','joint'])
def test_owned_seed_campaign_replays_without_refit(tmp_path,pipeline):
    from research.s20_harness.baseline_outer_ledger import verify_receipt
    source=setup(tmp_path,pipeline);report=run(tmp_path,source,digest(source))
    assert report['same_fold_variation_allowed']==['random_seed']
    assert report['distinct_outer_folds']==1
    assert report['budget']['reserved_counts']['model_fits']==2
    cards=[load_plan(Path(job['directory'])/('joint_card.json' if pipeline=='joint' else 'baseline_card.json')) for job in report['jobs']]
    assert [c['parameters']['random_state'] for c in cards]==[20,71]
    assert all(c['randomness']['independent_market_evidence'] is False for c in cards)
    repeat=run(tmp_path,source,digest(source),directory=report['directory'])
    assert not any(j['executed_this_call'] for j in repeat['jobs'])
    receipt=Path(report['directory'])/'seed-consumer-receipt.json';atomic_json(receipt,repeat)
    assert verify_receipt(tmp_path,receipt,digest(receipt))['models_refit']==0


@pytest.mark.parametrize('kind',['missing','duplicate','undeclared','implicit','family','policy','legacy'])
def test_bad_seed_campaign_rejected_before_execution(tmp_path,kind):
    source=setup(tmp_path);manifest=load_plan(source)
    job=manifest['jobs'][1];path=Path(job['input_path']);plan=load_plan(path)
    if kind=='missing':manifest['fold_grid'].append('absent')
    elif kind=='duplicate':plan['random_seed']=20
    elif kind=='undeclared':plan['random_seed']=72
    elif kind=='implicit':del plan['random_seed']
    elif kind=='family':plan['model_family']='logistic'
    elif kind=='policy':plan['policy']['n_cap']=2
    else:
        manifest['schema_version']='3'
        for field in ['comparison_contract','seed_grid','fold_grid']:del manifest[field]
    atomic_json(path,plan);job['input_sha256']=digest(path)
    manifest['budget']['trials'][1]['input_sha256']=digest(path);atomic_json(source,manifest)
    with pytest.raises(ValueError):run(tmp_path,source,digest(source))
    assert not (tmp_path/'output/experiments/s20_safe_v4/sources').exists()


@pytest.mark.parametrize('seeds',[[True,71],[20,20],[20],[20,21,22,23],[20,-1]])
def test_seed_grid_schema(seeds):
    manifest=dict(schema_version='7',comparison_contract='fixed_model_seed_control',seed_grid=seeds,
        fold_grid=['f'],evidence_mode='synthetic',budget={},jobs=[],limits={})
    with pytest.raises(ValueError):comparison_variation(manifest)


def test_missing_plan_or_changed_model_across_folds_rejected():
    manifest=dict(schema_version='7',seed_grid=[20,71],fold_grid=['a','b'],jobs=[
        dict(fold_id=f,pipeline_kind='baseline') for f in ['a','a','b','b']])
    plans=[dict(random_seed=s,model_family='logistic',target_id='P.safe.v4',feature_contract={})
           for s in [20,71,20,71]]
    check_seed_grid(manifest,plans)
    with pytest.raises(ValueError,match='cardinality'):check_seed_grid(manifest,plans[:-1])
    plans[2]['model_family']=plans[3]['model_family']='shallow_tree'
    with pytest.raises(ValueError,match='fixed model'):check_seed_grid(manifest,plans)


def test_across_fold_policy_only_freeze_time_may_change():
    manifest=dict(schema_version='7',seed_grid=[20,71],fold_grid=['a','b'],jobs=[
        dict(fold_id=f,pipeline_kind='baseline') for f in ['a','a','b','b']])
    plans=[dict(random_seed=s,model_family='logistic',target_id='P.safe.v4',feature_contract={},
        policy=dict(n_cap=1,frozen_at=str(i//2))) for i,s in enumerate([20,71,20,71])]
    check_seed_grid(manifest,plans)
    plans[2]['policy']['n_cap']=plans[3]['policy']['n_cap']=2
    with pytest.raises(ValueError,match='fixed model'):check_seed_grid(manifest,plans)
