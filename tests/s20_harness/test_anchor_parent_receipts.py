from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.baseline_run import build
from research.s20_harness.anchor_parent_receipts import bind
from research.s20_harness.oof_audit import membership_hash
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_baseline_run import plan
import json


def test_real_parent_output_binding_rejects_fabricated_scores_and_dependencies(tmp_path):
    value=plan();path=tmp_path/'parent.json';atomic_json(path,value)
    result=build(tmp_path,path,digest(path));directory=Path(result['directory'])
    saved=pd.read_parquet(directory/'calibrated_predictions.parquet')
    picked=saved.loc[saved.segment.eq('outer-test')]
    deps=load_plan(directory/'baseline_card.json')['fit_sample_ids']+load_plan(directory/'calibration_card.json')['calibration_sample_ids']
    context=dict(samples=pd.DataFrame(value['samples']),
        predictions=pd.DataFrame(dict(sample_id=picked.sample_id,model_id='m',score=picked.calibrated_probability)),
        models={'m':dict(model_path=str(directory/'calibration_card.json'),model_sha256=digest(directory/'calibration_card.json'),
            information_cutoff_at='2024-04-01T00:00:00Z',dependency_sha256=membership_hash(deps))},
        dependencies={'m':deps},feature_name='anchor')
    bindings=[dict(model_id='m',directory=str(directory),summary_sha256=digest(directory/'summary.json'))]
    assert bind(tmp_path,context,bindings)['bound_prediction_rows']==2
    original=context['predictions'].copy()
    context['predictions'].iloc[0,2]+=.01
    with pytest.raises(AssertionError): bind(tmp_path,context,bindings)
    context['predictions']=original
    context['dependencies']['m']=deps[:-1]
    with pytest.raises(ValueError,match='dependency mismatch'): bind(tmp_path,context,bindings)


@pytest.mark.parametrize('budgeted',[False,True])
def test_complete_parent_child_chain_and_replay(tmp_path,budgeted):
    from research.s20_harness.baseline_replay import replay
    from research.s20_harness.baseline_outer_ledger import collect
    child=plan()
    parent=json.loads(json.dumps(plan()).replace('2024','2023'))
    for field in ['samples','features','fit_labels','calibration_labels']:
        for row in parent[field]: row['sample_id']='parent_'+row['sample_id']
    parent['boundaries'][-1]['end_at']='2025-01-01T00:00:00Z'
    parent['samples'].extend(child['samples'])
    parent['features'].extend(child['features'])
    parent['calendar']=sorted(set(parent['calendar']+[s['signal_date'] for s in child['samples']]))
    source=tmp_path/'parent-chain.json';atomic_json(source,parent)
    if budgeted:
        from research.s20_harness.bounded_baseline import run as owned
        from research.s20_harness.trial_budget import Budget
        from research.s20_harness.process_runner import Limits
        from tests.s20_harness.test_trial_budget import contract
        cfg=contract(digest(source))
        cfg['limits']={k:v*2 for k,v in cfg['limits'].items()}
        budget=Budget(tmp_path/'parent-budget.sqlite',cfg)
        limits=Limits(120,2*1024**3,1)
        executed=owned(tmp_path,source,digest(source),budget,'baseline','one',limits)
        directory=Path(executed['artifact']['directory'])
    else:
        report=build(tmp_path,source,digest(source));directory=Path(report['directory'])
    predicted=pd.read_parquet(directory/'calibrated_predictions.parquet').set_index('sample_id')
    selected=predicted.loc[[s['sample_id'] for s in child['samples']]]
    assert selected.calibrated_probability.notna().all()
    deps=load_plan(directory/'baseline_card.json')['fit_sample_ids']+load_plan(directory/'calibration_card.json')['calibration_sample_ids']
    child['feature_contract']['columns'].append('anchor_score')
    child['anchor_context']=dict(samples=parent['samples'],
        predictions=[dict(sample_id=key,model_id='parent',score=float(row.calibrated_probability)) for key,row in selected.iterrows()],
        models={'parent':dict(model_path=str(directory/'calibration_card.json'),model_sha256=digest(directory/'calibration_card.json'),
            information_cutoff_at='2023-04-01T00:00:00Z',dependency_sha256=membership_hash(deps))},
        dependencies={'parent':deps},feature_name='anchor_score')
    child['anchor_parent_bindings']=[dict(model_id='parent',directory=str(directory),summary_sha256=digest(directory/'summary.json'))]
    if budgeted:
        child['anchor_parent_bindings'][0]['budget_reference']=dict(path=str(budget.path),contract_sha256=budget.sha,
            trial_id='baseline',attempt_id='one',input_path=str(source),input_sha256=digest(source),
            artifact=executed['artifact'],limits=limits.__dict__)
    child_path=tmp_path/'child-chain.json';atomic_json(child_path,child)
    output=build(tmp_path,child_path,digest(child_path));child_dir=Path(output['directory'])
    evidence=load_plan(child_dir/'anchor_parent_evidence.json')
    assert evidence['bound_prediction_rows']==10 and evidence['bound_parent_models']==1
    assert evidence['upstream_fit_budget_verified']==budgeted
    if budgeted:
        budget.reserve('baseline','two',digest(source))
        budget.finish('baseline','two','FAILED',{'synthetic':'later unrelated attempt'})
        from research.s20_harness.campaign_parent_budget import inspect
        inventory=inspect(tmp_path,[child,child],{'model_fits':2},tmp_path/'child-budget.sqlite')
        assert inventory['unique_parent_attempts']==1
        assert inventory['external_reserved_counts']['model_fits']==2
        assert inventory['combined_model_fit_capacity']==4
        with pytest.raises(ValueError,match='exceeds 72'):
            inspect(tmp_path,[child],{'model_fits':71},tmp_path/'child-budget.sqlite')
    outer=collect([dict(job_id='child',fold_id='one',directory=str(child_dir),summary_sha256=digest(child_dir/'summary.json'))])
    assert len(outer)==2 and outer.recorded_oof_provenance_valid.all()
    assert outer.recorded_label_dependency_count.eq(8).all()
    assert replay(tmp_path,child_dir,digest(child_dir/'summary.json'))['semantic_replay_performed']
    from research.s20_harness.baseline_run import verify
    changed=pd.read_parquet(directory/'calibrated_predictions.parquet')
    changed.loc[changed.sample_id.eq('outer-test0'),'calibrated_probability']=.12345
    changed.to_parquet(directory/'calibrated_predictions.parquet',index=False)
    with pytest.raises(ValueError,match='anchor parent source changed'):
        verify(child_dir,digest(child_dir/'summary.json'))
    with pytest.raises(ValueError,match='anchor parent source changed'):
        collect([dict(job_id='child',fold_id='one',directory=str(child_dir),summary_sha256=digest(child_dir/'summary.json'))])
