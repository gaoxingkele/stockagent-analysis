from pathlib import Path
import pytest
from research.s20_harness.feature_group_ablation import plans
from research.s20_harness.selection_plan import inspect
from research.s20_harness.runtime import load_plan,atomic_json,digest
from tests.s20_harness.test_selection_plan import setup


def ablation_setup(tmp_path):
    path=setup(tmp_path,('multinomial',)*3);selection=load_plan(path)
    contract=dict(groups={'base':['x'],'addition':['z']},base_groups=['base'],added_group='addition',noise_seed=20,
                  arms={'base':'0','augmented':'1','noise_control':'2'})
    for ref,arm in zip(selection['candidates'],contract['arms']):
        manifest=load_plan(ref['manifest_path'])
        for job,trial in zip(manifest['jobs'],manifest['budget']['trials']):
            value=load_plan(job['input_path']);value['feature_contract']['columns']=['x','z']
            for i,row in enumerate(value['features']):row['z']=float(i%3)
            variants,_=plans(value,contract['groups'],contract['base_groups'],contract['added_group'])
            atomic_json(Path(job['input_path']),variants[arm]);job['input_sha256']=digest(Path(job['input_path']))
            trial['input_sha256']=job['input_sha256']
        atomic_json(Path(ref['manifest_path']),manifest);ref['manifest_sha256']=digest(Path(ref['manifest_path']))
    contract['control_strata']=[dict(feature='x',cuts=[.5,1.5])]
    selection['feature_ablation']=contract;atomic_json(path,selection)
    return path


def test_complete_registered_ablation_uses_shared_grid_and_budget(tmp_path):
    path=ablation_setup(tmp_path);report=inspect(tmp_path,path,digest(path))
    assert report['candidate_ids']==['0','1','2']
    assert report['planned_initial_costs']['model_fits']==18
    assert not report['training_authorized']


def test_registered_ablation_executes_all_eighteen_owned_jobs(tmp_path):
    from research.s20_harness.search_registration import register
    from research.s20_harness.registered_search_run import run,verify_receipt
    path=ablation_setup(tmp_path);sha=digest(path)
    registration=register(tmp_path,path,sha)
    result=run(tmp_path,registration['directory'],sha,max_new_jobs=18)
    assert result['status']=='COMPLETED_SYNTHETIC' and len(result['jobs'])==18
    assert result['budget']['reserved_counts']==dict(model_fits=18,underlying_fits=36,calibrator_fits=18,policy_evaluations=18)
    receipt=Path(result['receipt_path'])
    assert verify_receipt(tmp_path,receipt,digest(receipt))['search_execution_receipt_verified']
    import pandas as pd
    from research.s20_harness.search_evaluation import build as evaluate
    from research.s20_harness.search_candidates import inspect as summarize
    references=[]
    for fold in ['0','1','2']:
        job=next(j for j in result['jobs'] if j['fold_id']==fold)
        rows=pd.read_parquet(Path(job['artifact']['directory'])/'candidate_ledger.parquet')
        outcomes=tmp_path/f'outcomes-{fold}.json'
        atomic_json(outcomes,dict(target_id='P.joint.v4',evaluation_at='2026-09-14T00:00:00Z',outcomes=[
            dict(sample_id=sid,target='A' if i==0 else None,label_available_at='2026-06-01T00:00:00Z' if i==0 else None)
            for i,sid in enumerate(rows.sample_id)]))
        references.append(dict(fold_id=fold,path=str(outcomes),sha256=digest(outcomes)))
    budget=Path(registration['shared_budget_path']);before=digest(budget)
    evaluated=evaluate(tmp_path,receipt,digest(receipt),references);directory=Path(evaluated['directory'])
    summary=summarize(tmp_path,directory,digest(directory/'summary.json'))
    assert evaluated['rows']==36 and len(summary['cells'])==18
    ablation=summary['feature_ablation_comparisons']
    assert [(c['left_arm'],c['right_arm']) for c in ablation['comparisons']]==[
        ('augmented','base'),('augmented','noise_control'),('noise_control','base')]
    assert not ablation['orthogonal_information_proven'] and not ablation['formal_H06_accepted']
    assert all(len(c['per_seed'])==2 for c in summary['candidates'])
    assert len(summary['feature_control_strata'])==18
    assert sum(s['candidates'] for row in summary['feature_control_strata'] for s in row['strata'])==36
    assert pd.read_parquet(directory/'evaluated_predictions.parquet').safe_profit_target.isna().sum()==18
    assert digest(budget)==before


@pytest.mark.parametrize('bad',['noise_values','base_values','policy','groups','missing_contract'])
def test_rehashed_unregistered_variations_rejected(tmp_path,bad):
    path=ablation_setup(tmp_path);selection=load_plan(path)
    if bad=='groups':selection['feature_ablation']['groups']['addition']=['x','z']
    elif bad=='missing_contract':del selection['feature_ablation']
    else:
        ref=selection['candidates'][2 if bad=='noise_values' else 0]
        manifest=load_plan(ref['manifest_path']);job=manifest['jobs'][0]
        value=load_plan(job['input_path'])
        if bad=='policy':value['policy']['weights']['mu']=4.
        else:value['features'][0]['z' if bad=='noise_values' else 'x']+=.1
        atomic_json(Path(job['input_path']),value);job['input_sha256']=digest(Path(job['input_path']))
        manifest['budget']['trials'][0]['input_sha256']=job['input_sha256']
        atomic_json(Path(ref['manifest_path']),manifest);ref['manifest_sha256']=digest(Path(ref['manifest_path']))
    atomic_json(path,selection)
    with pytest.raises(ValueError):inspect(tmp_path,path,digest(path))
