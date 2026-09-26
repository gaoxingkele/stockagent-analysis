from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.joint_run import build as train
from research.s20_harness.joint_endpoints import build as endpoints
from research.s20_harness.joint_paired_comparison import compare_bound
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_joint_binary_scope import joint_plan


def test_bound_joint_policies_keep_empty_days_and_coverage_mismatch(tmp_path):
    refs=[]
    outcome=tmp_path/'endpoints.json'
    for i in range(2):
        value=joint_plan()
        value['policy']['selection']['min_score']=0. if i==0 else 1.
        value['policy']['selection']['max_risk']=1.
        path=tmp_path/f'plan{i}.json';atomic_json(path,value)
        model=train(tmp_path,path,digest(path));directory=Path(model['directory'])
        if i==0:
            ids=pd.read_parquet(directory/'candidate_ledger.parquet').sample_id.tolist()
            atomic_json(outcome,dict(evaluation_at='2024-06-01T00:00:00Z',
                joint=dict(target_id='P.joint.v4',outcomes=[dict(sample_id=sid,target='A',label_available_at='2024-05-29T00:00:00Z') for sid in ids]),
                risk10=dict(target_id='P.B10.v4',outcomes=[dict(sample_id=sid,target=False,label_available_at='2024-05-29T00:00:00Z') for sid in ids])))
        report=endpoints(tmp_path,directory,digest(directory/'summary.json'),outcome,digest(outcome))
        out=Path(report['directory']);refs.append(dict(directory=str(out),summary_sha256=digest(out/'summary.json')))
    result=compare_bound(tmp_path,*refs,draws=200)
    assert result['models_refit']==0 and not result['formal_G3_passed']
    comparison=result['comparison']
    assert comparison['calendar_days']==2 and not comparison['all_dates_same_recommendation_count']
    assert comparison['matched_active_days']==0
    assert all(r['reason']=='coverage_mismatch' and r['intervals'] is None for r in comparison['block_sensitivity'])
    identical=compare_bound(tmp_path,refs[0],refs[0],draws=200)['comparison']
    assert identical['matched_active_days']==1
    assert all(r['reason']=='insufficient_date_blocks' for r in identical['block_sensitivity'])
    from research.s20_harness.joint_paired_comparison import build as persist,verify
    from research.s20_harness.runtime import load_plan
    input_path=tmp_path/'paired-input.json'
    atomic_json(input_path,dict(schema_version='1',candidate=refs[0],baseline=refs[1],draws=200,seed=20))
    saved=persist(tmp_path,input_path,digest(input_path));saved_dir=Path(saved['directory'])
    assert verify(tmp_path,saved_dir,digest(saved_dir/'summary.json'))['comparison_recomputed']
    forged=load_plan(saved_dir/'result.json');forged['comparison']['formal_G3_passed']=True
    atomic_json(saved_dir/'result.json',forged)
    saved['artifacts']['result.json']=digest(saved_dir/'result.json');atomic_json(saved_dir/'summary.json',saved)
    with pytest.raises(ValueError,match='reconstruction'):verify(tmp_path,saved_dir,digest(saved_dir/'summary.json'))
    # Same apparent outcomes with a different evaluation clock are not the same
    # registered snapshot; don't silently allow endpoint-specific cutoff tuning.
    from research.s20_harness.runtime import load_plan
    payload=load_plan(outcome);payload['evaluation_at']='2024-06-02T00:00:00Z'
    newer=tmp_path/'newer.json';atomic_json(newer,payload)
    report=endpoints(tmp_path,directory,digest(directory/'summary.json'),newer,digest(newer))
    updated=dict(directory=report['directory'],summary_sha256=digest(Path(report['directory'])/'summary.json'))
    with pytest.raises(ValueError,match='snapshot/cutoff'):compare_bound(tmp_path,refs[0],updated,draws=200)
