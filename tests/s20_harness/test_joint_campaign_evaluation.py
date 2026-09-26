from pathlib import Path
import json
import pandas as pd
import pytest
from research.s20_harness.campaign_evaluation import build,verify
from research.s20_harness.multifold_run import run
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_joint_multifold import setup


def test_joint_campaign_shared_maturity_and_reconstruction(tmp_path):
    source=setup(tmp_path);campaign=run(tmp_path,source,digest(source))
    receipt=next(Path(campaign['directory']).glob('receipt-*.json'))
    refs=[]
    for i,job in enumerate(campaign['jobs']):
        path=tmp_path/f'outcomes{i}.json'
        ids=pd.read_parquet(Path(job['directory'])/'candidate_ledger.parquet').sample_id.tolist()
        # IDs intentionally recur across folds. Fold binding must keep their
        # different outcomes and clocks separate, not deduplicate by sample_id.
        atomic_json(path,dict(target_id='P.joint.v4',evaluation_at='2025-06-01T00:00:00Z',outcomes=[
            dict(sample_id=ids[0],target='A' if i==0 else 'D',label_available_at=f'{2024+i}-05-29T00:00:00Z'),
            dict(sample_id=ids[1],target='B',label_available_at='2025-06-02T00:00:00Z')]))
        refs.append(dict(fold_id=job['fold_id'],path=str(path),sha256=digest(path)))
    result=build(tmp_path,receipt,digest(receipt),refs);out=Path(result['directory'])
    checked=verify(tmp_path,out,digest(out/'summary.json'))
    assert checked['metrics_recomputed'] and checked['models_refit']==0
    rows=pd.read_parquet(out/'evaluated_predictions.parquet')
    assert len(rows)==4 and not rows.duplicated(['job_id','sample_id']).any()
    assert rows.evaluation_class.dropna().tolist()==['A','D']
    assert rows.evaluation_status.eq('not_mature_at_evaluation').sum()==2
    assert rows.p_A.isna().sum()==2
    members=json.loads((out/'metrics_by_job.json').read_text(encoding='utf-8'))
    assert members[0]['metrics']['events']['safe_profit']['all_candidates']['event_bounds']['rate_lower']==.5
    assert members[1]['metrics']['events']['down5']['all_candidates']['event_bounds']['rate_lower']==.5
    assert not result['model_rows_are_independent_observations']
    with pytest.raises(ValueError,match='one shared'):build(tmp_path,receipt,digest(receipt),refs[:1])
    payload=load_plan(Path(refs[1]['path']));payload['evaluation_at']='2025-06-03T00:00:00Z'
    atomic_json(Path(refs[1]['path']),payload);refs[1]['sha256']=digest(Path(refs[1]['path']))
    with pytest.raises(ValueError,match='shared campaign evaluation cutoff'):
        build(tmp_path,receipt,digest(receipt),refs)
    with pytest.raises(ValueError,match='pin mismatch'):verify(tmp_path,out,digest(out/'summary.json'))
