import json
from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.multifold_run import run
from research.s20_harness.runtime import atomic_json,digest,load_plan
from research.s20_harness.baseline_outer_ledger import verify_receipt
from tests.s20_harness.test_multifold_run import setup as binary_setup
from tests.s20_harness.test_joint_run import plan


def setup(tmp_path):
    source=binary_setup(tmp_path);manifest=load_plan(source)
    manifest['schema_version']='3'
    for i,job in enumerate(manifest['jobs']):
        path=Path(job['input_path'])
        # A serialized campaign uses JSON null for missing availability, not
        # pandas' non-standard NaN token; keep the unavailable candidate.
        atomic_json(path,json.loads(json.dumps(plan()).replace('2024',str(2024+i)),parse_constant=lambda _:None))
        job['pipeline_kind']='joint';job['input_sha256']=digest(path)
        manifest['budget']['trials'][i]['input_sha256']=digest(path)
    atomic_json(source,manifest)
    return source


def test_joint_two_folds_retained_and_reused(tmp_path):
    source=setup(tmp_path)
    first=run(tmp_path,source,digest(source))
    rows=pd.read_parquet(first['outer_prediction_ledger']['path'])
    assert len(rows)==4 and rows.p_A.isna().sum()==2
    assert set(rows.target_id)=={'P.joint.v4'} and 'raw_p_D' in rows
    assert rows.loc[rows.p_A.notna(),'recorded_oof_provenance_valid'].all()
    assert not rows.historical_availability_proven.any()
    second=run(tmp_path,source,digest(source),directory=first['directory'])
    assert second['distinct_outer_folds']==2 and second['budget']['reserved_counts']['model_fits']==2
    assert not any(j['executed_this_call'] for j in second['jobs'])
    receipt=next(Path(first['directory']).glob('receipt-*.json'))
    assert verify_receipt(tmp_path,receipt,digest(receipt))['outer_ledger_reconstructed']


@pytest.mark.parametrize('kind',['identity','unknown','legacy'])
def test_joint_campaign_identity_rejected_prelaunch(tmp_path,kind):
    source=setup(tmp_path);manifest=load_plan(source)
    if kind=='legacy':
        manifest['schema_version']='2'
        for j in manifest['jobs']:del j['pipeline_kind']
    else:manifest['jobs'][0]['pipeline_kind']='baseline' if kind=='identity' else 'other'
    atomic_json(source,manifest)
    with pytest.raises(ValueError,match='identity'):run(tmp_path,source,digest(source))
    assert not (tmp_path/'output').exists()
