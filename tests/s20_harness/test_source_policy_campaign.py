import json
from pathlib import Path
import pandas as pd
import pytest

from research.s20_harness.atr_control_sources import derive
from research.s20_harness.multifold_run import run,comparison_scope,check_control_scope
from research.s20_harness.baseline_outer_ledger import verify_receipt
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_policy_campaign import setup as bare_setup
from tests.s20_harness.test_atr_control_sources import fixture,rewrite_source


def setup(tmp_path):
    path=bare_setup(tmp_path);manifest=json.loads(path.read_text())
    manifest.update(schema_version='6',comparison_contract='source_bound_atr_policy_control')
    bindings=[]
    for i in range(2):
        folder=tmp_path/('source'+str(i));folder.mkdir()
        _,spec=fixture(folder)
        rewrite_source(spec,lambda d:d.update(entity_id='A' if i==0 else 'B'))
        if i:rewrite_source(spec,lambda d:[b.update(high=11.,low=9.) for b in d['bars']])
        spec['bindings'][0]['sample_id']='outer-test'+str(i)
        bindings.extend(spec['bindings'])
    spec['bindings']=bindings
    for i,job in enumerate(manifest['jobs']):
        source=Path(job['input_path']);plan=json.loads(source.read_text())
        for s in plan['samples']:
            if s['sample_id'].startswith('outer-test'):s['prediction_at']='2024-05-03T21:00:00+08:00'
        if i:
            samples=pd.DataFrame(plan['samples']);samples=samples.loc[samples.sample_id.str.startswith('outer-test')]
            controls,_=derive(tmp_path,samples[['sample_id','entity_id','signal_date','prediction_at']],spec)
            plan['policy_controls']=controls.to_dict('records');plan['policy_control_sources']=spec
            plan['policy']['max_atr_fraction']=.05
        atomic_json(source,plan);job['input_sha256']=digest(source)
        manifest['budget']['trials'][i]['input_sha256']=digest(source)
    atomic_json(path,manifest)
    return path


def test_owned_source_comparison_and_reverified_receipt(tmp_path):
    path=setup(tmp_path);report=run(tmp_path,path,digest(path))
    receipt=next(Path(report['directory']).glob('receipt-*.json'))
    assert verify_receipt(tmp_path,receipt,digest(receipt))['rows']==4
    ledgers=[pd.read_parquet(Path(j['directory'])/'candidate_ledger.parquet') for j in report['jobs']]
    pd.testing.assert_series_equal(ledgers[0].score,ledgers[1].score,check_exact=True)
    assert ledgers[0].loc[ledgers[0].selected,'sample_id'].tolist()==['outer-test1']
    assert ledgers[1].loc[ledgers[1].selected,'sample_id'].tolist()==['outer-test0']
    assert report['budget']['reserved_counts']['model_fits']==2
    reused=run(tmp_path,path,digest(path),directory=report['directory'])
    assert not any(j['executed_this_call'] for j in reused['jobs'])
    source_plan=json.loads((Path(report['jobs'][1]['directory'])/'input.json').read_text())
    artifact=Path(source_plan['policy_control_sources']['bindings'][0]['artifact_path'])
    atomic_json(artifact,{})
    with pytest.raises(ValueError,match='hash mismatch'):verify_receipt(tmp_path,receipt,digest(receipt))


def test_source_scope_and_legacy_contract_remain_strict(tmp_path):
    path=setup(tmp_path);manifest=json.loads(path.read_text())
    base,filtered=[json.loads(Path(j['input_path']).read_text()) for j in manifest['jobs']]
    variation=['policy','policy_controls','policy_control_sources']
    assert comparison_scope(base,variation)==comparison_scope(filtered,variation)
    assert comparison_scope(base,['policy','policy_controls'])!=comparison_scope(filtered,['policy','policy_controls'])
    scopes={};check_control_scope(filtered,'fold',variation,scopes)
    changed=json.loads(json.dumps(filtered));changed['policy_control_sources']['bindings'][0]['artifact_sha256']='0'*64
    with pytest.raises(ValueError,match='control data mismatch'):check_control_scope(changed,'fold',variation,scopes)
    del changed['policy_control_sources']
    with pytest.raises(ValueError,match='source evidence'):comparison_scope(changed,variation)
    changed=dict(base,model_family='shallow_tree')
    assert comparison_scope(changed,variation)!=comparison_scope(filtered,variation)


@pytest.mark.parametrize('fully_mature',[False,True])
def test_source_policy_evaluation_to_verified_h03_bundle(tmp_path,fully_mature):
    from research.s20_harness.campaign_evaluation import build as evaluate
    from research.s20_harness.baseline_bundle import build,verify
    path=setup(tmp_path);campaign=run(tmp_path,path,digest(path))
    receipt=next(Path(campaign['directory']).glob('receipt-*.json'))
    outcomes=tmp_path/'outcomes.json'
    atomic_json(outcomes,dict(target_id='P.safe.v4',evaluation_at='2024-06-01T00:00:00Z',
        outcomes=[dict(sample_id='outer-test'+str(i),target=i==0,
            label_available_at='2024-05-28T21:00:00Z' if i==0 or fully_mature else '2024-06-02T00:00:00Z')
            for i in range(2)]))
    evaluated=evaluate(tmp_path,receipt,digest(receipt),
        [dict(fold_id='common',path=str(outcomes),sha256=digest(outcomes))])
    source=Path(evaluated['directory']);bundle=build(tmp_path,source,digest(source/'summary.json'))
    out=Path(bundle['directory'])
    assert verify(tmp_path,out,digest(out/'summary.json'))['bundle_values_recomputed']
    rows=pd.read_parquet(out/'baseline_oof.parquet')
    assert len(rows)==4 and rows.evaluation_target.isna().sum()==(0 if fully_mature else 2)
    meta=json.loads((out/'baseline_bundle.json').read_text())
    assert 'control_source_evidence' not in meta['jobs'][0]
    evidence=meta['jobs'][1]['control_source_evidence']
    assert evidence['status_counts']=={'derived_from_bound_snapshot':2}
    assert evidence['local_receipt_artifact_bindings_verified']
    assert not evidence['price_adjustment_and_calendar_independently_verified']
    assert not meta['formal_H03_accepted']
    # Rehashed report metadata cannot turn local binding into PIT acceptance.
    evidence['price_adjustment_and_calendar_independently_verified']=True
    atomic_json(out/'baseline_bundle.json',meta)
    bundle['artifacts']['baseline_bundle.json']=digest(out/'baseline_bundle.json')
    atomic_json(out/'summary.json',bundle)
    with pytest.raises(ValueError,match='semantic counts'):
        verify(tmp_path,out,digest(out/'summary.json'))
