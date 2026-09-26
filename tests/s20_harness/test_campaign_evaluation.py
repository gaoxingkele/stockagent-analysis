import json
from pathlib import Path
import pandas as pd
import pytest

from research.s20_harness.multifold_run import run
from research.s20_harness.campaign_evaluation import build,verify
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_multifold_run import same_fold_models


@pytest.mark.parametrize('atr_control', [False, True])
def test_same_fold_outcomes_maturity_and_no_refit(tmp_path, monkeypatch, capsys, atr_control):
    manifest=same_fold_models(tmp_path)
    if atr_control:
        payload=json.loads(manifest.read_text())
        for i,job in enumerate(payload['jobs']):
            path=Path(job['input_path']);value=json.loads(path.read_text())
            value['policy'].update(mode='atr_liquidity_control',max_atr_fraction=.04,min_traded_value_cny=1000000.)
            value['policy_controls']=[dict(sample_id='outer-test'+str(k),
                available_at='2024-05-02T20:00:00Z',atr_fraction=.02 if k==0 else .1,
                traded_value_cny=2000000.) for k in range(2)]
            atomic_json(path,value);job['input_sha256']=digest(path)
            payload['budget']['trials'][i]['input_sha256']=digest(path)
        atomic_json(manifest,payload)
    campaign=run(tmp_path,manifest,digest(manifest))
    directory=Path(campaign['directory']);receipt=next(directory.glob('receipt-*.json'))
    outcomes=tmp_path/'outcomes.json'
    atomic_json(outcomes,dict(target_id='P.safe.v4',evaluation_at='2024-06-01T00:00:00Z',outcomes=[
        dict(sample_id='outer-test0',target=True,label_available_at='2024-05-28T21:00:00Z'),
        dict(sample_id='outer-test1',target=False,label_available_at='2024-06-02T00:00:00Z')]))
    refs=[dict(fold_id='common',path=str(outcomes),sha256=digest(outcomes))]
    report=build(tmp_path,receipt,digest(receipt),refs)
    table=pd.read_parquet(Path(report['directory'])/'evaluated_predictions.parquet')
    assert len(table)==4 and report['outer_folds']==1 and report['models_refit']==0
    assert table.evaluation_status.value_counts().to_dict()=={'mature':2,'not_mature_at_evaluation':2}
    assert table.loc[table.sample_id.eq('outer-test1'),'evaluation_target'].isna().all()
    assert json.loads((directory/'checkpoint.json').read_text())['budget']['reserved_counts']['model_fits']==2
    evaluated=Path(report['directory'])
    assert verify(tmp_path,evaluated,digest(evaluated/'summary.json'))['metrics_recomputed']
    from research.s20_harness import cli
    monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    assert cli.main(['verify-campaign-evaluation','--directory',str(evaluated),
        '--summary-sha256',digest(evaluated/'summary.json')])==0
    assert json.loads(capsys.readouterr().out)['models_refit']==0
    assert cli.main(['verify-campaign-evaluation','--directory',str(evaluated),
        '--summary-sha256','0'*64])==2
    assert json.loads(capsys.readouterr().out)['valid'] is False
    assert cli.main(['build-baseline-bundle','--directory',str(evaluated),
        '--summary-sha256',digest(evaluated/'summary.json')])==0
    packaged=json.loads(capsys.readouterr().out)
    bundle_dir=Path(packaged['directory'])
    assert set(packaged['artifacts'])=={'split_manifest.json','baseline_oof.parquet',
        'baseline_metrics.csv','baseline_bundle.json'}
    bundled=pd.read_parquet(bundle_dir/'baseline_oof.parquet')
    assert len(bundled)==4 and bundled.evaluation_target.isna().sum()==2
    assert bundled.recorded_oof_provenance_valid.all()
    assert len(pd.read_csv(bundle_dir/'baseline_metrics.csv'))==4
    assert packaged['formal_H03_accepted'] is False and packaged['models_refit']==0
    assert json.loads((bundle_dir/'baseline_bundle.json').read_text())['legacy_reproduction_complete'] is False
    from research.s20_harness.baseline_bundle import verify as verify_bundle
    checked=verify_bundle(tmp_path,bundle_dir,digest(bundle_dir/'summary.json'))
    assert checked['bundle_values_recomputed'] and checked['rows']==4 and checked['models_refit']==0
    assert cli.main(['verify-baseline-bundle','--directory',str(bundle_dir),
        '--summary-sha256',digest(bundle_dir/'summary.json')])==0
    assert json.loads(capsys.readouterr().out)['valid']
    bundle_meta=json.loads((bundle_dir/'baseline_bundle.json').read_text())
    coverage=bundle_meta['baseline_coverage']['observed_scope'][0]
    assert ('atr_liquidity_control' not in coverage['missing_families']) == atr_control
    assert {c['model_family'] for c in coverage['policy_inventory']}=={'logistic','shallow_tree'}
    assert {c['policy_mode'] for c in coverage['policy_inventory']}=={
        'atr_liquidity_control' if atr_control else 'score_only_control'}
    assert not bundle_meta['baseline_coverage']['formal_H03_accepted']
    original_csv=(bundle_dir/'baseline_metrics.csv').read_bytes()
    changed_metrics=pd.read_csv(bundle_dir/'baseline_metrics.csv')
    changed_metrics.loc[0,'brier_known_only']=.987654321
    changed_metrics.to_csv(bundle_dir/'baseline_metrics.csv',index=False)
    forged_meta=json.loads(json.dumps(bundle_meta));forged_summary=json.loads(json.dumps(packaged))
    forged_meta['artifacts']['baseline_metrics.csv']=digest(bundle_dir/'baseline_metrics.csv')
    atomic_json(bundle_dir/'baseline_bundle.json',forged_meta)
    forged_summary['artifacts']['baseline_metrics.csv']=digest(bundle_dir/'baseline_metrics.csv')
    forged_summary['artifacts']['baseline_bundle.json']=digest(bundle_dir/'baseline_bundle.json')
    atomic_json(bundle_dir/'summary.json',forged_summary)
    with pytest.raises(AssertionError): verify_bundle(tmp_path,bundle_dir,digest(bundle_dir/'summary.json'))
    (bundle_dir/'baseline_metrics.csv').write_bytes(original_csv)
    atomic_json(bundle_dir/'baseline_bundle.json',bundle_meta);atomic_json(bundle_dir/'summary.json',packaged)
    original_metrics=json.loads((evaluated/'metrics_by_job.json').read_text())
    changed=json.loads(json.dumps(original_metrics))
    changed[0]['metrics']['status_counts']['mature']=999
    atomic_json(evaluated/'metrics_by_job.json',changed)
    modified=dict(report)
    modified['artifacts']=dict(report['artifacts'])
    modified['artifacts']['metrics_by_job.json']=digest(evaluated/'metrics_by_job.json')
    atomic_json(evaluated/'summary.json',modified)
    with pytest.raises(ValueError,match='metrics reconstruction'):
        verify(tmp_path,evaluated,digest(evaluated/'summary.json'))
    atomic_json(evaluated/'metrics_by_job.json',original_metrics)
    atomic_json(evaluated/'summary.json',report)
    with pytest.raises(ValueError,match='one shared'):
        build(tmp_path,receipt,digest(receipt),refs+refs)
    with pytest.raises(ValueError,match='exact fold'):
        build(tmp_path,receipt,digest(receipt),[None])
    from research.s20_harness import campaign_evaluation as module
    original=module.evaluate_job
    def corrupt_member(*args,**kwargs):
        result=original(*args,**kwargs)
        atomic_json(Path(result['directory'])/'metrics.json',{'tampered':True})
        return result
    monkeypatch.setattr(module,'evaluate_job',corrupt_member)
    with pytest.raises(ValueError,match='source changed'):
        build(tmp_path,receipt,digest(receipt),refs)
