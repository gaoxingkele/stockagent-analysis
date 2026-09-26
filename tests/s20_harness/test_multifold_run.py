import json
from pathlib import Path

import pytest

from research.s20_harness.multifold_run import run
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_baseline_run import plan


def setup(tmp_path, bad_second=False):
    jobs, trials = [], []
    costs = dict(model_fits=1, underlying_fits=2, calibrator_fits=1, policy_evaluations=1)
    for i in range(2):
        value = json.loads(json.dumps(plan()).replace("2024", str(2024+i)))
        if i == 1 and bad_second:
            value["calibration_labels"][1]["target"] = False
        path = tmp_path/f"fold{i}.json"
        atomic_json(path, value)
        trial_id = f"trial{i}"
        jobs.append(dict(job_id=f"job{i}", fold_id=f"fold{i}", input_path=str(path), input_sha256=digest(path),
                         trial_id=trial_id, attempt_id="one"))
        trials.append(dict(trial_id=trial_id, input_sha256=digest(path), costs=costs, max_attempts=1))
    manifest = dict(schema_version="2", evidence_mode="synthetic", jobs=jobs,
                    limits=dict(wall_seconds=120, memory_bytes=2147483648, cpu_threads=1),
                    budget=dict(budget_id="two-fold", limits={k:v*2 for k,v in costs.items()}, trials=trials))
    source = tmp_path/"campaign.json"
    atomic_json(source, manifest)
    return source


def test_two_folds_reuse_without_refit(tmp_path):
    source = setup(tmp_path)
    first = run(tmp_path, source, digest(source))
    second = run(tmp_path, source, digest(source), directory=first["directory"])
    assert first["distinct_outer_folds"] == 2
    assert all(j["executed_this_call"] for j in first["jobs"])
    assert not any(j["executed_this_call"] for j in second["jobs"])
    assert second["budget"]["reserved_counts"]["model_fits"] == 2


def test_failed_second_retains_first_and_charge(tmp_path):
    source = setup(tmp_path, bad_second=True)
    with pytest.raises(RuntimeError):
        run(tmp_path, source, digest(source))
    out = next((tmp_path/"output/experiments/s20_safe_v4/sources").glob("multifold-*"))
    cp = json.loads((out/"checkpoint.json").read_text())
    assert cp["status"] == "FAILED" and len(cp["completed_jobs"]) == 1
    assert cp["budget"]["reserved_counts"]["model_fits"] == 2
    with pytest.raises(ValueError, match="reconciliation"):
        run(tmp_path, source, digest(source), directory=out)


def test_overlap_rejected_before_fitting(tmp_path):
    source = setup(tmp_path)
    payload = json.loads(source.read_text())
    payload["jobs"][1]["input_path"] = payload["jobs"][0]["input_path"]
    payload["jobs"][1]["input_sha256"] = payload["jobs"][0]["input_sha256"]
    payload["budget"]["trials"][1]["input_sha256"] = payload["jobs"][0]["input_sha256"]
    atomic_json(source, payload)
    with pytest.raises(ValueError, match="overlap"):
        run(tmp_path, source, digest(source))


def same_fold_models(tmp_path):
    source=setup(tmp_path)
    manifest=json.loads(source.read_text())
    first=json.loads(Path(manifest['jobs'][0]['input_path']).read_text())
    for i,job in enumerate(manifest['jobs']):
        value=dict(first,model_family='logistic' if i==0 else 'shallow_tree')
        path=Path(job['input_path']);atomic_json(path,value)
        job['fold_id']='common';job['input_sha256']=digest(path)
        manifest['budget']['trials'][i]['input_sha256']=digest(path)
    atomic_json(source,manifest)
    return source


@pytest.mark.parametrize('field',['samples','features','fit_labels','calibration_labels','policy','target_id'])
def test_same_fold_scope_mismatch_rejected_before_budget(tmp_path,field):
    source=same_fold_models(tmp_path);manifest=json.loads(source.read_text())
    job=manifest['jobs'][1];path=Path(job['input_path']);value=json.loads(path.read_text())
    if field in ['samples','features','fit_labels','calibration_labels']:
        value[field]=value[field][:-1]
    elif field=='policy': value[field]['n_cap']=2
    else: value[field]='different-target'
    atomic_json(path,value);job['input_sha256']=digest(path)
    manifest['budget']['trials'][1]['input_sha256']=digest(path);atomic_json(source,manifest)
    with pytest.raises(ValueError,match='comparison scope mismatch'):
        run(tmp_path,source,digest(source))
    assert not (tmp_path/'output/experiments/s20_safe_v4/sources').exists()


def test_two_models_same_fold_not_two_independent_folds(tmp_path,monkeypatch):
    source=same_fold_models(tmp_path)
    report=run(tmp_path,source,digest(source))
    assert report['distinct_outer_folds']==1 and len(report['jobs'])==2
    assert set(report['same_fold_comparison_scope_sha256'])=={'common'}
    assert report['budget']['reserved_counts']['model_fits']==2
    import pandas as pd
    ledger=pd.read_parquet(report['outer_prediction_ledger']['path'])
    assert len(ledger)==4 and ledger.recorded_oof_provenance_valid.all()
    assert set(ledger.model_family)=={'logistic','shallow_tree'}
    assert not ledger.historical_availability_proven.any()
    replay=run(tmp_path,source,digest(source),directory=report['directory'])
    assert not any(j['executed_this_call'] for j in replay['jobs'])
    from research.s20_harness import baseline_outer_ledger
    receipt=Path(report['directory'])/'receipt-consumer-test.json'
    atomic_json(receipt,replay)
    checked=baseline_outer_ledger.verify_receipt(tmp_path,receipt,digest(receipt))
    assert checked['outer_ledger_reconstructed'] and checked['models_refit']==0
    corrupt=pd.read_parquet(replay['outer_prediction_ledger']['path'])
    corrupt.loc[0,'score']=.123456789
    corrupt.to_parquet(replay['outer_prediction_ledger']['path'],index=False)
    replay['outer_prediction_ledger']['sha256']=digest(Path(replay['outer_prediction_ledger']['path']))
    atomic_json(receipt,replay)
    with pytest.raises(AssertionError):
        baseline_outer_ledger.verify_receipt(tmp_path,receipt,digest(receipt))
    def fail(_): raise ValueError('injected aggregation failure')
    monkeypatch.setattr(baseline_outer_ledger,'collect',fail)
    with pytest.raises(ValueError,match='injected aggregation'):
        run(tmp_path,source,digest(source),directory=report['directory'])
    checkpoint=json.loads((Path(report['directory'])/'checkpoint.json').read_text())
    assert checkpoint['status']=='FAILED' and len(checkpoint['completed_jobs'])==2
    assert checkpoint['budget']['reserved_counts']['model_fits']==2
