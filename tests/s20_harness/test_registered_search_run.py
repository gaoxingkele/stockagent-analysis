from pathlib import Path
import sqlite3
import pytest
from research.s20_harness.registered_search_run import run,verify_receipt,writable_path
from research.s20_harness.search_registration import register
from research.s20_harness.runtime import digest,atomic_json,load_plan
from tests.s20_harness.test_selection_plan import setup


def test_owned_partial_resume_shared_budget(tmp_path):
    path=setup(tmp_path);sha=digest(path);registered=register(tmp_path,path,sha)
    first=run(tmp_path,registered['directory'],sha,max_new_jobs=1)
    assert first['status']=='PARTIAL_SYNTHETIC' and first['executed_this_call']==1
    assert first['budget']['reserved_counts']['model_fits']==1
    second=run(tmp_path,registered['directory'],sha,max_new_jobs=1)
    assert len(second['jobs'])==2 and second['executed_this_call']==1
    assert not second['jobs'][0]['executed_this_call']
    assert second['jobs'][0]['artifact']==first['jobs'][0]['artifact']
    assert second['budget']['reserved_counts']['model_fits']==2
    assert Path(first['receipt_path']).is_file() and Path(second['receipt_path']).is_file()
    assert not second['formal_H04_accepted']
    final=run(tmp_path,registered['directory'],sha,max_new_jobs=12)
    assert final['status']=='COMPLETED_SYNTHETIC' and final['executed_this_call']==10
    assert len(final['jobs'])==12 and final['budget']['reserved_counts']['model_fits']==12
    repeated=run(tmp_path,registered['directory'],sha,max_new_jobs=1)
    assert repeated['status']=='COMPLETED_SYNTHETIC' and repeated['executed_this_call']==0
    for receipt in [first,final,repeated]:
        checked=verify_receipt(tmp_path,receipt['receipt_path'],digest(Path(receipt['receipt_path'])))
        assert checked['search_execution_receipt_verified'] and checked['models_refit']==0
    # Historical partial evidence remains valid after later jobs complete.
    import json
    original=load_plan(first['receipt_path'])
    for change in [dict(status='COMPLETED_SYNTHETIC'),dict(registered_jobs=1),dict(executed_this_call=0),
                   dict(formal_H04_accepted=True),dict(jobs=[])]:
        target=writable_path(Path(registered['directory'])/'tampered.json')
        atomic_json(target,dict(original,**change))
        with pytest.raises(ValueError):verify_receipt(tmp_path,target,digest(target))
    changed=json.loads(json.dumps(original));changed['budget']['reserved_counts']['model_fits']=0
    atomic_json(target,changed)
    with pytest.raises(ValueError,match='budget count'):verify_receipt(tmp_path,target,digest(target))


def test_live_execution_lock_does_not_start_job(tmp_path):
    path=setup(tmp_path);sha=digest(path);registered=register(tmp_path,path,sha)
    with sqlite3.connect(Path(registered['directory'])/'execution_lock.sqlite') as db:
        db.execute('BEGIN IMMEDIATE')
        with pytest.raises(ValueError,match='already owned'):run(tmp_path,registered['directory'],sha)


def test_actual_failed_job_remains_charged_and_not_retried(tmp_path):
    path=setup(tmp_path);plan=load_plan(path)
    # Shared scope must remain identical across configurations for this fold/seed.
    for candidate in plan['candidates']:
        manifest_path=Path(candidate['manifest_path']);manifest=load_plan(manifest_path)
        for i,job in enumerate(manifest['jobs']):
            source=Path(job['input_path']);payload=load_plan(source)
            payload['calibration_labels'][1]['target']='A'
            atomic_json(source,payload);job['input_sha256']=digest(source)
            manifest['budget']['trials'][i]['input_sha256']=digest(source)
        atomic_json(manifest_path,manifest);candidate['manifest_sha256']=digest(manifest_path)
    atomic_json(path,plan);sha=digest(path);registered=register(tmp_path,path,sha)
    with pytest.raises(RuntimeError):run(tmp_path,registered['directory'],sha)
    with pytest.raises(ValueError,match='reconciliation'):run(tmp_path,registered['directory'],sha)
    cp=load_plan(Path(registered['directory'])/'execution_checkpoint.json')
    assert cp['budget']['reserved_counts']['model_fits']==1
    assert len(cp['budget']['attempts'])==1 and cp['budget']['attempts'][0]['state']=='FAILED'
