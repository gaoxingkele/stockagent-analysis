from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pytest

from research.s20_harness.search_registration import register,verify
from research.s20_harness.runtime import digest,atomic_json
from tests.s20_harness.test_selection_plan import setup


def test_register_reuse_and_read_only_verify(tmp_path):
    path=setup(tmp_path);sha=digest(path)
    first=register(tmp_path,path,sha);second=register(tmp_path,path,sha)
    assert first==second
    assert len(first['mapping'])==12 and len({r['shared_trial_id'] for r in first['mapping']})==12
    assert len(first['budget']['trials'])==12
    assert first['budget']['reserved_counts']['model_fits']==0
    assert not first['formal_training_authorized'] and not first['preregistration_before_training_proven']
    directory=Path(first['directory']);before={n:digest(directory/n) for n in ['registration.sqlite','budget.sqlite']}
    assert verify(tmp_path,directory,sha)==first
    assert before=={n:digest(directory/n) for n in before}
    with sqlite3.connect(directory/'registration.sqlite') as db:
        with pytest.raises(sqlite3.IntegrityError,match='immutable'):db.execute("UPDATE registration SET registered_at='old'")
        with pytest.raises(sqlite3.IntegrityError,match='immutable'):db.execute('DELETE FROM registration')


def test_concurrent_registration_has_one_identity(tmp_path):
    path=setup(tmp_path);sha=digest(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:register(tmp_path,path,sha),range(2)))
    assert results[0]==results[1]


def test_registration_preserves_shared_reservations(tmp_path):
    from research.s20_harness.search_registration import specification
    from research.s20_harness.trial_budget import Budget
    path=setup(tmp_path);sha=digest(path);first=register(tmp_path,path,sha)
    budget=Budget(first['shared_budget_path'],specification(tmp_path,path,sha)['budget_contract'])
    job=first['mapping'][0]
    budget.reserve(job['shared_trial_id'],job['attempt_id'],job['input_sha256'])
    budget.finish(job['shared_trial_id'],job['attempt_id'],'FAILED',dict(error='test'))
    repeated=register(tmp_path,path,sha)
    assert repeated['registered_at']==first['registered_at']
    assert repeated['budget']['reserved_counts']['model_fits']==1
    assert any(t['attempts'] for t in repeated['budget']['trials'])


def test_changed_input_cannot_reregister(tmp_path):
    path=setup(tmp_path);sha=digest(path);first=register(tmp_path,path,sha)
    atomic_json(tmp_path/'0-0-20.json',{})
    with pytest.raises(ValueError,match='input pin'):register(tmp_path,path,sha)
    with pytest.raises(ValueError,match='input pin'):verify(tmp_path,first['directory'],sha)


def test_resume_after_budget_initialization_failure(tmp_path,monkeypatch):
    from research.s20_harness import search_registration as module
    path=setup(tmp_path);sha=digest(path)
    with monkeypatch.context() as patch:
        def fail(*args,**kwargs):raise RuntimeError('injected initialization failure')
        patch.setattr(module,'Budget',fail)
        with pytest.raises(RuntimeError,match='injected'):register(tmp_path,path,sha)
    directory=tmp_path/'output/experiments/s20_safe_v4/sources'/('search-registration-'+sha)
    before=digest(directory/'registration.sqlite')
    result=register(tmp_path,path,sha)
    assert digest(directory/'registration.sqlite')==before
    assert result['budget']['reserved_counts']['model_fits']==0


def test_cli_registration_and_verification(tmp_path,monkeypatch,capsys):
    import json
    from research.s20_harness import cli
    path=setup(tmp_path);sha=digest(path);monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    assert cli.main(['register-search','--input',str(path),'--input-sha256',sha])==0
    result=json.loads(capsys.readouterr().out)
    assert result['valid'] and not result['shared_runner_connected']
    assert cli.main(['verify-search-registration','--directory',result['directory'],'--plan-sha256',sha])==0
    assert json.loads(capsys.readouterr().out)['local_registration_verified']
