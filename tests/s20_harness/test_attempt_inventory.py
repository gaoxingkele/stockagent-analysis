import json
import sqlite3
import pytest

from research.s20_harness.attempt_inventory import inspect
from research.s20_harness.trial_budget import Budget
from research.s20_harness.runtime import digest
from tests.s20_harness.test_trial_budget import contract


def setup(tmp_path):
    spec=contract();spec['limits']={k:v*4 for k,v in spec['limits'].items()}
    spec['trials']=[dict(spec['trials'][0],trial_id=name) for name in ['new','reserved','failed','success']]
    budget=Budget(tmp_path/'budget.sqlite',spec)
    for name in ['reserved','failed','success']:budget.reserve(name,'one','a'*64)
    budget.finish('failed','one','FAILED',dict(type='RuntimeError',message='test'))
    budget.finish('success','one','SUCCEEDED_DIAGNOSTIC',dict(directory='not-verified'))
    return budget


def test_complete_read_only_inventory_no_liveness_or_success_claim(tmp_path,monkeypatch,capsys):
    from research.s20_harness import cli
    budget=setup(tmp_path);before=digest(budget.path)
    result=inspect(tmp_path,budget.path,budget.sha)
    by={t['trial_id']:t for t in result['trials']}
    assert by['new']['observation']=='NOT_RESERVED'
    assert by['reserved']['attempts'][0]['observation']=='RESERVED_LIVENESS_UNKNOWN'
    assert by['failed']['attempts'][0]['observation']=='RECORDED_FAILED'
    assert by['success']['attempts'][0]['observation']=='RECORDED_SUCCESS_UNVERIFIED'
    assert result['reserved_counts']['model_fits']==3
    assert not result['launch_authorized'] and not result['success_artifacts_verified']
    assert digest(budget.path)==before
    monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    assert cli.main(['inspect-attempts','--budget-path',str(budget.path),'--contract-sha256',budget.sha])==0
    assert json.loads(capsys.readouterr().out)['snapshot_sha256']==result['snapshot_sha256']
    assert digest(budget.path)==before


def test_missing_and_wrong_pin_do_not_initialize(tmp_path):
    path=tmp_path/'missing.sqlite'
    with pytest.raises(ValueError):inspect(tmp_path,path,'a'*64)
    assert not path.exists()
    budget=setup(tmp_path)
    with pytest.raises(ValueError,match='pin mismatch'):inspect(tmp_path,budget.path,'0'*64)


def test_failed_owned_campaign_inventory_keeps_success_and_failure(tmp_path):
    from tests.s20_harness.test_multifold_run import setup as campaign_setup
    from research.s20_harness.multifold_run import run
    from research.s20_harness.runtime import load_plan
    source=campaign_setup(tmp_path,bad_second=True)
    with pytest.raises(RuntimeError):run(tmp_path,source,digest(source))
    directory=next((tmp_path/'output/experiments/s20_safe_v4/sources').glob('multifold-*'))
    checkpoint=load_plan(directory/'checkpoint.json')
    path=directory/'budget.sqlite';before=digest(path)
    result=inspect(tmp_path,path,checkpoint['budget']['budget_sha256'])
    states=[t['attempts'][0]['observation'] for t in result['trials']]
    assert states==['RECORDED_SUCCESS_UNVERIFIED','RECORDED_FAILED']
    assert result['reserved_counts']['model_fits']==2
    assert not result['failed_attempts_refunded']
    assert digest(path)==before


@pytest.mark.parametrize('mutation',[
    "UPDATE attempts SET costs='{}' WHERE trial_id='failed'",
    "UPDATE attempts SET state='RUNNING' WHERE trial_id='reserved'",
    "UPDATE attempts SET result='{}' WHERE trial_id='reserved'",
])
def test_corrupt_database_rejected(tmp_path,mutation):
    budget=setup(tmp_path)
    with sqlite3.connect(budget.path) as db:
        db.execute('DROP TRIGGER immutable_reservation');db.execute(mutation)
    before=digest(budget.path)
    with pytest.raises(ValueError):inspect(tmp_path,budget.path,budget.sha)
    assert digest(budget.path)==before
