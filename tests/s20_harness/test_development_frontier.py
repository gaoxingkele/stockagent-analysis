import json
from pathlib import Path
import pytest

from research.s20_harness.development_frontier import compare,build,verify
from research.s20_harness.metrics import binary_bounds
from research.s20_harness.runtime import atomic_json,digest,load_plan


def row(job,safe,risk,fold='f',daily=None):
    return dict(job_id=job,fold_id=fold,scope_sha256='a'*64,selected=len(safe),
        daily_selected=daily or [dict(signal_date='20240503',count=len(safe))],
        safe_profit=binary_bounds(safe),down5=binary_bounds(risk))


def test_domination_tradeoff_tie_unknown_and_empty():
    rows=compare([row('best',[True],[False]),row('bad',[False],[True]),
                  row('unknown',[None],[None]),row('tie',[True],[False]),row('empty',[],[])])
    by={r['job_id']:r for r in rows}
    assert by['bad']['dominated_by']==['best','tie']
    assert by['best']['dominated_by']==[] and by['unknown']['dominated_by']==[]
    assert by['empty']['frontier_status']=='NO_SELECTION'
    trade=compare([row('safe',[True],[True]),row('quiet',[False],[False])])
    assert all(not r['dominated_by'] for r in trade)


def test_no_cross_fold_scope_or_daily_coverage_comparison():
    base=row('bad',[False],[True])
    variants=[row('other_fold',[True],[False],fold='other'),
        row('other_day',[True],[False],daily=[dict(signal_date='20240506',count=1)]),
        dict(row('other_scope',[True],[False]),scope_sha256='b'*64)]
    assert compare([base,*variants])[0]['comparable_jobs']==[]


def test_verified_owned_campaign_frontier_and_tamper(tmp_path,monkeypatch,capsys):
    from tests.s20_harness.test_joint_family_campaign import setup
    from research.s20_harness.multifold_run import run
    from research.s20_harness.campaign_evaluation import build as evaluate
    path=setup(tmp_path);campaign=run(tmp_path,path,digest(path))
    receipt=next(Path(campaign['directory']).glob('receipt-*.json'))
    import pandas as pd
    candidates=pd.read_parquet(Path(campaign['jobs'][0]['directory'])/'candidate_ledger.parquet')
    outcomes=tmp_path/'outcomes.json'
    atomic_json(outcomes,dict(target_id='P.joint.v4',evaluation_at='2024-06-01T00:00:00Z',
        outcomes=[dict(sample_id=sid,target='A',label_available_at='2024-05-29T00:00:00Z') for sid in candidates.sample_id]))
    result=evaluate(tmp_path,receipt,digest(receipt),[dict(fold_id='common',path=str(outcomes),sha256=digest(outcomes))])
    directory=Path(result['directory'])
    from research.s20_harness import cli
    monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    assert cli.main(['build-development-frontier','--directory',str(directory),
        '--summary-sha256',digest(directory/'summary.json')])==0
    response=json.loads(capsys.readouterr().out);assert response.pop('valid')
    report=response;out=Path(report['directory'])
    checked=verify(tmp_path,out,digest(out/'summary.json'))
    assert checked['frontier_recomputed'] and checked['models_refit']==0
    rows=json.loads((out/'frontier.json').read_text(encoding='utf-8'))
    assert len(rows)==2 and sum(r['candidates'] for r in rows)==4
    assert not report['finalists_selected'] and not report['risk10_evaluated']
    assert all(r['trial_id'] and r['input_sha256'] for r in rows)
    assert cli.main(['verify-development-frontier','--directory',str(out),
        '--summary-sha256',digest(out/'summary.json')])==0
    assert json.loads(capsys.readouterr().out)['frontier_recomputed']
    assert cli.main(['verify-development-frontier','--directory',str(out),
        '--summary-sha256','0'*64])==2
    assert json.loads(capsys.readouterr().out)['valid'] is False
    for mutation in [dict(comparison='cross-fold winner'),dict(finalists_selected=True),dict(unregistered_claim=True)]:
        atomic_json(out/'summary.json',dict(report,**mutation))
        with pytest.raises(ValueError):verify(tmp_path,out,digest(out/'summary.json'))
    atomic_json(out/'summary.json',report)
    rows[0]['frontier_status']='WINNER';atomic_json(out/'frontier.json',rows)
    report['artifacts']['frontier.json']=digest(out/'frontier.json');atomic_json(out/'summary.json',report)
    with pytest.raises(ValueError,match='reconstruction mismatch'):verify(tmp_path,out,digest(out/'summary.json'))
