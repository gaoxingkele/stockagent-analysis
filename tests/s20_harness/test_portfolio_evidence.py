import json
from pathlib import Path
import pytest

from research.s20_harness.portfolio_evidence import bind
from research.s20_harness import portfolio_replay as replay
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_portfolio_replay import FIXTURE


def fixture(root):
    data=root/'evidence.json';atomic_json(data,{'synthetic':True})
    plan=json.loads(FIXTURE.read_text());rows=[]
    for i,event in enumerate(plan['events']):
        for key in ['source_id','evidence_id']:
            if key in event['args']:
                rows.append(dict(event_index=i,field='args/'+key,identity=event['args'][key],path='evidence.json',sha256=digest(data)))
    plan['evidence_bindings']=rows
    return plan,data


def test_all_declared_refs_bound_and_persisted_replay_rechecks_bytes(tmp_path):
    plan,data=fixture(tmp_path);p=tmp_path/'input-plan.json';atomic_json(p,plan)
    r=replay.build(tmp_path,p,digest(p));out=Path(r['directory'])
    assert r['declared_evidence_bindings_verified'] and r['declared_evidence_references']==4
    assert not r['independent_execution_evidence_verified']
    replay.verify(tmp_path,out,digest(out/'summary.json'))
    atomic_json(data,{'synthetic':False})
    with pytest.raises(ValueError,match='evidence bytes differ'): replay.verify(tmp_path,out,digest(out/'summary.json'))


@pytest.mark.parametrize('fault',['missing','extra','duplicate','identity','hash','escape','declared_hash'])
def test_invalid_or_incomplete_bindings_reject(tmp_path,fault):
    plan,_=fixture(tmp_path);rows=plan['evidence_bindings']
    if fault=='missing': rows.pop()
    if fault=='extra': rows.append({**rows[0],'event_index':999})
    if fault=='duplicate': rows.append(dict(rows[0]))
    if fault=='identity': rows[0]['identity']='wrong'
    if fault=='hash': rows[0]['sha256']='a'*64
    if fault=='escape': rows[0]['path']='../outside.json'
    if fault=='declared_hash': plan['events'][rows[0]['event_index']]['args']['evidence_sha256']='b'*64
    with pytest.raises(ValueError): bind(tmp_path,plan)


def test_change_during_event_does_not_commit_its_state(tmp_path,monkeypatch):
    plan,data=fixture(tmp_path);p=tmp_path/'input-plan.json';atomic_json(p,plan)
    real=replay.step
    def tamper(book,packet):
        result=real(book,packet);atomic_json(data,{'changed':True});return result
    monkeypatch.setattr(replay,'step',tamper)
    with pytest.raises(ValueError,match='replay source changed'): replay.build(tmp_path,p,digest(p))
    out=next((tmp_path/'output/experiments/s20_safe_v4/sources').glob('portfolio-replay-*'))
    checkpoint=json.loads((out/'checkpoint.json').read_text())
    assert checkpoint['completed_events']==0 and checkpoint['book']['cash']==1000


def test_nested_mark_references_are_required(tmp_path):
    plan,data=fixture(tmp_path)
    plan['events'][-1]['args']['marks']=[dict(source_id='price',source_sha256=digest(data))]
    with pytest.raises(ValueError,match='missing declared'): bind(tmp_path,plan)
    plan['evidence_bindings'].append(dict(event_index=7,field='args/marks/0/source_id',identity='price',path='evidence.json',sha256=digest(data)))
    _,r=bind(tmp_path,plan)
    assert r['declared_evidence_references']==5
