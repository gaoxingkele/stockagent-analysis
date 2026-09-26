import json
from pathlib import Path

import pytest

from research.s20_harness import portfolio_replay as replay
from research.s20_harness.runtime import atomic_json, digest


FIXTURE = Path(__file__).parent/'fixtures/portfolio_reference.json'
PENDING = Path(__file__).parent/'fixtures/portfolio_pending_exit.json'


def tax_plan():
    plan=json.loads(FIXTURE.read_text())
    plan['calendar']+=['20240731','20240801']
    plan['events'] += [
        dict(kind='dividend_tax_settlement',args=dict(event_id='tax-debit',distribution_id='dividend',
            tax_delta=2.,settled_at='2024-07-30T16:00:00+08:00',received_at='2024-07-31T09:00:00+08:00',
            processing_at='2024-07-31T09:01:00+08:00',evidence_id='synthetic-tax-debit')),
        dict(kind='valuation',args=dict(event_id='after-tax',valuation_at='2024-07-31T15:00:00+08:00',
            knowledge_at='2024-07-31T15:01:00+08:00',marks=[])),
        dict(kind='dividend_tax_settlement',args=dict(event_id='tax-refund',distribution_id='dividend',
            tax_delta=-1.,settled_at='2024-08-01T09:00:00+08:00',received_at='2024-08-01T09:01:00+08:00',
            processing_at='2024-08-01T09:02:00+08:00',evidence_id='synthetic-tax-refund')),
        dict(kind='valuation',args=dict(event_id='after-refund',valuation_at='2024-08-01T15:00:00+08:00',
            knowledge_at='2024-08-01T15:01:00+08:00',marks=[]))]
    return plan


def test_tax_receipts_replay_into_nav_without_rewriting_earlier_value(tmp_path):
    p=tmp_path/'tax.json';atomic_json(p,tax_plan())
    r=replay.build(tmp_path,p,digest(p));out=Path(r['directory'])
    journal=json.loads((out/'journal.json').read_text())
    assert [e['report']['nav'] for e in journal if e['kind']=='valuation']==[916,914,915]
    assert r['final_accounting']['cash']==915 and r['final_accounting']['distribution_tax_paid']==3
    assert all(s['conservation_error']==0 for s in json.loads((out/'accounting_snapshots.json').read_text()))
    assert not r['formal_H02_passed']
    tax=[e for e in journal if e['kind']=='dividend_tax_settlement']
    assert [e['cash_delta'] for e in tax]==[-2,1]
    assert all(not e['final_personal_tax_proven'] for e in tax)
    verified=replay.verify(tmp_path,out,digest(out/'summary.json'))
    assert verified['events_recomputed']==12 and not verified['formal_H02_passed']


@pytest.mark.parametrize('artifact',['journal.json','checkpoint.json','accounting_snapshots.json','summary.json'])
def test_saved_replay_rejects_rehashed_accounting_changes(tmp_path,artifact):
    p=tmp_path/'tax.json';atomic_json(p,tax_plan())
    r=replay.build(tmp_path,p,digest(p));out=Path(r['directory'])
    payload=json.loads((out/artifact).read_text())
    if artifact=='journal.json': payload[-1]['report']['nav']+=1
    elif artifact=='checkpoint.json': payload['book']['cash']+=1
    elif artifact=='accounting_snapshots.json': payload[-1]['cash']+=1
    else: payload['formal_H02_passed']=True
    atomic_json(out/artifact,payload)
    if artifact!='summary.json':
        r['artifacts'][artifact]=digest(out/artifact);atomic_json(out/'summary.json',r)
    with pytest.raises(ValueError,match='reconstruction mismatch'):
        replay.verify(tmp_path,out,digest(out/'summary.json'))


def test_late_tax_book_cannot_be_backdated_to_previous_nav(tmp_path):
    plan=tax_plan();plan['events']=plan['events'][:10]
    plan['events'][-1]['args']['valuation_at']='2024-07-30T15:00:00+08:00'
    p=tmp_path/'bad-tax.json';atomic_json(p,plan)
    with pytest.raises(ValueError,match='later processing'):replay.build(tmp_path,p,digest(p))
    out=next((tmp_path/'output').rglob('checkpoint.json'))
    state=json.loads(out.read_text())
    assert state['status']=='FAILED' and state['completed_events']==9 and state['book']['cash']==914
    earlier=[e for e in state['book']['journal'] if e['kind']=='valuation']
    assert earlier[0]['report']['nav']==916


def test_full_cash_trade_valuation_replay_saves_recoverable_evidence(tmp_path):
    r=replay.build(tmp_path,FIXTURE,digest(FIXTURE)); out=Path(r['directory'])
    assert r['completed_events']==8 and r['journal_rows']==8
    assert r['final_accounting']['cash']==916 and not r['formal_H02_passed']
    journal=json.loads((out/'journal.json').read_text())
    assert journal[-1]['report']['nav']==916
    for n,h in r['artifacts'].items():assert digest(out/n)==h
    checkpoint=json.loads((out/'checkpoint.json').read_text())
    assert len(checkpoint['book']['cash_claims'])==1 and not checkpoint['resume_supported']
    for name,h in json.loads((out/'inputs.json').read_text()).items():
        if name.endswith('.py'):assert digest(out/'code_blobs'/h)==h


@pytest.mark.parametrize('fault',['bad_fill','unknown_kind','duplicate'])
def test_failure_retains_last_good_state_without_summary_success(tmp_path,fault):
    p=tmp_path/'plan.json'; plan=json.loads(FIXTURE.read_text())
    if fault=='bad_fill':plan['events'][1]['args']['price']=99
    if fault=='unknown_kind':plan['events'][1]['kind']='unsupported'
    if fault=='duplicate':plan['events'][1]=plan['events'][0]
    atomic_json(p,plan)
    with pytest.raises(ValueError):replay.build(tmp_path,p,digest(p))
    runs=list((tmp_path/'output/experiments/s20_safe_v4/sources').glob('portfolio-replay-*'))
    assert len(runs)==1 and not (runs[0]/'summary.json').exists()
    state=json.loads((runs[0]/'checkpoint.json').read_text())
    assert state['status']=='FAILED' and state['completed_events']==1 and state['book']['cash']==199


def test_changed_input_stops_before_committing_next_event(tmp_path,monkeypatch):
    p=tmp_path/'plan.json'; atomic_json(p,json.loads(FIXTURE.read_text())); real=replay.step
    def mutate(book,packet):
        result=real(book,packet);atomic_json(p,{});return result
    monkeypatch.setattr(replay,'step',mutate)
    with pytest.raises(ValueError,match='source changed'):replay.build(tmp_path,p,digest(p))
    state=next((tmp_path/'output').rglob('checkpoint.json'))
    s=json.loads(state.read_text()); assert s['completed_events']==0 and s['book']['cash']==1000


def test_suspension_partial_exit_and_duplicate_signal_end_to_end(tmp_path):
    r=replay.build(tmp_path,PENDING,digest(PENDING)); out=Path(r['directory'])
    journal=json.loads((out/'journal.json').read_text()); snapshots=json.loads((out/'accounting_snapshots.json').read_text())
    assert len(journal)==12 and len(snapshots)==13
    assert snapshots[4]['cash']==299 and snapshots[4]['pending_exits']==1
    assert snapshots[8]['cash']==538 and snapshots[8]['remaining_cost_basis']==pytest.approx(420.6)
    assert journal[5]['reason']=='already_held' and journal[5]['recommendation_kept']
    assert [e['report']['nav'] for e in journal if e['kind']=='valuation']==[None,None,837]
    assert r['unknown_valuation_events']==2 and r['terminal_held_units']==0
    assert all(abs(s['conservation_error'])<1e-8 for s in snapshots)
    assert r['final_accounting']['realized_pnl']==-163


def test_truncated_stream_preserves_unresolved_positions_and_null_nav(tmp_path):
    p=tmp_path/'pending.json'; plan=json.loads(PENDING.read_text());plan['events']=plan['events'][:9];atomic_json(p,plan)
    r=replay.build(tmp_path,p,digest(p))
    assert r['completed_events']==9 and r['terminal_held_units']==60
    assert r['final_accounting']['nav'] is None and r['final_accounting']['pending_exits']==1
    assert r['event_stream_completion_is_not_position_resolution'] and not r['formal_H02_passed']
    out=Path(r['directory'])
    assert replay.verify(tmp_path,out,digest(out/'summary.json'))['events_recomputed']==9


def test_unreturned_exit_result_stays_in_checkpoint(tmp_path):
    p=tmp_path/'unreturned.json';plan=json.loads(PENDING.read_text());plan['events']=plan['events'][:3];atomic_json(p,plan)
    r=replay.build(tmp_path,p,digest(p))
    assert r['terminal_exit_orders']==1 and r['terminal_held_units']==100
    assert r['final_accounting']['cash']==299 and r['final_accounting']['nav'] is None
