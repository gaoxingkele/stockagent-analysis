import json
from pathlib import Path
from dataclasses import asdict,replace

import pytest

from research.s20_harness.portfolio_share_actions import allocate_share_basis,transfer_share_position
from research.s20_harness.portfolio_book import accounting_snapshot,reserve_exit,settle_portfolio_exit
from research.s20_harness.portfolio_valuation import value_portfolio
from research.s20_harness import portfolio_replay as replay
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_share_basis_allocation import credit,basis
from tests.s20_harness.test_portfolio_share_actions import args,receipt_args,share_mark
from tests.s20_harness.test_portfolio_valuation import mark
from tests.s20_harness.test_portfolio_book import POLICY
from tests.s20_harness.test_portfolio_replay import FIXTURE


def transfer(**changes):
    a=dict(event_id='transfer',distribution_id='share1',position_id='bonus',due_date='20240730',
        sellable_from='20240730',eligibility_available_at='2024-07-30T09:00:00+08:00',
        processing_at='2024-07-30T10:00:00+08:00',evidence_id='synthetic-eligibility',evidence_sha256='f'*64)
    a.update(changes);return a


def test_transfer_basis_once_then_full_exit_and_no_double_valuation():
    old=allocate_share_basis(credit(),**basis());b=transfer_share_position(old,**transfer())
    assert old.share_claims[0].transferred_position_id is None
    assert b.cash==old.cash and b.positions[-1].entry_cost==70.1
    assert accounting_snapshot(b)['remaining_cost_basis']==pytest.approx(701)
    assert accounting_snapshot(b)['credited_share_cost_basis']==0
    at='2024-07-30T15:00:00+08:00';known='2024-07-30T15:01:00+08:00'
    r=value_portfolio(b,[mark(price_at=at,available_at=known)],valuation_at=at,knowledge_at=known)
    assert r['nav']==959 and r['share_credit_valuations']==[]
    with pytest.raises(ValueError,match='non-credited'):
        value_portfolio(b,[],share_marks=[share_mark()],valuation_at=at,knowledge_at=known)
    b=reserve_exit(b,order_id='bonus-sell',position_id='bonus',policy=POLICY,
        submitted_at='2024-07-30T14:50:00+08:00')
    b=settle_portfolio_exit(b,event_id='bonus-sold',order_id='bonus-sell',
        processing_at='2024-07-30T15:02:00+08:00',outcome_at=at,received_at=known,
        filled_quantity=10,price=6.,fee=1.,reason='synthetic',evidence_id='synthetic-sale')
    assert b.cash==358 and b.positions[-1].remaining_quantity==0
    assert accounting_snapshot(b)['conservation_error']==pytest.approx(0)


@pytest.mark.parametrize('changes',[
    dict(sellable_from='20240729'),dict(due_date='20240729'),dict(position_id='buy1'),
    dict(evidence_sha256='bad'),dict(eligibility_available_at='2024-07-31T09:00:00+08:00')])
def test_invalid_transfer_rejected(changes):
    b=allocate_share_basis(credit(),**basis())
    with pytest.raises(ValueError): transfer_share_position(b,**transfer(**changes))
    assert b.share_claims[0].transferred_position_id is None


def test_missing_basis_and_duplicate_transfer_rejected():
    with pytest.raises(ValueError): transfer_share_position(credit(),**transfer())
    b=transfer_share_position(allocate_share_basis(credit(),**basis()),**transfer())
    with pytest.raises(ValueError): transfer_share_position(b,**transfer(event_id='again',position_id='other',evidence_id='other'))


@pytest.mark.parametrize('follow_on',[False,True])
def test_persisted_transfer_moves_credit_counts_to_positions(tmp_path,follow_on):
    plan=json.loads(FIXTURE.read_text());a=args(ts_code='SYNTHETIC')
    a['distribution']=asdict(replace(a['distribution'],bonus_per_share=.1))
    plan['events'].insert(3,dict(kind='share_record',args=a))
    plan['events'].insert(5,dict(kind='share_receipt',args=receipt_args(credited_quantity=10)))
    plan['events'].insert(-1,dict(kind='share_basis_allocation',args=basis(allocations={'buy':70.1})))
    plan['events'].insert(-1,dict(kind='share_position_transfer',args=transfer()))
    plan['events'][-1]['args']['marks']=[asdict(mark(ts_code='SYNTHETIC',price_at='2024-07-30T15:00:00+08:00',available_at='2024-07-30T15:01:00+08:00'))]
    if follow_on:
        dist=dict(event_id='next',record_date='20240730',ex_date='20240731',known_date='20240701',
            cash_per_share=.2,pay_date='20240801',beneficiary_scope='existing_shareholders_verified')
        plan['events'].insert(-1,dict(kind='follow_on_eligibility',args=dict(event_id='eligibility-next',
            source_distribution_id='share1',target_distribution=dist,eligible=True,
            available_at='2024-07-30T10:30:00+08:00',processing_at='2024-07-30T11:00:00+08:00',
            evidence_id='next-eligibility',evidence_sha256='1'*64)))
        plan['events'].append(dict(kind='cash_record',args=dict(event_id='next-record',ts_code='SYNTHETIC',
            distribution=dist,processing_at='2024-07-30T15:05:00+08:00',
            terms_available_at='2024-07-01T12:00:00+08:00',source_id='synthetic')))
    p=tmp_path/'transfer.json';atomic_json(p,plan)
    r=replay.build(tmp_path,p,digest(p));out=Path(r['directory'])
    assert r['terminal_held_units']==10 and r['terminal_unallocated_share_credits']==0
    assert r['unknown_valuation_events']==0
    if follow_on:
        checkpoint=json.loads((out/'checkpoint.json').read_text())
        assert checkpoint['book']['cash_claims'][-1]['record_quantity']==10
        assert checkpoint['book']['cash_claims'][-1]['gross_amount']==2
    replay.verify(tmp_path,out,digest(out/'summary.json'))
