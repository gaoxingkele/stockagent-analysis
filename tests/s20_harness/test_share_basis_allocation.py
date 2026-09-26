from dataclasses import replace,asdict
import json
from pathlib import Path

import pytest

from research.s20_harness.portfolio_book import create,accounting_snapshot,reserve_exit,settle_portfolio_exit
from research.s20_harness.portfolio_share_actions import record_share_claim,receive_share_claim,allocate_share_basis
from research.s20_harness.exit_book import ExitLot,BasisTransfer,allocated_exit_cost
from research.s20_harness import portfolio_replay as replay
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_portfolio_share_actions import args,receipt_args
from tests.s20_harness.test_portfolio_book import buy,fill,CAL,POLICY
from tests.s20_harness.test_portfolio_replay import FIXTURE


def credit():
    a=args();a['distribution']=replace(a['distribution'],bonus_per_share=.1)
    return receive_share_claim(record_share_claim(fill(buy(create(1000,CAL))),**a),
        **receipt_args(credited_quantity=10))


def basis(**changes):
    a=dict(event_id='basis',distribution_id='share1',allocations={'buy1':70.1},
        available_at='2024-07-30T09:00:00+08:00',processing_at='2024-07-30T09:01:00+08:00',
        evidence_id='synthetic-basis',evidence_sha256='e'*64)
    a.update(changes);return a


@pytest.mark.parametrize('sold',[0,40,100])
def test_basis_transfer_after_partial_or_full_sale_preserves_cash_and_past_events(sold):
    b=credit()
    if sold:
        b=reserve_exit(b,order_id='sell',position_id='buy1',policy=POLICY,
            submitted_at='2024-07-29T14:50:00+08:00')
        b=settle_portfolio_exit(b,event_id='sold',order_id='sell',
            processing_at='2024-07-29T15:02:00+08:00',outcome_at='2024-07-29T15:00:00+08:00',
            received_at='2024-07-29T15:01:00+08:00',filled_quantity=sold,price=6.,fee=1.,reason='synthetic',evidence_id='sale')
    before=asdict(b);done=allocate_share_basis(b,**basis())
    assert asdict(b)==before and done.journal[:-1]==b.journal
    assert done.cash==b.cash and done.share_claims[0].allocated_basis==70.1
    assert done.positions[0].entry_cost==pytest.approx(630.9)
    assert done.positions[0].realized_cost==pytest.approx(630.9*sold/100)
    assert done.journal[-1]['allocations'][0]['realized_pnl_adjustment']==pytest.approx(70.1*sold/100)
    assert accounting_snapshot(done)['conservation_error']==pytest.approx(0)
    with pytest.raises(ValueError): allocate_share_basis(done,**basis(event_id='again',evidence_id='another'))


def test_future_exits_preserve_pre_record_sold_cost():
    lot=ExitLot('lot','stock','20240701','20240729',100,60,940.,
        realized_cost=400.,basis_transfers=(BasisTransfer(60,60.),))
    assert allocated_exit_cost(lot,60)==400
    assert allocated_exit_cost(lot,30)==670
    assert allocated_exit_cost(lot,0)==940


@pytest.mark.parametrize('changes',[
    dict(allocations={'wrong':70.1}),dict(allocations={'buy1':702}),dict(allocations={'buy1':-1}),
    dict(allocations={'buy1':True}),dict(evidence_sha256='bad'),dict(evidence_id=''),
    dict(available_at='2024-07-31T09:00:00+08:00')])
def test_invalid_allocations_do_not_mutate(changes):
    b=credit()
    with pytest.raises(ValueError): allocate_share_basis(b,**basis(**changes))
    assert b.share_claims[0].allocated_basis is None and b.positions[0].entry_cost==701


def test_persisted_post_sale_basis_adjustment_replays(tmp_path):
    plan=json.loads(FIXTURE.read_text());a=args(ts_code='SYNTHETIC')
    a['distribution']=asdict(replace(a['distribution'],bonus_per_share=.1))
    plan['events'].insert(3,dict(kind='share_record',args=a))
    plan['events'].insert(5,dict(kind='share_receipt',args=receipt_args(credited_quantity=10)))
    plan['events'].insert(-1,dict(kind='share_basis_allocation',args=basis(allocations={'buy':70.1},
        processing_at='2024-07-30T10:00:00+08:00')))
    path=tmp_path/'basis.json';atomic_json(path,plan)
    result=replay.build(tmp_path,path,digest(path));out=Path(result['directory'])
    assert result['final_accounting']['credited_share_cost_basis']==70.1
    assert result['final_accounting']['cash']==916
    assert result['final_accounting']['conservation_error']==pytest.approx(0)
    replay.verify(tmp_path,out,digest(out/'summary.json'))
