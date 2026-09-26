import json
from dataclasses import asdict
from pathlib import Path

import pytest

from research.s20_harness.portfolio_share_actions import record_share_claim,receive_share_claim,settle_fractional_share
from research.s20_harness.portfolio_book import create,accounting_snapshot
from research.s20_harness import portfolio_replay as replay
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_portfolio_share_actions import args,receipt_args,credit_value,share_mark
from tests.s20_harness.test_portfolio_book import buy,fill,CAL
from tests.s20_harness.test_portfolio_replay import FIXTURE


def settlement(**changes):
    a=dict(event_id='fraction',distribution_id='share1',quantity='0.5',net_cash=2.,
        settled_at='2024-07-29T10:00:00+08:00',received_at='2024-07-29T10:01:00+08:00',
        processing_at='2024-07-29T10:02:00+08:00',evidence_id='synthetic-fraction',evidence_sha256='d'*64)
    a.update(changes);return a


def credited():
    return receive_share_claim(record_share_claim(fill(buy(create(1000,CAL))),**args()),
        **receipt_args(credited_quantity=10))


def test_explicit_cash_resolves_only_fraction_and_preserves_conservation():
    b=credited();done=settle_fractional_share(b,**settlement())
    assert b.cash==299 and b.share_claims[0].fractional_settlement is None
    assert done.cash==301 and done.share_settlement_net_income==2
    assert accounting_snapshot(done)['conservation_error']==0
    assert credit_value(done,[share_mark()])['nav']==961
    assert credit_value(done,[])['nav'] is None
    assert done.positions==b.positions
    with pytest.raises(ValueError): settle_fractional_share(done,**settlement(event_id='again'))
    with pytest.raises(ValueError): receive_share_claim(done,**receipt_args(event_id='more',evidence_id='more'))


@pytest.mark.parametrize('changes',[
    dict(quantity='0.4'),dict(quantity='1'),dict(quantity='NaN'),dict(quantity='no'),dict(quantity=.5),
    dict(net_cash=-1),dict(net_cash=True),dict(net_cash=float('inf')),
    dict(evidence_id=''),dict(evidence_sha256='bad'),dict(distribution_id='missing'),
    dict(settled_at='2024-07-28T10:00:00+08:00'),dict(received_at='2024-07-29T09:00:00+08:00'),
    dict(received_at='2024-07-30T10:00:00+08:00')])
def test_invalid_settlement_rejected(changes):
    b=credited()
    with pytest.raises(ValueError): settle_fractional_share(b,**settlement(**changes))
    assert b.cash==299 and b.share_claims[0].fractional_settlement is None


def test_cannot_cash_settle_fraction_while_whole_share_residual_remains():
    b=record_share_claim(fill(buy(create(1000,CAL))),**args())
    with pytest.raises(ValueError,match='exact residual'): settle_fractional_share(b,**settlement())


def test_persisted_cash_in_lieu_after_original_sale_is_not_double_counted(tmp_path):
    plan=json.loads(FIXTURE.read_text());a=args(ts_code='SYNTHETIC')
    a['distribution']=asdict(a['distribution'])
    plan['events'].insert(3,dict(kind='share_record',args=a))
    plan['events'].insert(5,dict(kind='share_receipt',args=receipt_args(credited_quantity=10)))
    plan['events'].insert(-1,dict(kind='fractional_share_settlement',args=settlement(
        settled_at='2024-07-30T10:00:00+08:00',received_at='2024-07-30T10:01:00+08:00',
        processing_at='2024-07-30T10:02:00+08:00')))
    plan['events'][-1]['args']['share_marks']=[asdict(share_mark(
        price_at='2024-07-30T15:00:00+08:00',available_at='2024-07-30T15:01:00+08:00'))]
    path=tmp_path/'fraction.json';atomic_json(path,plan)
    report=replay.build(tmp_path,path,digest(path));out=Path(report['directory'])
    journal=json.loads((out/'journal.json').read_text())
    assert report['final_accounting']['cash']==918
    assert report['final_accounting']['nav'] is None  # credits still need explicit marks
    assert journal[-1]['report']['nav']==978
    assert journal[-1]['report']['unresolved_share_distribution_ids']==[]
    replay.verify(tmp_path,out,digest(out/'summary.json'))
