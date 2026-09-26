from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from research.s20_harness.portfolio_book import create, accounting_snapshot, append
from research.s20_harness.portfolio_share_actions import record_mixed_distribution
from research.s20_harness.portfolio_cash_actions import accrue_cash_claim, receive_cash_claim
from research.s20_harness import portfolio_replay as replay
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_portfolio_share_actions import args
from tests.s20_harness.test_portfolio_book import buy,fill,CAL
from tests.s20_harness.test_portfolio_replay import FIXTURE


def mixed_args():
    a=args();a['distribution']=replace(a['distribution'],cash_per_share=.2,pay_date='20240730')
    return a


def test_atomic_mixed_record_uses_identical_full_terms_and_snapshot():
    original=fill(buy(create(1000,CAL)));a=mixed_args()
    b=record_mixed_distribution(original,**a)
    assert original.cash_claims==original.share_claims==()
    assert b.cash_claims[0].distribution==b.share_claims[0].distribution==a['distribution']
    assert b.cash_claims[0].record_quantity==b.share_claims[0].record_quantity==100
    assert b.cash_claims[0].gross_amount==20 and b.share_claims[0].contractual_quantity==10.5
    assert b.positions==original.positions and b.cash==299
    b=accrue_cash_claim(b,event_id='accrue',distribution_id='share1',processing_at='2024-07-29T09:00:00+08:00')
    b=receive_cash_claim(b,event_id='paid',distribution_id='share1',cash_received=18.,tax_withheld=2.,
        evidence_id='synthetic',received_at='2024-07-30T09:00:00+08:00',processing_at='2024-07-30T09:01:00+08:00')
    assert b.cash==317 and b.share_claims[0].receipts==()
    assert accounting_snapshot(b)['conservation_error']==0


@pytest.mark.parametrize('fault',['hash','zero_cash','zero_share','duplicate','child_id_collision','overflow'])
def test_mixed_failure_never_returns_half_registered_state(fault):
    original=fill(buy(create(1000,CAL)));a=mixed_args()
    if fault=='hash': a['source_sha256']='bad'
    if fault=='zero_cash': a['distribution']=replace(a['distribution'],cash_per_share=0.)
    if fault=='zero_share': a['distribution']=replace(a['distribution'],bonus_per_share=0.)
    if fault=='overflow': a['distribution']=replace(a['distribution'],cash_per_share=1e308)
    if fault=='duplicate': original=record_mixed_distribution(original,**a)
    if fault=='child_id_collision':
        original=append(original,a['event_id']+'::share','2024-07-02T10:00:00+08:00',dict(kind='synthetic'))
    before=asdict(original)
    with pytest.raises(ValueError): record_mixed_distribution(original,**a)
    assert asdict(original)==before


def test_persisted_mixed_packet_has_two_journal_rows_and_replays(tmp_path):
    plan=json.loads(FIXTURE.read_text());a=mixed_args()
    a['ts_code']='SYNTHETIC';a['distribution']=asdict(replace(a['distribution'],event_id='dividend'))
    plan['events'][2]=dict(kind='mixed_distribution_record',args=a)
    path=tmp_path/'mixed.json';atomic_json(path,plan)
    result=replay.build(tmp_path,path,digest(path));out=Path(result['directory'])
    assert result['completed_events']==8 and result['journal_rows']==9
    checkpoint=json.loads((out/'checkpoint.json').read_text())
    assert len(checkpoint['book']['cash_claims'])==len(checkpoint['book']['share_claims'])==1
    assert result['unknown_valuation_events']==1
    replay.verify(tmp_path,out,digest(out/'summary.json'))


def test_second_leg_failure_preserves_last_good_replay_checkpoint(tmp_path):
    plan=json.loads(FIXTURE.read_text());a=mixed_args()
    a['ts_code']='SYNTHETIC';a['distribution']=asdict(replace(a['distribution'],cash_per_share=1e308))
    plan['events'][2]=dict(kind='mixed_distribution_record',args=a)
    path=tmp_path/'failure.json';atomic_json(path,plan)
    with pytest.raises(ValueError,match='mixed cash entitlement'):
        replay.build(tmp_path,path,digest(path))
    out=next((tmp_path/'output/experiments/s20_safe_v4/sources').glob('portfolio-replay-*'))
    checkpoint=json.loads((out/'checkpoint.json').read_text())
    assert checkpoint['completed_events']==2
    assert checkpoint['book']['cash_claims']==checkpoint['book']['share_claims']==[]
    assert checkpoint['book']['cash']==299 and not (out/'summary.json').exists()
