from dataclasses import asdict, replace

import pytest

from research.s20_harness.corporate_actions import Distribution
from research.s20_harness.portfolio_book import create, accounting_snapshot, reserve_exit, settle_portfolio_exit
from research.s20_harness.portfolio_share_actions import record_share_claim
from research.s20_harness.portfolio_share_actions import receive_share_claim
from research.s20_harness.portfolio_replay import step
from research.s20_harness.portfolio_valuation import value_portfolio, ShareCreditMark
from tests.s20_harness.test_portfolio_book import buy, fill, CAL, POLICY
from tests.s20_harness.test_portfolio_valuation import mark, value


def args(**changes):
    result = dict(event_id='share-record', ts_code='stock',
        distribution=Distribution('share1','20240702','20240729','20240701',
            bonus_per_share=.105, bonus_list_date='20240730',
            beneficiary_scope='existing_shareholders_verified'),
        processing_at='2024-07-02T15:02:00+08:00',
        terms_available_at='2024-07-01T12:00:00+08:00',
        source_id='synthetic', source_sha256='a'*64)
    result.update(changes)
    return result


def receipt_args(**changes):
    result=dict(event_id='credit',distribution_id='share1',credited_quantity=6,
        credited_at='2024-07-29T09:00:00+08:00',received_at='2024-07-29T09:01:00+08:00',
        processing_at='2024-07-29T09:02:00+08:00',evidence_id='synthetic-credit',evidence_sha256='b'*64)
    result.update(changes)
    return result


def share_mark(**changes):
    m=ShareCreditMark('share1','2024-07-29T15:00:00+08:00',
        '2024-07-29T15:01:00+08:00','observed',6.,'synthetic-credit-mark','c'*64)
    return replace(m,**changes)


def credit_value(b,sm):
    return value_portfolio(b,[mark()],valuation_at='2024-07-29T15:00:00+08:00',
        knowledge_at='2024-07-29T15:02:00+08:00',share_marks=sm)


def test_explicit_credit_marks_value_assets_without_inventing_cash_or_sale():
    a=args(); a['distribution']=replace(a['distribution'],bonus_per_share=.1)
    b=record_share_claim(fill(buy(create(1000,CAL))),**a)
    b=receive_share_claim(b,**receipt_args(credited_quantity=10))
    assert credit_value(b,[])['nav'] is None
    r=credit_value(b,[share_mark()])
    assert r['nav']==959 and r['known_share_credit_value']==60
    assert r['unresolved_share_distribution_ids']==[]
    assert r['share_credit_valuations'][0]['tradable_quantity'] is None
    assert b.cash==299 and b.positions[0].remaining_quantity==100


def test_partial_right_remains_unknown_despite_priced_credits():
    b=record_share_claim(fill(buy(create(1000,CAL))),**args())
    b=receive_share_claim(b,**receipt_args(credited_quantity=10))
    r=credit_value(b,[share_mark()])
    assert r['nav'] is None and r['known_component_value']==959
    assert r['unresolved_share_distribution_ids']==['share1']


@pytest.mark.parametrize('changes',[
    dict(distribution_id='wrong'),dict(source_sha256='bad'),dict(basis='per_original_share'),
    dict(price_at='2024-07-30T15:00:00+08:00'),dict(available_at='2024-07-30T15:00:00+08:00'),
    dict(status='missing',price=6.),dict(price=-1.)])
def test_invalid_credit_marks_reject(changes):
    b=receive_share_claim(record_share_claim(fill(buy(create(1000,CAL))),**args()),**receipt_args())
    with pytest.raises(ValueError): credit_value(b,[share_mark(**changes)])


@pytest.mark.parametrize('sm',[
    share_mark(status='suspended',price=None),share_mark(status='missing',price=None),
    share_mark(price_at='2024-07-26T15:00:00+08:00')])
def test_stale_or_unobserved_credit_marks_stay_unknown(sm):
    a=args();a['distribution']=replace(a['distribution'],bonus_per_share=.1)
    b=receive_share_claim(record_share_claim(fill(buy(create(1000,CAL))),**a),**receipt_args(credited_quantity=10))
    assert credit_value(b,[sm])['nav'] is None


def test_partial_credit_retains_residual_and_does_not_invent_sellability():
    b=record_share_claim(fill(buy(create(1000,CAL))),**args())
    first=receive_share_claim(b,**receipt_args())
    second=receive_share_claim(first,**receipt_args(event_id='credit2',evidence_id='second',credited_quantity=4))
    assert b.share_claims[0].receipts==()
    assert [r.quantity for r in second.share_claims[0].receipts]==[6,4]
    assert second.journal[-1]['contractual_residual']=='0.500'
    assert second.positions==b.positions and second.cash==b.cash
    assert second.journal[-1]['tradable_quantity'] is None
    assert value(second,[mark()])['nav'] is None
    assert accounting_snapshot(second)['conservation_error']==0
    with pytest.raises(ValueError,match='exceed'):
        receive_share_claim(second,**receipt_args(event_id='third',evidence_id='third',credited_quantity=1))
    with pytest.raises(ValueError,match='duplicate share receipt'):
        receive_share_claim(first,**receipt_args(event_id='different'))


@pytest.mark.parametrize('changes',[
    dict(credited_quantity=0),dict(credited_quantity=True),dict(credited_quantity=1.5),
    dict(credited_quantity=11),dict(evidence_id=' '),dict(evidence_sha256='bad'),
    dict(distribution_id='unknown'),dict(credited_at='2024-07-28T09:00:00+08:00'),
    dict(received_at='2024-07-29T08:00:00+08:00'),
    dict(received_at='2024-07-30T09:00:00+08:00')])
def test_invalid_credit_does_not_mutate_claim(changes):
    b=record_share_claim(fill(buy(create(1000,CAL))),**args())
    with pytest.raises(ValueError): receive_share_claim(b,**receipt_args(**changes))
    assert b.share_claims[0].receipts==()


@pytest.mark.parametrize('credited',[False,True])
@pytest.mark.parametrize('cash',[False,True])
def test_follow_on_distribution_cannot_silently_omit_share_assets(credited,cash):
    from research.s20_harness.portfolio_cash_actions import record_cash_claim
    b=record_share_claim(fill(buy(create(1000,CAL))),**args())
    if credited: b=receive_share_claim(b,**receipt_args())
    a=args(processing_at='2024-07-30T15:02:00+08:00',event_id='follow-on')
    a['distribution']=Distribution('next','20240730','20240731','20240701',
        cash_per_share=.2 if cash else 0,bonus_per_share=0 if cash else .1,
        pay_date='20240801' if cash else None,bonus_list_date=None if cash else '20240801',
        beneficiary_scope='existing_shareholders_verified')
    if cash: a.pop('source_sha256')
    with pytest.raises(ValueError,match='unresolved share eligibility'):
        (record_cash_claim if cash else record_share_claim)(b,**a)
    assert len(b.share_claims)==1 and b.cash_claims==()


def test_other_security_is_not_blocked_by_unresolved_share_claim():
    from research.s20_harness.portfolio_book import record_date_quantity
    import pandas as pd
    b=record_share_claim(fill(buy(create(1000,CAL))),**args())
    assert record_date_quantity(b,'other',pd.Timestamp('2024-07-30T15:02:00+08:00'))==0


def test_claim_preserves_fractional_entitlement_and_blocks_incomplete_nav():
    original=fill(buy(create(1000,CAL)))
    b=record_share_claim(original,**args())
    assert original.share_claims==()
    assert b.share_claims[0].contractual_quantity==10.5
    assert b.cash==299 and b.positions==original.positions
    assert accounting_snapshot(b)['conservation_error']==0
    report=value(b,[mark()])
    assert report['nav'] is None
    assert report['known_component_value']==899
    assert report['unresolved_share_distribution_ids']==['share1']
    assert b.journal[-1]['received_quantity'] is None


def test_no_automatic_delivery_after_listing_date_even_after_sale():
    b=record_share_claim(fill(buy(create(1000,CAL))),**args())
    b=reserve_exit(b,order_id='sell',position_id='buy1',policy=POLICY,
        submitted_at='2024-07-29T14:50:00+08:00')
    b=settle_portfolio_exit(b,event_id='sold',order_id='sell',
        processing_at='2024-07-29T15:02:00+08:00',
        outcome_at='2024-07-29T15:00:00+08:00',received_at='2024-07-29T15:01:00+08:00',
        filled_quantity=100,price=6.,fee=1.,reason='synthetic',evidence_id='sale')
    r=value_portfolio(b,[],valuation_at='2024-07-31T15:00:00+08:00',
        knowledge_at='2024-07-31T15:01:00+08:00')
    assert b.cash==898 and r['nav'] is None
    assert b.share_claims[0].record_quantity==100


@pytest.mark.parametrize('changes',[
    dict(source_id=' '),dict(source_sha256='bad'),dict(ts_code=''),
    dict(terms_available_at='2024-07-03T09:00:00+08:00'),
    dict(processing_at='2024-07-02T14:59:00+08:00'),
    dict(processing_at='2024-07-03T15:02:00+08:00')])
def test_invalid_claims_rejected(changes):
    with pytest.raises(ValueError):
        record_share_claim(fill(buy(create(1000,CAL))),**args(**changes))


def test_duplicate_and_unsettled_holdings_rejected_and_dispatch_works():
    a=args(); a['distribution']=asdict(a['distribution'])
    original=fill(buy(create(1000,CAL)))
    b=step(original,dict(kind='share_record',args=a))
    assert b.share_claims[0].contractual_quantity==10.5
    with pytest.raises(ValueError,match='duplicate share'):
        record_share_claim(b,**args(event_id='second'))
    with pytest.raises(ValueError,match='unsettled'):
        record_share_claim(buy(create(1000,CAL)),**args())


def test_before_ex_date_claim_does_not_double_count_embedded_right():
    b=record_share_claim(fill(buy(create(1000,CAL))),**args())
    at='2024-07-02T15:02:00+08:00'
    report=value_portfolio(b,[mark(price_at=at,available_at=at)],
        valuation_at=at,knowledge_at=at)
    assert report['nav']==899 and report['unresolved_share_distribution_ids']==[]


@pytest.mark.parametrize('priced',[False,True])
def test_persisted_replay_keeps_claim_after_sale_and_verifies(tmp_path,priced):
    import json
    from pathlib import Path
    from research.s20_harness import portfolio_replay as replay
    from research.s20_harness.runtime import atomic_json, digest
    from tests.s20_harness.test_portfolio_replay import FIXTURE
    plan=json.loads(FIXTURE.read_text())
    a=args(ts_code='SYNTHETIC'); a['distribution']=asdict(a['distribution'])
    plan['events'].insert(3,dict(kind='share_record',args=a))
    plan['events'].insert(5,dict(kind='share_receipt',args=receipt_args()))
    if priced:
        plan['events'][-1]['args']['share_marks']=[asdict(share_mark(
            price_at='2024-07-30T15:00:00+08:00',available_at='2024-07-30T15:01:00+08:00'))]
    path=tmp_path/'shares.json'; atomic_json(path,plan)
    result=replay.build(tmp_path,path,digest(path)); out=Path(result['directory'])
    checkpoint=json.loads((out/'checkpoint.json').read_text())
    assert checkpoint['book']['share_claims'][0]['contractual_quantity']==10.5
    assert checkpoint['book']['share_claims'][0]['receipts'][0]['quantity']==6
    assert result['terminal_held_units']==0 and result['unknown_valuation_events']==1
    assert result['terminal_unallocated_share_credits']==6
    journal=json.loads((out/'journal.json').read_text())
    assert journal[-1]['report']['nav'] is None
    assert journal[-1]['report']['known_share_credit_value']==(36 if priced else 0)
    assert journal[-1]['report']['unresolved_share_distribution_ids']==['share1']
    replay.verify(tmp_path,out,digest(out/'summary.json'))
