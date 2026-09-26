from dataclasses import replace
import pytest

from research.s20_harness.portfolio_book import create,reserve_exit,settle_portfolio_exit
from research.s20_harness.portfolio_share_actions import record_share_claim,allocate_share_basis,transfer_share_position,settle_fractional_share
from tests.s20_harness.test_portfolio_book import buy,fill,CAL,POLICY
from tests.s20_harness.test_portfolio_share_actions import args
from tests.s20_harness.test_share_basis_allocation import credit,basis
from tests.s20_harness.test_share_position_transfer import transfer
from tests.s20_harness.test_fractional_share_settlement import settlement


def sell(b,position='buy1',day='2024-07-29',quantity=100):
    b=reserve_exit(b,order_id='sell-'+position,position_id=position,policy=POLICY,submitted_at=day+'T14:50:00+08:00')
    return settle_portfolio_exit(b,event_id='sold-'+position,order_id='sell-'+position,
        outcome_at=day+'T15:00:00+08:00',received_at=day+'T15:01:00+08:00',processing_at=day+'T15:02:00+08:00',
        filled_quantity=quantity,price=6.,fee=1.,reason='synthetic',evidence_id='sale-'+position)


def rebuy(b,code='stock',day='20240730'):
    date=day[:4]+'-'+day[4:6]+'-'+day[6:]
    return buy(b,order_id='again',ts_code=code,buy_date=day,due_date=day,quantity=100,price_cap=6,fee_cap=1,
        submitted_at=date+'T09:20:00+08:00')


@pytest.mark.parametrize('credited',[False,True])
def test_untransferred_rights_block_duplicate_but_preserve_recommendation(credited):
    b=credit() if credited else record_share_claim(fill(buy(create(1000,CAL))),**args())
    b=sell(b);result=rebuy(b)
    assert result.journal[-1]['reason']=='already_held'
    assert result.journal[-1]['recommendation_kept'] and result.cash==b.cash and not result.buys
    assert rebuy(b,code='other').journal[-1]['reason']=='reserved'


def test_transferred_and_fully_sold_credits_release_name():
    b=sell(credit());b=allocate_share_basis(b,**basis());b=transfer_share_position(b,**transfer())
    assert rebuy(b,day='20240731').journal[-1]['reason']=='already_held'
    b=sell(b,position='bonus',day='2024-07-30',quantity=10)
    assert rebuy(b,day='20240731').journal[-1]['reason']=='reserved'


def test_pure_fraction_cash_settlement_releases_name():
    a=args();a['distribution']=replace(a['distribution'],bonus_per_share=.005)
    b=sell(record_share_claim(fill(buy(create(1000,CAL))),**a))
    assert rebuy(b).journal[-1]['reason']=='already_held'
    b=settle_fractional_share(b,**settlement(settled_at='2024-07-30T10:00:00+08:00',
        received_at='2024-07-30T10:01:00+08:00',processing_at='2024-07-30T10:02:00+08:00'))
    assert rebuy(b,day='20240731').journal[-1]['reason']=='reserved'
