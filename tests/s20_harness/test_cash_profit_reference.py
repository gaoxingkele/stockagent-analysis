import pandas as pd
import pytest

from research.s20_harness.cash_profit_reference import classify
from research.s20_harness.corporate_actions import Distribution
from research.s20_harness.economic_window import value_window
from research.s20_harness.labels import label_p_track


@pytest.mark.parametrize('pay_date',['20240104','20240105','20240108'])
@pytest.mark.parametrize('terminal',[8.,9.,10.])
def test_cash_reference_matches_accounting_and_p_without_double_count(pay_date,terminal):
    dates=['20240102','20240103','20240104','20240105']
    event=Distribution('cash',dates[0],dates[1],'20240101',cash_per_share=1.,pay_date=pay_date,
        beneficiary_scope='existing_shareholders_verified')
    prices=[10.,9.,9.,terminal]
    raw=pd.DataFrame({k:prices for k in ['open','high','low','close']},index=dates)
    expected=classify(dates,prices,prices,prices,prices,[event])
    bars,state=value_window(raw,[event])
    for key in ['open','high','low','close','cash','receivable_cash']:
        assert bars[key].tolist()==[row[key] for row in expected['bars']]
    daily=raw.reset_index(names='trade_date');daily['ts_code']='stock'
    actual=label_p_track(daily,['20240101']+dates,'20240101','stock',horizon=4,distributions=[event])
    for key in ['p_class','up_event','b5','first_up_day','first_down_day','time_to_b5']:
        assert actual[key]==expected[key]
    for key in ['terminal_net','mae','max_drawdown']: assert actual[key]==pytest.approx(expected[key],abs=1e-14)


def test_record_after_window_not_added_and_bonus_not_silently_approximated():
    event=Distribution('future','20240105','20240108','20240101',cash_per_share=1.,pay_date='20240108',
        beneficiary_scope='existing_shareholders_verified')
    result=classify(['20240102'],[10.],[10.],[10.],[10.],[event],buy_cost=0,sell_cost=0)
    assert result['cash']==result['receivable_cash']==0 and result['terminal_net']==0
    bonus=Distribution('bonus','20240102','20240103','20240101',bonus_per_share=1.,bonus_list_date='20240104',
        beneficiary_scope='existing_shareholders_verified')
    with pytest.raises(ValueError,match='cash-only'):
        classify(['20240102'],[10.],[10.],[10.],[10.],[bonus])


def test_multiple_cash_events_and_zero_event_preserve_value():
    dates=['20240102','20240103','20240104','20240105'];prices=[10.,9.5,8.5,8.5]
    events=[Distribution('first',dates[0],dates[1],'20240101',cash_per_share=.5,pay_date=dates[3],
        beneficiary_scope='existing_shareholders_verified'),
        Distribution('second',dates[1],dates[2],'20240101',cash_per_share=1.,pay_date='20240108',
        beneficiary_scope='existing_shareholders_verified'),
        Distribution('zero',dates[0],dates[1],'20240101',beneficiary_scope='existing_shareholders_verified')]
    result=classify(dates,prices,prices,prices,prices,events,buy_cost=0,sell_cost=0)
    assert [r['close'] for r in result['bars']]==[10.]*4
    assert result['cash']==.5 and result['receivable_cash']==1. and result['terminal_net']==0.
    raw=pd.DataFrame({k:prices for k in ['open','high','low','close']},index=dates)
    bars,_=value_window(raw,events)
    assert bars.close.tolist()==[r['close'] for r in result['bars']]
