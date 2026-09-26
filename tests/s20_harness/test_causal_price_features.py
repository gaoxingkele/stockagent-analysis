import pandas as pd
import pytest

from research.s20_harness.causal_price_features import compute


def fixture():
    dates=pd.bdate_range('2024-01-01',periods=21).strftime('%Y%m%d').tolist()
    c=pd.DataFrame(dict(sample_id=['a'],entity_id=['A'],trading_code=['000001.SZ'],signal_date=[dates[-1]]))
    q=pd.DataFrame([dict(entity_id='A',trading_code='000001.SZ',trade_date=d,close=10.+i,
        high=11.+i,low=9.+i,pre_close=10.+max(0,i-1),vol=100.) for i,d in enumerate(dates)])
    return c,q,dates


def test_future_mutations_do_not_change_features():
    c,q,dates=fixture(); before=compute(c,q,dates)
    future=q.iloc[-1:].copy(); future['trade_date']='20990101'; future['close']=-999
    pd.testing.assert_frame_equal(before,compute(c,pd.concat([q,future]),dates))
    assert before.raw_return20.iloc[0]==2. and before.volume_ratio20.iloc[0]==1.


def test_missing_session_and_reference_jump_not_imputed():
    c,q,dates=fixture()
    missing=compute(c,q.drop(index=5),dates)
    assert missing.price_window_status.iloc[0]=='incomplete_market_window' and pd.isna(missing.raw_return20.iloc[0])
    q.loc[5,'pre_close']=1.
    jump=compute(c,q,dates)
    assert jump.price_window_status.iloc[0]=='unresolved_reference_discontinuity'
    assert pd.isna(jump.raw_return20.iloc[0]) and jump.volume_ratio20.iloc[0]==1.


def test_duplicate_session_rejected():
    c,q,dates=fixture()
    with pytest.raises(ValueError,match='duplicate'): compute(c,pd.concat([q,q.iloc[:1]]),dates)
