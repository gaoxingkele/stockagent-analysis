import pandas as pd
import pytest

from research.s20_harness.factor_price_features import compute
from tests.s20_harness.test_causal_price_features import fixture


def factors(q):
    return pd.DataFrame(dict(ts_code=q.trading_code,trade_date=q.trade_date,adj_factor=1.))


def test_consistent_split_like_ratio_not_counted_as_loss():
    c,q,dates=fixture(); f=factors(q)
    q.loc[10:,['high','low','close','pre_close']]/=2
    f.loc[10:,'adj_factor']=2.
    result=compute(c,q,f,dates)
    assert result.vendor_factor_return20.iloc[0]==2.
    assert result.price_window_status.iloc[0]=='vendor_factor_price_diagnostic'


def test_factor_only_conflict_and_missing_factor_withheld():
    c,q,dates=fixture(); f=factors(q); f.loc[10:,'adj_factor']=2.
    result=compute(c,q,f,dates)
    assert pd.isna(result.vendor_factor_return20.iloc[0])
    assert result.price_window_status.iloc[0]=='unresolved_reference_discontinuity'
    f=factors(q).drop(index=10)
    assert compute(c,q,f,dates).price_window_status.iloc[0]=='incomplete_market_window'


def test_future_factor_cannot_change_current_features():
    c,q,dates=fixture(); f=factors(q)
    future=f.iloc[-1:].copy(); future['trade_date']='20990101'; future['adj_factor']=999.
    pd.testing.assert_frame_equal(compute(c,q,f,dates),compute(c,q,pd.concat([f,future]),dates))
