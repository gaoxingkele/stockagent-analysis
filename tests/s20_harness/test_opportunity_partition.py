import pandas as pd

from research.s20_harness.opportunity_partition import materialize


def test_safe_touch_survives_later_fall_and_unknown_rows_stay():
    dates=pd.bdate_range('2024-01-01',periods=21).strftime('%Y%m%d').tolist()
    ids=['touch','fallback','tie','missing','event']
    candidates=pd.DataFrame([dict(sample_id=k,entity_id=k,signal_date=dates[0],event_codes=[k]) for k in ids])
    quotes=pd.DataFrame([dict(entity_id=k,trade_date=d,open=10.,high=11.,low=9.5,close=10.)
        for k in ids if k!='missing' for d in dates])
    quotes.loc[quotes.entity_id.eq('touch') & quotes.trade_date.eq(dates[2]),'high']=12.
    quotes.loc[quotes.entity_id.eq('touch') & quotes.trade_date.eq(dates[10]),'low']=8.
    quotes.loc[quotes.entity_id.eq('fallback') & quotes.trade_date.eq(dates[2]),'high']=11.6
    tie=quotes.entity_id.eq('tie') & quotes.trade_date.eq(dates[2])
    quotes.loc[tie,['high','low']]=[12.,8.]
    distributions=pd.DataFrame([dict(ts_code='event',normalized_event_id='unresolved',record_date=dates[2])])
    result=materialize(candidates,quotes,dates,distributions,[],[])
    assert result.sample_id.tolist()==ids
    assert result.o_class.iloc[:3].tolist()==[0,1,-1]
    assert result.o_safe_opportunity.iloc[:2].tolist()==[True,True]
    assert result.o_safe_opportunity.iloc[2:].isna().all()
    assert result.o_reason.iloc[-1]=='unknown_event_terms'
    assert not result.formal_training_authorized.any()
