import json
import pandas as pd
from research.s20_harness.raw_opportunity_audit import compare_paths
from research.s20_harness.opportunity_reference import classify


def test_reference_audit_retains_unknown_and_detects_payload_change():
    dates=pd.bdate_range('2024-01-01',periods=20).strftime('%Y%m%d').tolist()
    payload=classify(10.,[12.]*20,[9.5]*20);payload['o_class']=payload.pop('s20_class')
    labels=pd.DataFrame(dict(sample_id=['a','b'],entity_id=['a','b'],raw_calendar_class=[0,None],
        raw_calendar_reason=['immediate20','unfilled_or_missing_entry'],
        raw_calendar_payload_json=[json.dumps(payload),json.dumps(dict(o_class=None,reason='unfilled_or_missing_entry'))]))
    quotes=pd.DataFrame([dict(entity_id='a',trade_date=d,open=10.,high=12.,low=9.5,close=10.) for d in dates])
    result=compare_paths(labels,quotes,dates)
    assert result.matched.all() and result.path_classifiable.tolist()==[True,False]
    payload['hit20_day']=2;labels.loc[0,'raw_calendar_payload_json']=json.dumps(payload)
    result=compare_paths(labels,quotes,dates)
    assert not result.matched.iloc[0] and result.matched.iloc[1]
