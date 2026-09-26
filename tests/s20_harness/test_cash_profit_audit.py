import json
import pandas as pd
import pytest
from research.s20_harness.cash_profit_audit import compare
from research.s20_harness.cash_profit_reference import classify
from research.s20_harness.corporate_actions import Distribution


def test_unknown_not_passed_and_numerical_corruption_detected():
    window=pd.bdate_range('2024-01-01',periods=20).strftime('%Y%m%d').tolist()
    c=pd.DataFrame(dict(sample_id=['a','b'],entity_id=['a','b'],event_codes=[['a'],['b']]))
    payload=classify(window,*[[10.]*20]*4,[])
    labels=pd.DataFrame(dict(sample_id=['a','b'],p_class=[payload['p_class'],None],label_realized=[True,False],
        payload_json=[json.dumps(payload),'{}']))
    quotes=pd.DataFrame([dict(entity_id='a',trade_date=d,open=10.,high=10.,low=10.,close=10.) for d in window])
    dist=pd.DataFrame(columns=['ts_code','normalized_event_id','record_date'])
    result=compare(c,labels,quotes,window,dist,[],[])
    assert result.matched.iloc[0] and pd.isna(result.matched.iloc[1])
    payload['terminal_net']=.3;labels.loc[0,'payload_json']=json.dumps(payload)
    result=compare(c,labels,quotes,window,dist,[],[])
    assert not result.matched.iloc[0] and 'terminal_net' in result.differences.iloc[0]


@pytest.mark.parametrize('case,status',[
    ('unknown_event','unresolved_event_selection'),
    ('bonus','unsupported_bonus'),
    ('missing_quote','invalid_reference_path'),
])
def test_unsupported_or_unknown_reference_never_passes(case,status):
    window=pd.bdate_range('2024-01-02',periods=20).strftime('%Y%m%d').tolist()
    candidates=pd.DataFrame(dict(sample_id=['a'],entity_id=['a'],event_codes=[['a']]))
    payload=classify(window,*[[10.]*20]*4,[])
    labels=pd.DataFrame(dict(sample_id=['a'],p_class=[payload['p_class']],label_realized=[True],
        payload_json=[json.dumps(payload)]))
    quotes=pd.DataFrame([dict(entity_id='a',trade_date=d,open=10.,high=10.,low=10.,close=10.) for d in window])
    frame=pd.DataFrame(columns=['ts_code','normalized_event_id','record_date'])
    events=[];decisions=[]
    if case=='missing_quote':
        quotes=quotes.iloc[:-1]
    else:
        frame=pd.DataFrame([dict(ts_code='a',normalized_event_id='event',
            record_date=None if case=='unknown_event' else window[0])])
        decisions=[dict(normalized_event_id='event',accepted_for_accounting=case=='bonus')]
        if case=='bonus':
            events=[Distribution('event',window[0],window[1],'20240101',bonus_per_share=1.,
                bonus_list_date=window[2],beneficiary_scope='existing_shareholders_verified')]
    result=compare(candidates,labels,quotes,window,frame,events,decisions)
    assert result.sample_id.tolist()==['a']
    assert result.status.tolist()==[status]
    assert result.matched.isna().all()
