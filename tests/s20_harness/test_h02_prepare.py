import pandas as pd
import pytest

from research.s20_harness.h02_prepare import join_tracks


def frames():
    p=pd.DataFrame(dict(sample_id=['a','b'],entity_id=['A','B'],signal_date=['20240102']*2,p_class=['D',None]))
    o=p.copy()
    o['o_class']=pd.array([0,None],dtype='Int64')
    o['o_label_realized']=[True,False];o['o_reason']=['immediate20','unknown_event_terms']
    o['o_safe_opportunity']=pd.array([True,None],dtype='boolean');o['o_payload_json']=['{}','{}']
    return p,o


def test_o_success_does_not_replace_p_loss_or_unknown():
    p,o=frames();result=join_tracks(p,o)
    assert result.p_class.iloc[0]=='D' and pd.isna(result.p_class.iloc[1]) and result.o_safe_opportunity.iloc[0]
    assert pd.isna(result.o_safe_opportunity.iloc[1]) and len(result)==2
    pd.testing.assert_frame_equal(result[p.columns],p)


@pytest.mark.parametrize('kind',['order','remove','p_change'])
def test_mismatched_tracks_rejected(kind):
    p,o=frames()
    if kind=='order': o=o.iloc[::-1]
    if kind=='remove': o=o.iloc[:1]
    if kind=='p_change': o.loc[0,'p_class']='A'
    with pytest.raises(ValueError): join_tracks(p,o)


def test_compatibility_is_separate_and_identity_bound():
    p,o=frames();compat=p[['sample_id','entity_id','signal_date']].copy()
    compat['compat_class']=pd.array([3,None],dtype='Int64')
    compat['compat_reason']=['negative_flat','incomplete_stock_session_horizon']
    for name in ['compat_entry_date','compat_horizon_end']: compat[name]=['20240103',None]
    compat['compat_payload_json']=['{}','{}'];compat['different_horizon']=[False,False]
    compat['compat_label_realized']=[True,False]
    compat['raw_calendar_class']=pd.array([3,None],dtype='Int64')
    compat['raw_calendar_reason']=['negative_flat','unknown_path'];compat['raw_calendar_payload_json']=['{}','{}']
    result=join_tracks(p,o,compat)
    assert result.compat_class.iloc[0]==3 and result.o_class.iloc[0]==0 and result.p_class.iloc[0]=='D'
    assert pd.isna(result.compat_class.iloc[1])
    with pytest.raises(ValueError,match='compatibility candidate'): join_tracks(p,o,compat.iloc[::-1])
