import pandas as pd


def test_saved_replay_rejects_rehashed_path_and_input_changes(tmp_path,monkeypatch):
    import pytest
    from research.s20_harness import stock_session_compat as module
    from research.s20_harness.runtime import atomic_json,digest
    directory=tmp_path/'compat';directory.mkdir()
    reference=pd.DataFrame(dict(sample_id=['a'],compat_reason=['negative_flat'],different_horizon=[False],compat_payload_json=['{}']))
    raw=tmp_path/'raw.json';atomic_json(raw,{'value':1})
    pins={str(raw):digest(raw)}
    monkeypatch.setattr(module,'reconstruct',lambda root,parent,sha:(reference.copy(),pins))
    reference.to_parquet(directory/'labels.parquet',index=False)
    inputs=dict(partition=str(tmp_path/'parent'),partition_sha256='fixture',source_pins=pins.copy())
    atomic_json(directory/'inputs.json',inputs)
    def seal():
        atomic_json(directory/'summary.json',dict(rows=1,reason_counts={'negative_flat':1},different_horizon_rows=0,
            formal_training_authorized=False,artifacts={n:digest(directory/n) for n in ['labels.parquet','inputs.json']}))
        return digest(directory/'summary.json')
    assert module.replay(tmp_path,directory,seal())['current_code_paths_recomputed']
    bad=reference.copy();bad.loc[0,'compat_payload_json']='{"wrong":true}'
    bad.to_parquet(directory/'labels.parquet',index=False)
    with pytest.raises(AssertionError): module.replay(tmp_path,directory,seal())
    reference.to_parquet(directory/'labels.parquet',index=False)
    inputs['source_pins']={str(raw):'0'*64}
    atomic_json(directory/'inputs.json',inputs)
    with pytest.raises(ValueError,match='non-code inputs differ'): module.replay(tmp_path,directory,seal())
from research.s20_harness.stock_session_compat import materialize


def test_missing_market_day_extends_stock_horizon_without_dropping_candidate():
    dates=pd.bdate_range('2024-01-01',periods=23).strftime('%Y%m%d').tolist()
    candidates=pd.DataFrame([dict(sample_id=k,entity_id=k,signal_date=dates[0]) for k in ['halt','short']])
    quotes=pd.DataFrame([dict(entity_id=k,trade_date=d,open=10.,high=12.,low=9.5,close=10.)
        for k,ds in [('halt',[d for d in dates if d!=dates[4]]),('short',dates[:5])] for d in ds])
    result=materialize(candidates,quotes,dates)
    assert result.sample_id.tolist()==['halt','short']
    assert result.compat_horizon_end.iloc[0]==dates[21] and result.compat_class.iloc[0]==0
    assert pd.isna(result.compat_class.iloc[1])
    assert result.compat_reason.iloc[1]=='incomplete_stock_session_horizon'
    assert result.raw_calendar_class.isna().all()
    assert result.compat_label_realized.tolist()==[True,False]


def test_same_day_tie_is_unresolved_in_both_raw_tracks():
    dates=pd.bdate_range('2024-01-01',periods=21).strftime('%Y%m%d').tolist()
    c=pd.DataFrame([dict(sample_id='a',entity_id='a',signal_date=dates[0])])
    q=pd.DataFrame([dict(entity_id='a',trade_date=d,open=10.,high=12.,low=8.,close=10.) for d in dates])
    result=materialize(c,q,dates)
    assert result.compat_class.iloc[0]==result.raw_calendar_class.iloc[0]==-1
    assert not result.compat_label_realized.iloc[0]
