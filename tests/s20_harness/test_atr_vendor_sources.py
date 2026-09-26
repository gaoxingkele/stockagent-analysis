import json
from pathlib import Path
import pandas as pd
import pytest

from research.s20_harness.atr_control_sources import derive
from research.s20_harness.runtime import atomic_json,digest


def fixture(tmp_path):
    dates=pd.bdate_range(end='2024-05-03',periods=15).strftime('%Y%m%d').tolist()
    daily=pd.DataFrame([dict(ts_code='000001.SZ',trade_date=d,high=10.2 if i<7 else 5.1,
        low=9.8 if i<7 else 4.9,close=10. if i<7 else 5.,amount=2000.) for i,d in enumerate(dates)])
    factors=pd.DataFrame(dict(ts_code='000001.SZ',trade_date=dates,adj_factor=[1.]*7+[2.]*8))
    entry=dict(sample_id='a',trading_code='000001.SZ')
    for name,table,hour in [('daily',daily,16),('factors',factors,17)]:
        path=tmp_path/(name+'.parquet');table.to_parquet(path,index=False)
        receipt=tmp_path/(name+'.json')
        atomic_json(receipt,dict(file=path.name,sha256=digest(path),
            requested_at='2024-05-03T15:00:00+08:00',received_at=f'2024-05-03T{hour}:00:00+08:00'))
        entry[name]=dict(artifact_path=str(path),artifact_sha256=digest(path),receipt_path=str(receipt),receipt_sha256=digest(receipt))
    samples=pd.DataFrame([dict(sample_id='a',entity_id='stable-A',signal_date='20240503',prediction_at='2024-05-03T21:00:00+08:00')])
    return samples,dict(method='sma_true_range_over_close',period=14,calendar=dates,vendor_bindings=[entry])


def repin(spec,name,mutate):
    binding=spec['vendor_bindings'][0][name];path=Path(binding['artifact_path'])
    frame=mutate(pd.read_parquet(path));frame.to_parquet(path,index=False)
    binding['artifact_sha256']=digest(path)
    receipt=Path(binding['receipt_path']);value=json.loads(receipt.read_text());value['sha256']=digest(path)
    atomic_json(receipt,value);binding['receipt_sha256']=digest(receipt)


def test_independent_receipts_split_math_and_turnover(tmp_path):
    samples,spec=fixture(tmp_path);controls,evidence=derive(tmp_path,samples,spec)
    assert controls.atr_fraction.iloc[0]==pytest.approx(.04)
    assert controls.traded_value_cny.iloc[0]==2000000.
    assert controls.available_at.iloc[0]=='2024-05-03T09:00:00+00:00'
    assert len(evidence['source_pins'])==4
    assert not evidence['external_timestamp_authenticity_proven']


def test_late_factor_receipt_withholds_even_when_daily_ready(tmp_path):
    samples,spec=fixture(tmp_path);samples['prediction_at']='2024-05-03T16:30:00+08:00'
    controls,evidence=derive(tmp_path,samples,spec)
    assert len(controls)==1 and controls.atr_fraction.isna().all()
    assert evidence['audit'][0]['reason']=='source_not_available_at_prediction'


def test_missing_factor_does_not_default_to_one(tmp_path):
    samples,spec=fixture(tmp_path);repin(spec,'factors',lambda f:f.iloc[:-1])
    controls,evidence=derive(tmp_path,samples,spec)
    assert controls.atr_fraction.isna().all()
    assert evidence['audit'][0]['reason']=='incomplete_market_session_window'


@pytest.mark.parametrize('kind',['duplicate','future','outcome','tamper'])
def test_invalid_vendor_data_rejected(tmp_path,kind):
    samples,spec=fixture(tmp_path)
    if kind=='duplicate':repin(spec,'factors',lambda f:pd.concat([f,f.iloc[:1]]))
    elif kind=='future':repin(spec,'daily',lambda f:f.assign(trade_date=['20240506']+f.trade_date.tolist()[1:]))
    elif kind=='outcome':repin(spec,'daily',lambda f:f.assign(target=True))
    else:atomic_json(Path(spec['vendor_bindings'][0]['factors']['receipt_path']),{})
    with pytest.raises(ValueError):derive(tmp_path,samples,spec)


def test_vendor_baseline_saved_replay(tmp_path):
    from tests.s20_harness.test_baseline_run import plan
    from research.s20_harness.baseline_run import build
    from research.s20_harness.baseline_replay import replay
    samples,spec=fixture(tmp_path);samples['sample_id']='outer-test0';samples['entity_id']='A'
    spec['vendor_bindings'][0]['sample_id']='outer-test0'
    value=plan()
    for field in ['samples','features']:value[field]=[r for r in value[field] if r['sample_id']!='outer-test1']
    for row in value['samples']:
        if row['sample_id']=='outer-test0':row['prediction_at']=samples.prediction_at.iloc[0]
    controls,_=derive(tmp_path,samples,spec)
    value.update(policy_control_sources=spec,policy_controls=controls.to_dict('records'))
    value['policy'].update(mode='atr_liquidity_control',max_atr_fraction=.05,min_traded_value_cny=1000000.)
    path=tmp_path/'input.json';atomic_json(path,value)
    result=build(tmp_path,path,digest(path));out=Path(result['directory'])
    assert result['selected']==1
    assert replay(tmp_path,out,digest(out/'summary.json'))['semantic_replay_performed']


def test_shared_vendor_files_read_once_and_isolate_security(tmp_path,monkeypatch):
    samples,spec=fixture(tmp_path)
    for name in ['daily','factors']:
        repin(spec,name,lambda f:pd.concat([f,f.assign(ts_code='000002.SZ')],ignore_index=True))
    samples=pd.concat([samples,samples.assign(sample_id='b',entity_id='stable-B')],ignore_index=True)
    spec['vendor_bindings'].append(dict(spec['vendor_bindings'][0],sample_id='b',trading_code='000002.SZ'))
    from research.s20_harness import atr_vendor_sources as module
    original=module.pd.read_parquet;reads=[]
    def counted(path,*a,**k):reads.append(str(path));return original(path,*a,**k)
    monkeypatch.setattr(module.pd,'read_parquet',counted)
    controls,_=derive(tmp_path,samples,spec)
    assert controls.sample_id.tolist()==['a','b'] and len(reads)==2
    assert controls.atr_fraction.tolist()==pytest.approx([.04,.04])


def test_cached_vendor_data_mutation_rejected_at_derivation_end(tmp_path,monkeypatch):
    samples,spec=fixture(tmp_path)
    from research.s20_harness import atr_vendor_sources as module
    original=module.pd.read_parquet
    daily=Path(spec['vendor_bindings'][0]['daily']['artifact_path'])
    def mutate(path,*a,**k):
        frame=original(path,*a,**k)
        if Path(path).name=='factors.parquet':atomic_json(daily,{'changed':True})
        return frame
    monkeypatch.setattr(module.pd,'read_parquet',mutate)
    with pytest.raises(ValueError,match='source changed during derivation'):derive(tmp_path,samples,spec)
