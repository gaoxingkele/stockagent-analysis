import json
from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness.atr_control_sources import derive
from research.s20_harness.runtime import atomic_json,digest


def fixture(tmp_path,unit='thousand_cny'):
    calendar=pd.bdate_range(end='2024-05-03',periods=15).strftime('%Y%m%d').tolist()
    data=dict(schema_version='atr-bars-1',entity_id='A',price_basis='declared_common_adjusted_cny',
        amount_unit=unit,bars=[dict(trade_date=d,high=10.2,low=9.8,close=10.,amount=2000.) for d in calendar])
    path=tmp_path/'bars.json';atomic_json(path,data)
    receipt=tmp_path/'receipt.json';atomic_json(receipt,dict(file=path.name,sha256=digest(path),
        requested_at='2024-05-03T15:00:00+08:00',received_at='2024-05-03T16:00:00+08:00'))
    binding=dict(sample_id='a',role='feature',dependency_id='completed-window',
        artifact_path=str(path),artifact_sha256=digest(path),receipt_path=str(receipt),receipt_sha256=digest(receipt))
    samples=pd.DataFrame([dict(sample_id='a',entity_id='A',signal_date='20240503',prediction_at='2024-05-03T21:00:00+08:00')])
    return samples,dict(method='sma_true_range_over_close',period=14,calendar=calendar,bindings=[binding])


@pytest.mark.parametrize('unit,factor',[('cny',1),('thousand_cny',1000),('ten_thousand_cny',10000)])
def test_bound_calculation_and_units(tmp_path,unit,factor):
    samples,spec=fixture(tmp_path,unit)
    controls,evidence=derive(tmp_path,samples,spec)
    assert controls.atr_fraction.iloc[0]==pytest.approx(.04)
    assert controls.traded_value_cny.iloc[0]==2000*factor
    assert evidence['local_receipt_artifact_bindings_verified']
    assert not evidence['price_adjustment_and_calendar_independently_verified']


def rewrite_source(spec,mutate):
    binding=spec['bindings'][0];path=Path(binding['artifact_path']);data=json.loads(path.read_text())
    mutate(data);atomic_json(path,data);binding['artifact_sha256']=digest(path)
    receipt=Path(binding['receipt_path']);r=json.loads(receipt.read_text());r['sha256']=digest(path)
    atomic_json(receipt,r);binding['receipt_sha256']=digest(receipt)


def test_gap_true_range_uses_previous_close_and_current_denominator(tmp_path):
    samples,spec=fixture(tmp_path)
    rewrite_source(spec,lambda d:d['bars'][-1].update(high=12.2,low=11.8,close=12.))
    controls,_=derive(tmp_path,samples,spec)
    assert controls.atr_fraction.iloc[0]==pytest.approx((13*.4+2.2)/14/12)


def test_raw_factor_packet_split_normalizes_before_atr(tmp_path):
    samples,spec=fixture(tmp_path)
    def change(d):
        d.update(schema_version='atr-raw-factor-bars-1',price_basis='unadjusted_cny')
        for i,b in enumerate(d['bars']):
            b['adj_factor']=1. if i<7 else 2.
            if i>=7:
                for field in ['high','low','close']:b[field]/=2.
    rewrite_source(spec,change)
    controls,evidence=derive(tmp_path,samples,spec)
    assert controls.atr_fraction.iloc[0]==pytest.approx(.04)
    assert controls.traded_value_cny.iloc[0]==2000000.  # Never scale turnover by price factor.
    assert not evidence['price_adjustment_and_calendar_independently_verified']


@pytest.mark.parametrize('bad',[0.,-1.,float('inf'),True,None])
def test_invalid_raw_factor_is_not_silently_filled(bad):
    from research.s20_harness.atr_control_sources import normalize
    payload=dict(schema_version='atr-raw-factor-bars-1',price_basis='unadjusted_cny',
        bars=[dict(trade_date='20240503',high=10.,low=9.,close=9.5,amount=2000.,adj_factor=bad)])
    with pytest.raises(ValueError,match='adjustment factors'):normalize(payload)


def test_missing_late_and_incomplete_retained(tmp_path):
    samples,spec=fixture(tmp_path)
    missing=dict(spec,bindings=[])
    controls,evidence=derive(tmp_path,samples,missing)
    assert len(controls)==1 and controls.atr_fraction.isna().all()
    assert evidence['audit'][0]['reason']=='missing_source'
    samples.loc[0,'prediction_at']='2024-05-03T16:00:00+08:00'
    controls,evidence=derive(tmp_path,samples,spec)
    assert controls.atr_fraction.isna().all()  # Equal receipt time is not before prediction.
    samples.loc[0,'prediction_at']='2024-05-03T21:00:00+08:00'
    rewrite_source(spec,lambda d:d['bars'].pop(3))
    controls,evidence=derive(tmp_path,samples,spec)
    assert evidence['audit'][0]['reason']=='incomplete_market_session_window'
    assert controls.traded_value_cny.isna().all()


@pytest.mark.parametrize('kind',['unit','entity','basis','negative','boolean','future','extra','tamper'])
def test_invalid_or_rehashed_semantic_source(tmp_path,kind):
    samples,spec=fixture(tmp_path)
    def change(d):
        if kind=='unit':d['amount_unit']='unknown'
        elif kind=='entity':d['entity_id']='B'
        elif kind=='basis':d['price_basis']='raw'
        elif kind=='negative':d['bars'][0]['amount']=-1
        elif kind=='boolean':d['bars'][0]['close']=True
        elif kind=='future':d['bars'][-1]['trade_date']='20240506';spec['calendar'].append('20240506')
        else:d['outcome']=True
    if kind=='tamper':atomic_json(Path(spec['bindings'][0]['artifact_path']),{})
    else:rewrite_source(spec,change)
    with pytest.raises(ValueError):derive(tmp_path,samples,spec)


def test_saved_baseline_replay_and_source_change(tmp_path):
    from tests.s20_harness.test_baseline_run import plan
    from research.s20_harness.baseline_run import build,verify
    from research.s20_harness.baseline_replay import replay
    samples,spec=fixture(tmp_path)
    value=plan();value['samples']=[s for s in value['samples'] if s['sample_id']!='outer-test1']
    value['features']=[s for s in value['features'] if s['sample_id']!='outer-test1']
    for s in value['samples']:
        if s['sample_id']=='outer-test0':s['prediction_at']=samples.prediction_at.iloc[0]
    samples['sample_id']='outer-test0';spec['bindings'][0]['sample_id']='outer-test0'
    controls,_=derive(tmp_path,samples,spec)
    value['policy_controls']=controls.to_dict('records');value['policy_control_sources']=spec
    value['policy'].update(mode='atr_liquidity_control',max_atr_fraction=.05,min_traded_value_cny=1000000.)
    path=tmp_path/'plan.json';atomic_json(path,value)
    result=build(tmp_path,path,digest(path));out=Path(result['directory'])
    assert result['outer_candidates']==1 and result['selected']==1
    assert replay(tmp_path,out,digest(out/'summary.json'))['semantic_replay_performed']
    atomic_json(Path(spec['bindings'][0]['artifact_path']),{})
    with pytest.raises(ValueError,match='hash mismatch'):verify(out,digest(out/'summary.json'))
