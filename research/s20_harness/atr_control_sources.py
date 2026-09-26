"""Receipt-bound diagnostic ATR controls; no inferred historical availability.

Inputs are explicitly normalized completed-bar snapshots, not arbitrary provider
payloads. Declared common adjusted prices still need independent action/PIT audit.
"""
from pathlib import Path
import math

import pandas as pd

from .dependency_receipts import bind, KEYS, BINDINGS
from .label_availability import _instant
from .runtime import digest, load_plan


def normalize(payload):
    """Normalize a receipt-bound raw/factor packet, without creating a receipt.

    The last row is the packet's as-of anchor, never a factor fetched from a
    later date. Corporate-action validity and raw packet lineage remain separate.
    """
    version=payload.get('schema_version')
    if version=='atr-bars-1':
        if payload.get('price_basis')!='declared_common_adjusted_cny':
            raise ValueError('control entity/common adjusted price basis mismatch')
        return payload
    if version!='atr-raw-factor-bars-1' or payload.get('price_basis')!='unadjusted_cny':
        raise ValueError('explicit raw-factor or common-adjusted snapshot required')
    bars=payload['bars']
    if (not isinstance(bars,list) or not 1<=len(bars)<=10000
            or any(set(b)!={'trade_date','high','low','close','amount','adj_factor'} for b in bars)):
        raise ValueError('exact bounded raw-factor bars required')
    for bar in bars:
        factor=bar['adj_factor']
        if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or factor<=0:
            raise ValueError('positive finite adjustment factors required')
        for field in ['high','low','close']:
            v=bar[field]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError('positive finite raw prices required')
    anchor=bars[-1]['adj_factor']
    normalized=[]
    for bar in bars:
        row={k:v for k,v in bar.items() if k!='adj_factor'}
        for field in ['high','low','close']:
            row[field]=bar[field]*(bar['adj_factor']/anchor)
        normalized.append(row)
    return dict(payload,schema_version='atr-bars-1',price_basis='declared_common_adjusted_cny',bars=normalized)


def derive(root, samples, specification):
    vendor='vendor_bindings' in specification
    source_key='vendor_bindings' if vendor else 'bindings'
    if set(specification) != {'method', 'period', 'calendar', source_key}:
        raise ValueError('exact ATR source specification required')
    if (specification['method'] != 'sma_true_range_over_close'
            or type(specification['period']) is not int or specification['period'] != 14):
        raise ValueError('fixed SMA true-range14 contract required')
    calendar=specification['calendar']
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError('unique chronological control calendar required')
    for date in calendar:
        if not isinstance(date,str) or len(date)!=8:
            raise ValueError('YYYYMMDD control calendar required')
        pd.to_datetime(date,format='%Y%m%d',errors='raise')
    columns={'sample_id','entity_id','signal_date','prediction_at'}
    if (not 1 <= len(samples) <= 100000 or samples.columns.duplicated().any() or set(samples.columns)!=columns
            or samples.isna().any().any() or samples.sample_id.duplicated().any()):
        raise ValueError('exact unique control sample metadata required')
    for col in ['sample_id','entity_id']:
        if any(not isinstance(v,str) or not v.strip() for v in samples[col]):
            raise ValueError('named control sample identities required')
    entries=specification[source_key]
    if not isinstance(entries,list) or len(entries)>100000:
        raise ValueError('bounded source bindings required')
    if not vendor and any(set(e)!=set(KEYS+BINDINGS) for e in entries):
        raise ValueError('exact source binding fields required')
    bindings=pd.DataFrame([] if vendor else entries,columns=KEYS+BINDINGS)
    if (bindings.sample_id.duplicated().any() or not set(bindings.sample_id).issubset(samples.sample_id)
            or not bindings.role.eq('feature').all()):
        raise ValueError('one feature snapshot per declared sample required')
    receipts,evidence=bind(root,bindings)
    clocks=receipts.set_index('sample_id')
    sources=bindings.set_index('sample_id')
    packets={}
    if vendor:
        from .atr_vendor_sources import prepare
        packets,evidence=prepare(root,samples,entries,calendar)
    controls=[];audit=[]
    for sample in samples.itertuples(index=False):
        at=_instant(sample.prediction_at)
        if sample.signal_date not in calendar or at.tz_convert('Asia/Shanghai').strftime('%Y%m%d')!=sample.signal_date:
            raise ValueError('control signal calendar/time mismatch')
        value=dict(sample_id=sample.sample_id,available_at=None,atr_fraction=None,traded_value_cny=None)
        reason='missing_source'
        if sample.sample_id in sources.index or sample.sample_id in packets:
            received=packets[sample.sample_id][0] if vendor else _instant(clocks.loc[sample.sample_id,'available_at'])
            reason='source_not_available_at_prediction'
            if received < at:
                if vendor:payload=packets[sample.sample_id][1]
                else:
                    path=(Path(root)/sources.loc[sample.sample_id,'artifact_path']).resolve()
                    payload=load_plan(path)
                if set(payload)!={'schema_version','entity_id','price_basis','amount_unit','bars'}:
                    raise ValueError('exact normalized completed-bar snapshot required')
                payload=normalize(payload)
                if payload['entity_id']!=sample.entity_id or payload['price_basis']!='declared_common_adjusted_cny':
                    raise ValueError('control entity/common adjusted price basis mismatch')
                factors={'cny':1.,'thousand_cny':1000.,'ten_thousand_cny':10000.}
                if payload['amount_unit'] not in factors:
                    raise ValueError('explicit supported turnover unit required')
                bars=payload['bars']
                if not isinstance(bars,list) or len(bars)>10000 or any(set(b)!={'trade_date','high','low','close','amount'} for b in bars):
                    raise ValueError('bounded exact completed-bar rows required')
                dates=[b['trade_date'] for b in bars]
                if dates!=sorted(set(dates)) or not set(dates).issubset(calendar):
                    raise ValueError('unique chronological source bars on calendar required')
                for bar in bars:
                    close_at=pd.Timestamp(bar['trade_date']).tz_localize('Asia/Shanghai')+pd.Timedelta(hours=15)
                    if close_at > received or bar['trade_date']>sample.signal_date:
                        raise ValueError('future/incomplete bar in historical snapshot')
                    for key in ['high','low','close','amount']:
                        v=bar[key]
                        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
                            raise ValueError('finite nonnegative numeric bar values required')
                    if not 0 < bar['low'] <= bar['close'] <= bar['high']:
                        raise ValueError('invalid common-basis OHLC')
                end=calendar.index(sample.signal_date)
                needed=calendar[max(0,end-14):end+1]
                by_date={b['trade_date']:b for b in bars}
                reason='incomplete_market_session_window'
                if len(needed)==15 and all(d in by_date for d in needed):
                    window=[by_date[d] for d in needed]
                    tr=[max(b['high']-b['low'],abs(b['high']-p['close']),abs(b['low']-p['close']))
                        for p,b in zip(window,window[1:])]
                    amount=window[-1]['amount']*factors[payload['amount_unit']]
                    atr=math.fsum(tr)/14/window[-1]['close']
                    if not math.isfinite(amount) or not math.isfinite(atr):
                        raise ValueError('derived control overflow')
                    value.update(available_at=received.isoformat(),atr_fraction=atr,traded_value_cny=amount)
                    reason='derived_from_bound_snapshot'
        controls.append(value);audit.append(dict(sample_id=sample.sample_id,reason=reason))
    for path,sha in evidence['source_pins'].items():
        if digest(Path(path))!=sha:raise ValueError('control source changed during derivation')
    frame=pd.DataFrame(controls,columns=['sample_id','available_at','atr_fraction','traded_value_cny'])
    for col in ['atr_fraction','traded_value_cny']:frame[col]=pd.to_numeric(frame[col]).astype(float)
    return frame,dict(method=specification['method'],period=14,liquidity_measure='signal_day_traded_value_cny',
        rows=len(frame),audit=audit,source_pins=evidence['source_pins'],all_candidates_retained=True,
        local_receipt_artifact_bindings_verified=True,external_timestamp_authenticity_proven=False,
        price_adjustment_and_calendar_independently_verified=False,formal_training_authorized=False)
