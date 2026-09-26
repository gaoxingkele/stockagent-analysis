"""Frozen v3 path formula on retained identities and actual stock sessions.

This reproduces the label formula, not the legacy model/cache or PIT universe.
"""
import json
from pathlib import Path
import uuid

import pandas as pd

from .labels import label_o_track
from .runtime import atomic_json,digest,load_plan,now


def materialize(candidates,quotes,calendar=None):
    if candidates.sample_id.isna().any() or candidates.sample_id.duplicated().any():
        raise ValueError('unique compatibility candidate identities required')
    if quotes.duplicated(['entity_id','trade_date']).any(): raise ValueError('duplicate compatibility quote')
    groups={k:g.assign(ts_code=k).sort_values('trade_date') for k,g in quotes.groupby('entity_id')}
    rows=[]
    for r in candidates.itertuples(index=False):
        stock=groups.get(r.entity_id)
        if stock is None or str(r.signal_date) not in stock.trade_date.astype(str).tolist():
            payload=dict(o_class=None,reason='missing_signal_session')
        elif len(stock.loc[stock.trade_date.astype(str).gt(str(r.signal_date))])<20:
            payload=dict(o_class=None,reason='incomplete_stock_session_horizon')
        else:
            payload=label_o_track(stock,None,str(r.signal_date),r.entity_id,mode='stock_session_v3_compat')
        row=dict(sample_id=r.sample_id,entity_id=r.entity_id,signal_date=str(r.signal_date),
            compat_class=payload.get('o_class'),compat_reason=payload['reason'],
            compat_entry_date=payload.get('entry_date'),compat_horizon_end=payload.get('horizon_end'),
            compat_payload_json=json.dumps(payload,sort_keys=True),formal_training_authorized=False)
        if calendar is not None:
            raw=label_o_track(stock if stock is not None else quotes.iloc[:0].assign(ts_code=pd.Series(dtype=str)),
                calendar,str(r.signal_date),r.entity_id,mode='calendar_pit_v4')
            row.update(raw_calendar_class=raw.get('o_class'),raw_calendar_reason=raw['reason'],
                raw_calendar_payload_json=json.dumps(raw,sort_keys=True))
        rows.append(row)
    result=pd.DataFrame(rows);result['compat_class']=pd.array(result.compat_class,dtype='Int64')
    result['compat_label_realized']=result.compat_class.fillna(-1).ge(0)
    if calendar is not None:
        result['raw_calendar_class']=pd.array(result.raw_calendar_class,dtype='Int64')
    return result


def reconstruct(root,partition,partition_sha):
    from .label_partition_verify import verify
    root,partition=Path(root).resolve(),Path(partition).resolve()
    labels,_=verify(partition,partition_sha)
    sources=load_plan(partition/'inputs.json')
    inventory_path=root/'config/s20_v4_data_sources.json'
    if digest(inventory_path)!=sources[str(inventory_path)]: raise ValueError('compatibility inventory changed')
    spec=next(s for s in load_plan(inventory_path)['sources'] if s['role']=='identity_aware_daily_panel')
    panel=root/spec['path']
    if digest(panel/'summary.json')!=spec['summary_sha256']: raise ValueError('identity summary changed')
    summary=load_plan(panel/'summary.json')
    if digest(panel/'inputs_outputs.json')!=summary['receipt_sha256']: raise ValueError('identity receipt changed')
    receipts=json.loads((panel/'inputs_outputs.json').read_text(encoding='utf-8'))
    candidates=pd.read_parquet(partition/'candidates.parquet')
    date=str(candidates.signal_date.iloc[0])
    if candidates.signal_date.nunique()!=1: raise ValueError('single-date compatibility input required')
    remaining=set(candidates.entity_id);counts={k:0 for k in remaining};frames=[]
    pins={str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}
    legacy=root/'src/stockagent_analysis/s20_v3.py';pins[str(legacy)]=digest(legacy)
    pins[str(inventory_path)]=sources[str(inventory_path)]
    pins[str(panel/'summary.json')]=spec['summary_sha256'];pins[str(panel/'inputs_outputs.json')]=summary['receipt_sha256']
    for item in sorted(receipts,key=lambda r:r['canonical']):
        path=(panel/item['canonical']).resolve()
        if not path.is_relative_to(panel.resolve()): raise ValueError('compatibility panel escape')
        if path.stem<date: continue
        if not remaining: break
        if digest(path)!=item['canonical_sha256']: raise ValueError('compatibility quote changed')
        pins[str(path)]=item['canonical_sha256']
        frame=pd.read_parquet(path)
        selected=frame.loc[frame.entity_id.isin(remaining)].copy()
        frames.append(selected)
        for entity in selected.entity_id:
            counts[entity]+=1
            if counts[entity]>=21: remaining.discard(entity)
    cal_paths=[Path(p) for p in sources if Path(p).name=='trade_cal.parquet']
    if len(cal_paths)!=1: raise ValueError('unique compatibility calendar required')
    cal_path=cal_paths[0];pins[str(cal_path)]=sources[str(cal_path)]
    if digest(cal_path)!=pins[str(cal_path)]: raise ValueError('compatibility calendar changed')
    cal=pd.read_parquet(cal_path)
    calendar=sorted(cal.loc[cal.exchange.eq('SSE') & cal.is_open.eq(1),'cal_date'].astype(str))
    sz=sorted(cal.loc[cal.exchange.eq('SZSE') & cal.is_open.eq(1),'cal_date'].astype(str))
    if calendar!=sz: raise ValueError('compatibility calendar mismatch')
    result=materialize(candidates,pd.concat(frames,ignore_index=True),calendar)
    result['calendar_horizon_end']=labels.horizon_end.to_numpy()
    result['different_horizon']=result.compat_horizon_end.notna() & result.compat_horizon_end.ne(result.calendar_horizon_end)
    verify(partition,partition_sha)
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('compatibility source changed')
    return result,pins


def build(root,partition,partition_sha):
    root,partition=Path(root).resolve(),Path(partition).resolve()
    result,pins=reconstruct(root,partition,partition_sha)
    out=root/'output/experiments/s20_safe_v4/sources'/('stock-compat-'+uuid.uuid4().hex);out.mkdir(parents=True)
    result.to_parquet(out/'labels.parquet',index=False)
    atomic_json(out/'inputs.json',dict(partition=str(partition),partition_sha256=partition_sha,source_pins=pins))
    report=dict(directory=str(out),at=now(),rows=len(result),all_candidates_retained=True,
        reason_counts=result.compat_reason.value_counts().to_dict(),different_horizon_rows=int(result.different_horizon.sum()),
        raw_stock_session_formula=True,legacy_model_or_cache_reproduced=False,formal_training_authorized=False,
        artifacts={n:digest(out/n) for n in ['labels.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report


def replay(root,directory,summary_sha):
    """Current-code compatibility formula replay, not reproduction of old models."""
    root,directory=Path(root).resolve(),Path(directory).resolve()
    if not directory.is_relative_to(root): raise ValueError('compatibility replay escapes root')
    if digest(directory/'summary.json')!=summary_sha: raise ValueError('compatibility summary pin mismatch')
    summary=load_plan(directory/'summary.json')
    if set(summary['artifacts'])!={'labels.parquet','inputs.json'}:
        raise ValueError('compatibility artifact set mismatch')
    def check():
        if digest(directory/'summary.json')!=summary_sha or any(
            digest(directory/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('compatibility replay artifact changed')
    check()
    inputs=load_plan(directory/'inputs.json')
    if not (root/inputs['partition']).resolve().is_relative_to(root):
        raise ValueError('compatibility parent escapes root')
    rebuilt,pins=reconstruct(root,inputs['partition'],inputs['partition_sha256'])
    data_pins=lambda values:{str(Path(p).resolve()):h for p,h in values.items() if Path(p).suffix!='.py'}
    if data_pins(pins)!=data_pins(inputs['source_pins']):
        raise ValueError('compatibility non-code inputs differ')
    pd.testing.assert_frame_equal(pd.read_parquet(directory/'labels.parquet'),rebuilt,check_exact=True)
    counts=dict(rows=len(rebuilt),reason_counts=rebuilt.compat_reason.value_counts().to_dict(),
                different_horizon_rows=int(rebuilt.different_horizon.sum()))
    if any(summary.get(k)!=v for k,v in counts.items()) or summary.get('formal_training_authorized') is not False:
        raise ValueError('compatibility replay summary mismatch')
    check()
    return dict(**counts,current_code_paths_recomputed=True,data_source_pins_equal=True,
                legacy_model_or_cache_reproduced=False,historical_availability_proven=False,
                formal_training_authorized=False,replay_code_sha256=digest(Path(__file__)))
