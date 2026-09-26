"""Scalar O-rule audit of pinned raw calendar paths; no economic/PIT acceptance."""
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .opportunity_reference import classify
from .runtime import atomic_json,digest,load_plan,now


def compare_paths(labels,quotes,window):
    if len(window)!=20 or window!=sorted(set(window)): raise ValueError('exact20 ordered market dates required')
    if labels.sample_id.isna().any() or labels.sample_id.duplicated().any(): raise ValueError('unique audit candidates required')
    if quotes.duplicated(['entity_id','trade_date']).any(): raise ValueError('duplicate audit quote')
    groups={k:g.set_index('trade_date') for k,g in quotes.groupby('entity_id')}
    rows=[]
    for r in labels.itertuples(index=False):
        group=groups.get(r.entity_id,pd.DataFrame(columns=['open','high','low','close']))
        path=group.reindex(window)[['open','high','low','close']].astype(float)
        entry=path.open.iloc[0]
        expected=None;reason=None
        if not np.isfinite(entry): reason='unfilled_or_missing_entry'
        elif (not np.isfinite(path.to_numpy()).all() or (path<=0).any().any()
              or (path.high<path.max(axis=1)).any() or (path.low>path.min(axis=1)).any()):
            reason='missing_or_invalid_quotes_no_imputation'
        else:
            expected=classify(entry,path.high,path.low);reason=expected['reason']
        payload=json.loads(r.raw_calendar_payload_json)
        differences=[]
        cls=None if pd.isna(r.raw_calendar_class) else int(r.raw_calendar_class)
        target=None if expected is None else expected['s20_class']
        if cls!=target or payload.get('o_class')!=target: differences.append('class')
        if r.raw_calendar_reason!=reason or payload.get('reason')!=reason: differences.append('reason')
        if expected is not None:
            for key in ['hit20_day','stop10_day','max_gain20','window_mae20','immediate','opportunity','down_risk']:
                if payload.get(key)!=expected[key]: differences.append(key)
        rows.append(dict(sample_id=r.sample_id,path_classifiable=expected is not None,
            expected_class=target,expected_reason=reason,matched=not differences,differences=json.dumps(differences)))
    return pd.DataFrame(rows)


def build(root,partition,partition_sha):
    from .label_partition_verify import verify
    from . import opportunity_reference
    root,partition=Path(root).resolve(),Path(partition).resolve()
    if digest(partition/'summary.json')!=partition_sha: raise ValueError('raw O audit partition pin mismatch')
    summary=load_plan(partition/'summary.json')
    pins={str(partition/'summary.json'):partition_sha}
    pins.update({str(partition/n):h for n,h in summary['artifacts'].items()})
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('raw O artifact changed')
    inputs=load_plan(partition/'inputs.json')
    pins.update({p:h for p,h in inputs['source_pins'].items() if Path(p).suffix!='.py'})
    for p in [Path(__file__).resolve(),Path(opportunity_reference.__file__).resolve()]: pins[str(p)]=digest(p)
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('raw O source changed')
    original,_=verify(inputs['partition'],inputs['partition_sha256'])
    labels=pd.read_parquet(partition/'labels.parquet')
    keys=['sample_id','entity_id','signal_date']
    if labels[keys].to_dict('records')!=original[keys].to_dict('records'): raise ValueError('raw O candidate mismatch')
    cal_path=next(Path(p) for p in pins if Path(p).name=='trade_cal.parquet')
    cal=pd.read_parquet(cal_path)
    dates=sorted(cal.loc[cal.exchange.eq('SSE') & cal.is_open.eq(1),'cal_date'].astype(str))
    start=dates.index(str(labels.signal_date.iloc[0]));window=dates[start+1:start+21]
    paths=[Path(p) for p in pins if Path(p).parent.name=='daily' and Path(p).stem in window]
    if len(paths)!=20: raise ValueError('missing raw O market partition')
    quotes=pd.concat([pd.read_parquet(p) for p in sorted(paths)],ignore_index=True)
    result=compare_paths(labels,quotes,window)
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('raw O input changed during audit')
    out=root/'output/experiments/s20_safe_v4/sources'/('raw-opportunity-audit-'+uuid.uuid4().hex);out.mkdir(parents=True)
    result.to_parquet(out/'comparisons.parquet',index=False)
    atomic_json(out/'inputs.json',dict(partition=str(partition),partition_sha256=partition_sha,source_pins=pins))
    report=dict(directory=str(out),at=now(),rows=len(result),matched=int(result.matched.sum()),
        classifiable_paths=int(result.path_classifiable.sum()),unknown_paths=int((~result.path_classifiable).sum()),
        all_candidates_retained=True,formal_training_authorized=False,
        scope='independent scalar raw calendar O arithmetic; not economic labels, fills or historical availability',
        artifacts={n:digest(out/n) for n in ['comparisons.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
