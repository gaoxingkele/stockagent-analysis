"""Retained-row scalar P arithmetic audit under supplied cash-event selection."""
import json
import math
from pathlib import Path
import uuid

import pandas as pd

from .cash_profit_reference import classify
from .event_index import EventIndex
from .runtime import atomic_json,digest,load_plan,now


def compare(candidates,labels,quotes,window,distributions,events,decisions):
    if candidates.sample_id.tolist()!=labels.sample_id.tolist(): raise ValueError('cash audit identity mismatch')
    if candidates.sample_id.duplicated().any() or quotes.duplicated(['entity_id','trade_date']).any():
        raise ValueError('cash audit duplicate identity')
    groups={k:g.set_index('trade_date') for k,g in quotes.groupby('entity_id')}
    index=EventIndex(distributions,events,decisions);rows=[]
    for candidate,label in zip(candidates.itertuples(index=False),labels.itertuples(index=False)):
        row=dict(sample_id=candidate.sample_id,status='upstream_unresolved',matched=None,differences='[]')
        if label.label_realized:
            selected=index.window(list(candidate.event_codes),window[0],window[-1])
            if selected['unresolved_event_ids']: row['status']='unresolved_event_selection'
            elif any(e.bonus_per_share for e in selected['events']): row['status']='unsupported_bonus'
            else:
                raw=groups.get(candidate.entity_id,pd.DataFrame(columns=['open','high','low','close'])).reindex(window)
                try:
                    expected=classify(window,*[raw[k].tolist() for k in ['open','high','low','close']],selected['events'])
                except ValueError as exc:
                    row.update(status='invalid_reference_path',differences=json.dumps([str(exc)]))
                else:
                    payload=json.loads(label.payload_json);differences=[]
                    for key in ['p_class','up_event','b5','first_up_day','first_down_day','time_to_b5']:
                        if payload.get(key)!=expected[key]: differences.append(key)
                    for key in ['terminal_net','mae','max_drawdown']:
                        actual=payload.get(key)
                        if actual is None or not math.isclose(actual,expected[key],rel_tol=0,abs_tol=1e-12): differences.append(key)
                    if label.p_class!=expected['p_class']: differences.append('class_column')
                    row.update(status='compared',matched=not differences,differences=json.dumps(differences))
        rows.append(row)
    result=pd.DataFrame(rows);result['matched']=pd.array(result.matched,dtype='boolean')
    return result


def build(root,partition,partition_sha):
    from .label_partition_verify import verify
    from .distribution_adapter import adapt
    from . import cash_profit_reference,event_index,distribution_adapter
    root,partition=Path(root).resolve(),Path(partition).resolve()
    labels,_=verify(partition,partition_sha)
    summary=load_plan(partition/'summary.json');sources=load_plan(partition/'inputs.json')
    pins={p:h for p,h in sources.items() if Path(p).suffix!='.py'}
    for module in [cash_profit_reference,event_index,distribution_adapter]:
        path=Path(module.__file__).resolve();pins[str(path)]=digest(path)
    pins[str(Path(__file__).resolve())]=digest(Path(__file__))
    def source(name):
        paths=[Path(p) for p in sources if Path(p).name==name]
        if len(paths)!=1: raise ValueError('unique cash audit source required')
        return paths[0]
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('cash audit source changed')
    candidates=pd.read_parquet(partition/'candidates.parquet')
    cal=pd.read_parquet(source('trade_cal.parquet'))
    dates=sorted(cal.loc[cal.exchange.eq('SSE') & cal.is_open.eq(1),'cal_date'].astype(str))
    start=dates.index(str(candidates.signal_date.iloc[0]));window=dates[start+1:start+21]
    paths=[Path(p) for p in pins if Path(p).parent.name=='daily' and Path(p).stem in window]
    if len(window)!=20 or len(paths)!=20: raise ValueError('cash audit requires20market sessions')
    quotes=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    distributions=pd.read_parquet(source('normalized_distributions.parquet'))
    review=Path(summary['review_path'])
    if pins.get(str(review))!=summary['review_sha256']: raise ValueError('cash review unbound')
    events,decisions=adapt(distributions,load_plan(review)['reviews'],rate_policy='gross_reference_diagnostic',
        unit_reviews=load_plan(source('s20_v4_cash_unit_reviews.json'))['reviews'])
    result=compare(candidates,labels,quotes,window,distributions,events,decisions)
    verify(partition,partition_sha)
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('cash audit sources changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('cash-profit-audit-'+uuid.uuid4().hex);out.mkdir(parents=True)
    result.to_parquet(out/'comparisons.parquet',index=False)
    atomic_json(out/'inputs.json',dict(partition=str(partition),partition_sha256=partition_sha,source_pins=pins))
    report=dict(directory=str(out),at=now(),rows=len(result),status_counts=result.status.value_counts().to_dict(),
        matched=int(result.matched.fillna(False).sum()),mismatched=int(result.matched.eq(False).fillna(False).sum()),
        unverified=int(result.matched.isna().sum()),numeric_abs_tolerance=1e-12,all_candidates_retained=True,
        event_selection_independently_verified=False,formal_training_authorized=False,
        artifacts={n:digest(out/n) for n in ['comparisons.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
