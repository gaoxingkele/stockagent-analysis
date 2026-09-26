"""Calendar O-track diagnostics on the exact retained P-track candidate set."""
import json
from pathlib import Path
import uuid

import pandas as pd

from .event_index import EventIndex
from .execution import market_horizon
from .labels import label_o_track
from .runtime import atomic_json,digest,load_plan,now


def materialize(candidates,quotes,calendar,distributions,events,decisions):
    if candidates.sample_id.isna().any() or candidates.sample_id.duplicated().any():
        raise ValueError('unique candidate identities required')
    if quotes.duplicated(['entity_id','trade_date']).any(): raise ValueError('duplicate entity/date quotes')
    index=EventIndex(distributions,events,decisions)
    groups={k:g.assign(ts_code=k) for k,g in quotes.groupby('entity_id',sort=False)}
    empty=quotes.iloc[:0].assign(ts_code=pd.Series(dtype=str))
    rows=[]
    for r in candidates.itertuples(index=False):
        horizon=market_horizon(calendar,str(r.signal_date),horizon=20)
        selected=index.window(list(r.event_codes),horizon['entry_date'],horizon['horizon_end'])
        if selected['unresolved_event_ids']:
            payload=dict(o_class=None,reason='unknown_event_terms',unresolved_event_ids=selected['unresolved_event_ids'])
        else:
            payload=label_o_track(groups.get(r.entity_id,empty),calendar,str(r.signal_date),r.entity_id,
                mode='calendar_pit_v4',distributions=selected['events'])
        cls=payload.get('o_class')
        resolved=cls is not None and cls>=0
        rows.append(dict(sample_id=r.sample_id,entity_id=r.entity_id,signal_date=str(r.signal_date),
            o_class=cls,o_label_realized=resolved,o_reason=payload['reason'],
            o_safe_opportunity=None if not resolved else cls in (0,1),
            o_payload_json=json.dumps(payload,sort_keys=True),formal_training_authorized=False))
    result=pd.DataFrame(rows)
    result['o_class']=pd.array(result.o_class,dtype='Int64')
    result['o_safe_opportunity']=pd.array(result.o_safe_opportunity,dtype='boolean')
    return result


def reconstruct(root,partition,partition_sha):
    from .label_partition_verify import verify
    from .distribution_adapter import adapt
    root,partition=Path(root).resolve(),Path(partition).resolve()
    code_paths=[*Path(__file__).parent.glob('*.py'),root/'src/stockagent_analysis/s20_v3.py']
    code_pins={str(p.resolve()):digest(p) for p in code_paths}
    p_labels,validation=verify(partition,partition_sha)
    summary=load_plan(partition/'summary.json');sources=load_plan(partition/'inputs.json')
    def source(name):
        paths=[Path(p) for p in sources if Path(p).name==name]
        if len(paths)!=1: raise ValueError('unique pinned source required: '+name)
        return paths[0]
    candidates=pd.read_parquet(partition/'candidates.parquet')
    cal=pd.read_parquet(source('trade_cal.parquet'))
    calendar=sorted(cal.loc[cal.exchange.eq('SSE') & cal.is_open.eq(1),'cal_date'].astype(str))
    sz=sorted(cal.loc[cal.exchange.eq('SZSE') & cal.is_open.eq(1),'cal_date'].astype(str))
    if calendar!=sz: raise ValueError('market calendar mismatch')
    window=market_horizon(calendar,summary['signal_date'],horizon=20)['window']
    paths=[Path(p) for p in sources if Path(p).parent.name=='daily' and Path(p).stem in window]
    if len(paths)!=20 or {p.stem for p in paths}!=set(window): raise ValueError('incomplete opportunity window')
    quotes=pd.concat([pd.read_parquet(p) for p in sorted(paths)],ignore_index=True)
    distributions=pd.read_parquet(source('normalized_distributions.parquet'))
    review=Path(summary['review_path'])
    if sources.get(str(review))!=summary['review_sha256']: raise ValueError('unpinned review')
    events,decisions=adapt(distributions,load_plan(review)['reviews'],rate_policy='gross_reference_diagnostic',
        unit_reviews=load_plan(source('s20_v4_cash_unit_reviews.json'))['reviews'])
    result=materialize(candidates,quotes,calendar,distributions,events,decisions)
    if result.sample_id.tolist()!=p_labels.sample_id.tolist(): raise ValueError('O/P candidate mismatch')
    result['p_class']=p_labels.p_class.to_numpy()
    result['p_label_status']=p_labels.label_status.to_numpy()
    verify(partition,partition_sha)
    if any(digest(Path(p))!=h for p,h in code_pins.items()): raise ValueError('opportunity code changed')
    return result,validation,code_pins


def build(root,partition,partition_sha):
    root,partition=Path(root).resolve(),Path(partition).resolve()
    result,validation,code_pins=reconstruct(root,partition,partition_sha)
    out=root/'output/experiments/s20_safe_v4/sources'/('opportunity-partition-'+uuid.uuid4().hex);out.mkdir(parents=True)
    result.to_parquet(out/'labels.parquet',index=False)
    transfer=pd.crosstab(result.o_reason,result.p_class.fillna('UNKNOWN'),dropna=False)
    transfer.to_csv(out/'op_transfer.csv')
    atomic_json(out/'inputs.json',dict(partition=str(partition),partition_sha256=partition_sha,
        code_pins=code_pins))
    report=dict(directory=str(out),at=now(),rows=len(result),all_candidates_retained=True,
        o_reason_counts=result.o_reason.value_counts().to_dict(),o_resolved_rows=int(result.o_label_realized.sum()),
        o_safe_rows=int(result.o_safe_opportunity.fillna(False).sum()),p_validation=validation,
        supplied_event_coverage_proven=False,historical_availability_proven=False,formal_training_authorized=False,
        artifacts={n:digest(out/n) for n in ['labels.parquet','op_transfer.csv','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report


def replay(root,directory,summary_sha):
    """Recompute saved O paths with current code, without writing a new partition."""
    root,directory=Path(root).resolve(),Path(directory).resolve()
    if not directory.is_relative_to(root): raise ValueError('O replay source escapes root')
    if digest(directory/'summary.json')!=summary_sha: raise ValueError('O summary pin mismatch')
    summary=load_plan(directory/'summary.json')
    if set(summary['artifacts'])!={'labels.parquet','op_transfer.csv','inputs.json'}:
        raise ValueError('O artifact set mismatch')
    def check():
        if digest(directory/'summary.json')!=summary_sha or any(
            digest(directory/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('O replay artifact changed')
    check()
    inputs=load_plan(directory/'inputs.json')
    rebuilt,validation,code_pins=reconstruct(root,inputs['partition'],inputs['partition_sha256'])
    pd.testing.assert_frame_equal(pd.read_parquet(directory/'labels.parquet'),rebuilt,check_exact=True)
    transfer=pd.crosstab(rebuilt.o_reason,rebuilt.p_class.fillna('UNKNOWN'),dropna=False)
    saved=pd.read_csv(directory/'op_transfer.csv',index_col=0)
    saved.columns.name=transfer.columns.name
    pd.testing.assert_frame_equal(saved,transfer,check_exact=True)
    counts=dict(rows=len(rebuilt),o_reason_counts=rebuilt.o_reason.value_counts().to_dict(),
        o_resolved_rows=int(rebuilt.o_label_realized.sum()),o_safe_rows=int(rebuilt.o_safe_opportunity.fillna(False).sum()))
    if any(summary.get(k)!=v for k,v in counts.items()) or summary.get('formal_training_authorized') is not False:
        raise ValueError('O replay summary mismatch')
    check()
    return dict(**counts,current_code_paths_recomputed=True,historical_code_replayed=False,
        input_economic_coverage_proven=False,historical_availability_proven=False,
        formal_training_authorized=False,code_pins=code_pins,p_validation=validation)
