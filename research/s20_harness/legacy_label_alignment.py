"""Same-code/date legacy-to-O/P label attribution, preserving unmatched rows."""
from pathlib import Path
import uuid
import pandas as pd
from .runtime import atomic_json,digest,load_plan,now


def align(current,legacy):
    keys=['trading_code','signal_date']
    old=legacy.rename(columns={'ts_code':'trading_code','trade_date':'signal_date',
        's20_class':'legacy_class','horizon_end_date':'legacy_horizon_end','entry_date':'legacy_entry_date'})
    dates=set(current.signal_date)
    old=old.loc[old.signal_date.isin(dates),keys+['legacy_class','legacy_horizon_end','legacy_entry_date']]
    if current.duplicated(keys).any() or old.duplicated(keys).any():
        raise ValueError('unique code/date labels required; aliases need explicit mapping')
    rows=current.merge(old,on=keys,how='outer',validate='one_to_one',indicator='alignment_status',sort=False)
    both=rows.alignment_status.eq('both')
    known=both & rows.legacy_class.ge(0) & rows.compat_class.ge(0)
    rows['legacy_compat_comparable']=known
    rows['legacy_compat_equal']=pd.array([bool(a==b) if ok else None for a,b,ok in
        zip(rows.legacy_class,rows.compat_class,known)],dtype='boolean')
    rows['legacy_compat_horizon_diff']=pd.array([a!=b if ok else None for a,b,ok in
        zip(rows.legacy_horizon_end,rows.compat_horizon_end,both)],dtype='boolean')
    table=pd.crosstab(rows.legacy_class.fillna(-99),rows.o_class.fillna(-99),dropna=False)
    report=dict(rows=len(rows),current_candidates=len(current),legacy_rows_on_dates=len(old),
        alignment_counts={str(k):int(v) for k,v in rows.alignment_status.value_counts().items()},
        comparable_legacy_compat=int(known.sum()),
        legacy_compat_mismatches=int((rows.legacy_compat_equal==False).sum()),
        legacy_compat_horizon_differences=int(rows.legacy_compat_horizon_diff.sum()),
        current_P_unknown=int((~current.p_class.isin(['A','B','C','D'])).sum()),
        current_O_unresolved=int((~current.o_label_realized).sum()),
        legacy_to_economic_O_counts={str(k):{str(j):int(v) for j,v in col.items()} for k,col in table.to_dict().items()},
        unmatched_rows_retained=True,model_performance_comparison=False,formal_H03_accepted=False)
    return rows,report


def build(root,preparation,summary_sha):
    root=Path(root).resolve();preparation=Path(preparation).resolve()
    legacy=root/'output/experiments/s20_v3/labels.parquet'
    if digest(preparation/'summary.json')!=summary_sha: raise ValueError('preparation pin mismatch')
    summary=load_plan(preparation/'summary.json')
    pins={str(preparation/'summary.json'):summary_sha,str(legacy):digest(legacy),str(Path(__file__)):digest(Path(__file__))}
    for name,sha in summary['artifacts'].items():
        path=(preparation/name).resolve()
        if not path.is_relative_to(preparation) or digest(path)!=sha: raise ValueError('preparation artifact mismatch')
        pins[str(path)]=sha
    rows,report=align(pd.read_parquet(preparation/'labels.parquet'),pd.read_parquet(legacy,
        columns=['ts_code','trade_date','s20_class','entry_date','horizon_end_date']))
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('alignment source changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('legacy-label-alignment-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    rows.to_parquet(out/'alignment.parquet',index=False)
    atomic_json(out/'inputs.json',pins)
    report.update(directory=str(out),at=now(),historical_availability_proven=False,
        artifacts={n:digest(out/n) for n in ['alignment.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
