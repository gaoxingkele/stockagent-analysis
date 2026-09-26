"""Vendor-factor price diagnostics, explicitly not cash/share economic returns."""
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from . import causal_price_features
from .adjustment_source import validate_factors
from .dependency_receipts import bind
from .runtime import atomic_json, digest, load_plan, now


def compute(candidates, quotes, factors, dates):
    q=quotes.loc[quotes.trade_date.astype(str).isin(dates)].copy()
    f=factors.loc[factors.trade_date.astype(str).isin(dates)].copy()
    q['trade_date']=q.trade_date.astype(str); f['trade_date']=f.trade_date.astype(str)
    if f.duplicated(['ts_code','trade_date']).any(): raise ValueError('duplicate factor stock-date')
    f=f.rename(columns={'ts_code':'trading_code'})
    q=q.merge(f[['trading_code','trade_date','adj_factor']],on=['trading_code','trade_date'],how='left',validate='many_to_one')
    factor=pd.to_numeric(q.adj_factor,errors='coerce')
    q['adj_factor']=factor.where(np.isfinite(factor)&factor.gt(0))
    anchor=q.loc[q.trade_date.eq(dates[-1])].set_index('entity_id').adj_factor
    scale=q.adj_factor/q.entity_id.map(anchor)
    for field in ['high','low','close','pre_close']:
        q[field]=q[field]*scale
    result=causal_price_features.compute(candidates,q,dates)
    result=result.rename(columns={f:f.replace('raw_','vendor_factor_',1) for f in causal_price_features.FEATURES if f.startswith('raw_')})
    result['price_window_status']=result.price_window_status.replace({'raw_price_diagnostic':'vendor_factor_price_diagnostic'})
    return result


def build(root, raw_directory, raw_sha, factor_source):
    root, raw_directory, factor_source=Path(root).resolve(),Path(raw_directory).resolve(),Path(factor_source).resolve()
    if not raw_directory.is_relative_to(root) or not factor_source.is_relative_to(root): raise ValueError('sources escape workspace')
    if digest(raw_directory/'summary.json')!=raw_sha: raise ValueError('raw feature summary mismatch')
    raw=load_plan(raw_directory/'summary.json')
    if digest(raw_directory/'inputs.json')!=raw['inputs_sha256'] or digest(raw_directory/'features.parquet')!=raw['table_sha256']:
        raise ValueError('raw feature artifact changed')
    pins={Path(p):h for p,h in load_plan(raw_directory/'inputs.json').items()}
    pins.update({raw_directory/'summary.json':raw_sha,raw_directory/'inputs.json':raw['inputs_sha256'],raw_directory/'features.parquet':raw['table_sha256']})
    if any(digest(p)!=h for p,h in pins.items()): raise ValueError('raw feature source changed')
    dates=raw['window']
    partitions=[p for p in pins if p.suffix=='.parquet' and p.parent.name=='daily' and p.stem in dates]
    if {p.stem for p in partitions}!=set(dates) or len(partitions)!=21: raise ValueError('complete market partitions required')
    candidates_path=next(p for p in pins if p.name=='candidates.parquet')
    candidates=pd.read_parquet(candidates_path)
    quotes=pd.concat([pd.read_parquet(p) for p in sorted(partitions)],ignore_index=True)
    pd.testing.assert_frame_equal(causal_price_features.compute(candidates,quotes,dates),pd.read_parquet(raw_directory/'features.parquet'),check_exact=True)
    frames,bindings=[],[]
    for date in dates:
        receipt=factor_source/(date+'.json'); path=factor_source/(date+'.parquet')
        bindings.append(dict(sample_id='shared-window',role='feature',dependency_id=date,
            artifact_path=str(path),artifact_sha256=digest(path),receipt_path=str(receipt),receipt_sha256=digest(receipt)))
        frame=pd.read_parquet(path)
        check=validate_factors(frame,date,set(quotes.loc[quotes.trade_date.astype(str).eq(date),'trading_code']))
        if not check['valid']: raise ValueError('invalid factor schema')
        frames.append(frame)
    receipts,evidence=bind(root,pd.DataFrame(bindings))
    pins.update({Path(p):h for p,h in evidence['source_pins'].items()})
    result=compute(candidates,quotes,pd.concat(frames,ignore_index=True),dates)
    pins[Path(__file__).resolve()]=digest(Path(__file__))
    if any(digest(p)!=h for p,h in pins.items()): raise ValueError('factor feature sources changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('factor-features-'+uuid.uuid4().hex); out.mkdir(parents=True)
    result.to_parquet(out/'features.parquet',index=False)
    atomic_json(out/'inputs.json',{str(p):h for p,h in pins.items()})
    atomic_json(out/'factor_receipts.json',dict(bindings=bindings,evidence=evidence,receipts=receipts.to_dict('records')))
    report=dict(directory=str(out),at=now(),rows=len(result),window=dates,
        status_counts=result.price_window_status.value_counts().to_dict(),
        price_feature_rows=int(result.vendor_factor_return20.notna().sum()),
        raw_feature_rows=raw['nonmissing_by_feature']['raw_return20'],all_candidates_retained=True,
        economic_total_return_proven=False,historical_availability_proven=False,formal_training_authorized=False,
        factor_available_at=max(receipts.available_at),
        artifacts={n:digest(out/n) for n in ['features.parquet','inputs.json','factor_receipts.json']})
    atomic_json(out/'summary.json',report)
    return report
