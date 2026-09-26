"""Explain unmatched archived label rows without changing either universe."""
from pathlib import Path
import uuid
import pandas as pd
from .runtime import atomic_json,digest,load_plan,now


def build(root,alignment,summary_sha):
    root=Path(root).resolve();alignment=Path(alignment).resolve()
    if digest(alignment/'summary.json')!=summary_sha: raise ValueError('alignment pin mismatch')
    summary=load_plan(alignment/'summary.json')
    table_path=alignment/'alignment.parquet'
    if digest(table_path)!=summary['artifacts']['alignment.parquet']: raise ValueError('alignment table mismatch')
    rows=pd.read_parquet(table_path)
    missing=rows.loc[rows.alignment_status.eq('left_only')].copy()
    codes=set(missing.trading_code);dates={c:[] for c in codes}
    legacy_path=root/'output/experiments/s20_v3/labels.parquet'
    alias_path=root/'config/s20_v4_security_aliases.json'
    pins={str(p):digest(p) for p in [alignment/'summary.json',table_path,legacy_path,alias_path,Path(__file__)]}
    # Replay the archived source-end contract, not whatever latest date exists.
    for path in sorted((root/'output/tushare_cache/daily').glob('*.parquet')):
        if not str(missing.signal_date.min())<=path.stem<='20260911': continue
        before=digest(path)
        frame=pd.read_parquet(path,columns=['ts_code','trade_date'])
        frame=frame.loc[frame.ts_code.isin(codes)]
        if frame.duplicated(['ts_code','trade_date']).any(): raise ValueError('duplicate raw code/date')
        for code,group in frame.groupby('ts_code'): dates[code].extend(group.trade_date.astype(str))
        if digest(path)!=before: raise ValueError('quote source changed')
        pins[str(path)]=before
    findings=[]
    for row in missing.itertuples(index=False):
        future=[d for d in dates[row.trading_code] if d>row.signal_date]
        findings.append(dict(trading_code=row.trading_code,signal_date=row.signal_date,
            side='current_only',observed_future_quote_rows=len(future),
            reason='fewer_than_20_future_stock_sessions' if len(future)<20 else 'unexplained_requires_review'))
    legacy=pd.read_parquet(legacy_path)
    aliases=load_plan(alias_path)['aliases']
    compare=[c for c in legacy.columns if c not in ['ts_code','trade_date']]
    for row in rows.loc[rows.alignment_status.eq('right_only')].itertuples(index=False):
        reason='unexplained_requires_review'
        for alias in aliases:
            if row.trading_code not in [alias['old_code'],alias['new_code']]: continue
            expected=alias['old_code'] if row.signal_date<alias['effective_date'] else alias['new_code']
            a=legacy.loc[legacy.ts_code.eq(row.trading_code)&legacy.trade_date.eq(row.signal_date),compare].reset_index(drop=True)
            b=legacy.loc[legacy.ts_code.eq(expected)&legacy.trade_date.eq(row.signal_date),compare].reset_index(drop=True)
            if expected!=row.trading_code and len(a)==len(b)==1 and a.equals(b):
                reason='reviewed_alias_duplicate_identical_label_payload'
        findings.append(dict(trading_code=row.trading_code,signal_date=row.signal_date,side='legacy_only',
            observed_future_quote_rows=None,reason=reason))
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('coverage source changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('legacy-coverage-reasons-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    result=pd.DataFrame(findings);result.to_parquet(out/'reasons.parquet',index=False)
    atomic_json(out/'inputs.json',pins)
    report=dict(directory=str(out),at=now(),rows=len(result),reason_counts=result.reason.value_counts().to_dict(),
        original_rows_removed=0,delisting_cause_proven=False,formal_H01_H03_accepted=False,
        source_end='20260911',artifacts={n:digest(out/n) for n in ['reasons.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
