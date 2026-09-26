"""Fixed raw-price feature diagnostic on market sessions, not economic/PIT approval."""
from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd

from .runtime import digest, load_plan, atomic_json, now


FEATURES = ['raw_return20', 'raw_ma20_distance', 'raw_mean_tr14_pct', 'raw_return_vol20', 'volume_ratio20']


def compute(candidates, quotes, dates):
    if len(dates) != 21 or dates != sorted(set(dates)):
        raise ValueError('exact 21 ordered market sessions required')
    pd.to_datetime(dates, format='%Y%m%d', errors='raise')
    identity = ['sample_id','entity_id','trading_code','signal_date']
    if (not set(identity).issubset(candidates) or candidates[identity].isna().any().any()
            or candidates.sample_id.duplicated().any() or not candidates.signal_date.astype(str).eq(dates[-1]).all()):
        raise ValueError('valid single signal-date candidates required')
    required = {'entity_id','trading_code','trade_date','high','low','close','pre_close','vol'}
    if not required.issubset(quotes):
        raise ValueError('canonical OHLCV fields required')
    # Cut off future rows before any price calculation or validation.
    q = quotes.loc[quotes.trade_date.astype(str).isin(dates)].copy()
    q['trade_date'] = q.trade_date.astype(str)
    if q[['entity_id','trade_date']].isna().any().any() or q.duplicated(['entity_id','trade_date']).any():
        raise ValueError('duplicate or missing entity-session keys')
    entities = candidates.entity_id.tolist()
    arrays = {f: q.pivot(index='trade_date', columns='entity_id', values=f).reindex(index=dates, columns=entities).to_numpy(dtype=float)
              for f in ['high','low','close','pre_close','vol']}
    c,h,l,p,v = [arrays[f] for f in ['close','high','low','pre_close','vol']]
    at_signal = q.loc[q.trade_date.eq(dates[-1])].set_index('entity_id')
    actual_codes = at_signal.trading_code.reindex(entities).tolist()
    if any(pd.notna(a) and a != b for a,b in zip(actual_codes, candidates.trading_code)):
        raise ValueError('candidate trading code differs from canonical identity')
    complete = np.isfinite(c).all(axis=0) & np.isfinite(h).all(axis=0) & np.isfinite(l).all(axis=0) & np.isfinite(p).all(axis=0)
    valid = complete & (c>0).all(axis=0) & (l>0).all(axis=0) & (p>0).all(axis=0) & (h>=c).all(axis=0) & (l<=c).all(axis=0) & (h>=l).all(axis=0)
    gap = (np.abs(p[1:]-c[:-1]) > np.maximum(.011, np.abs(c[:-1])*.001)).any(axis=0)
    usable = valid & ~gap
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = c[1:]/c[:-1]-1
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        values = [c[-1]/c[0]-1, c[-1]/c[1:].mean(axis=0)-1,
                  tr[-14:].mean(axis=0)/c[-1], returns.std(axis=0, ddof=0),
                  v[-1]/v[:-1].mean(axis=0)]
    result = candidates[identity].reset_index(drop=True).copy()
    for f,value in zip(FEATURES, values):
        mask = usable if f != 'volume_ratio20' else (np.isfinite(v).all(axis=0) & (v>=0).all(axis=0) & (v[:-1].mean(axis=0)>0))
        result[f] = np.where(mask & np.isfinite(value), value, np.nan)
    result['price_window_status'] = np.select([~complete, ~valid, gap],
        ['incomplete_market_window','invalid_price_window','unresolved_reference_discontinuity'], default='raw_price_diagnostic')
    result['formal_training_eligible'] = False
    return result


def build(root, candidate_path, candidate_sha):
    from .identity_panel_verify import verify
    root, candidate_path = Path(root).resolve(), Path(candidate_path).resolve()
    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve(); sha = digest(path)
        if (expected is not None and sha != expected) or (path in pins and pins[path] != sha):
            raise ValueError('causal feature input changed')
        pins[path] = sha
        return path
    candidates = pd.read_parquet(pin(candidate_path, candidate_sha))
    if not 1 <= len(candidates) <= 100000 or candidates.signal_date.nunique() != 1:
        raise ValueError('bounded single-date candidate partition required')
    inventory = load_plan(pin(root/'config/s20_v4_data_sources.json'))
    spec = next(s for s in inventory['sources'] if s['role']=='identity_aware_daily_panel')
    panel = root/spec['path']
    identity_check = verify(root, panel, spec['summary_sha256'])
    summary = load_plan(pin(panel/'summary.json',spec['summary_sha256']))
    receipts = json.loads(pin(panel/'inputs_outputs.json',summary['receipt_sha256']).read_text(encoding='utf-8'))
    cal_spec = next(s for s in inventory['sources'] if s['role']=='exchange_calendar' and s.get('status')=='validated_calendar_source')
    cal = pd.read_parquet(pin(root/cal_spec['path'],cal_spec['sha256']))
    dates = sorted(cal.loc[cal.exchange.eq('SSE') & cal.is_open.eq(1),'cal_date'].astype(str))
    sz = sorted(cal.loc[cal.exchange.eq('SZSE') & cal.is_open.eq(1),'cal_date'].astype(str))
    if dates != sz:
        raise ValueError('SH/SZ calendar mismatch')
    end = dates.index(str(candidates.signal_date.iloc[0]))
    window = dates[max(0,end-20):end+1]
    frames=[]
    for item in receipts:
        path=(panel/item['canonical']).resolve()
        if not path.is_relative_to(panel.resolve()): raise ValueError('canonical path escapes panel')
        if path.stem in window:
            frames.append(pd.read_parquet(pin(path,item['canonical_sha256'])))
    result = compute(candidates,pd.concat(frames,ignore_index=True),window)
    pin(Path(__file__))
    if any(digest(p)!=h for p,h in pins.items()): raise ValueError('causal source changed before publication')
    out=root/'output/experiments/s20_safe_v4/sources'/('causal-features-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    result.to_parquet(out/'features.parquet',index=False)
    atomic_json(out/'inputs.json',{str(p):h for p,h in pins.items()})
    report=dict(directory=str(out),at=now(),rows=len(result),window=window,features=FEATURES,
        price_status_counts=result.price_window_status.value_counts().to_dict(),
        nonmissing_by_feature={f:int(result[f].notna().sum()) for f in FEATURES},
        identity_validation=identity_check,all_candidates_retained=len(result)==len(candidates),
        historical_availability_proven=False,economic_price_semantics_proven=False,formal_training_authorized=False,
        table_sha256=digest(out/'features.parquet'),inputs_sha256=digest(out/'inputs.json'))
    atomic_json(out/'summary.json',report)
    return report
