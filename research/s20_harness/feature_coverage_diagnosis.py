"""Attribute missing joins to observed source coverage, not inferred ST/delisting causes."""
from pathlib import Path
import uuid

import pandas as pd

from .runtime import load_plan, digest, atomic_json, now


def classify(joined, source_keys):
    if not {'sample_id','trading_code','signal_date','feature_row_present'}.issubset(joined):
        raise ValueError('candidate join fields missing')
    if joined.sample_id.isna().any() or joined.sample_id.duplicated().any():
        raise ValueError('invalid sample identities')
    if not {'ts_code','trade_date'}.issubset(source_keys):
        raise ValueError('source keys missing')
    codes = set(source_keys.ts_code)
    pairs = set(zip(source_keys.ts_code, source_keys.trade_date.astype(str)))
    result = joined[['sample_id','trading_code','signal_date','feature_row_present']].copy()
    reasons = []
    for row in result.itertuples():
        present = (row.trading_code, str(row.signal_date)) in pairs
        if type(row.feature_row_present) is not bool or row.feature_row_present != present:
            raise ValueError('saved feature match disagrees with source inventory')
        reasons.append('matched' if present else 'absent_from_all_source_groups' if row.trading_code not in codes
                       else 'code_present_but_signal_date_absent')
    result['coverage_reason'] = reasons
    result['cause_established'] = False
    return result


def build(root, directory, summary_sha):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if digest(directory/'summary.json') != summary_sha:
        raise ValueError('feature coverage summary pin mismatch')
    summary = load_plan(directory/'summary.json')
    if digest(directory/'inputs.json') != summary['inputs_sha256'] or digest(directory/'features.parquet') != summary['table_sha256']:
        raise ValueError('feature coverage artifact changed')
    inputs = load_plan(directory/'inputs.json')
    pins = {Path(p): h for p,h in inputs.items()}
    pins.update({directory/'summary.json':summary_sha, directory/'inputs.json':summary['inputs_sha256'],
                 directory/'features.parquet':summary['table_sha256']})
    if any(digest(p) != h for p,h in pins.items()):
        raise ValueError('feature coverage source changed')
    groups = [p for p in pins if p.name.startswith('group_') and p.suffix == '.parquet']
    if len(groups) != summary['source_files'] or not groups:
        raise ValueError('feature group inventory mismatch')
    keys = pd.concat([pd.read_parquet(p, columns=['ts_code','trade_date']) for p in groups], ignore_index=True)
    result = classify(pd.read_parquet(directory/'features.parquet'), keys)
    if any(digest(p) != h for p,h in pins.items()):
        raise ValueError('feature source changed during coverage diagnosis')
    out = root/'output/experiments/s20_safe_v4/sources'/('feature-coverage-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    result.to_parquet(out/'coverage.parquet', index=False)
    atomic_json(out/'inputs.json', {str(p):h for p,h in pins.items()})
    report = dict(directory=str(out), at=now(), rows=len(result),
                  counts=result.coverage_reason.value_counts().to_dict(), candidates_removed=0,
                  stock_selection_cause_proven=False, formal_training_authorized=False,
                  source_files=len(groups), source_codes=int(keys.ts_code.nunique()),
                  table_sha256=digest(out/'coverage.parquet'), inputs_sha256=digest(out/'inputs.json'),
                  code_sha256=digest(Path(__file__)))
    atomic_json(out/'summary.json', report)
    return report
