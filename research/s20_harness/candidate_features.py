"""Candidate-first feature coverage diagnostic; no outcome/current-ST filtering."""
from pathlib import Path
import uuid

import pandas as pd
import pyarrow.parquet as pq

from .runtime import atomic_json, digest, now


def attach(candidates, factors, features):
    identity = ['sample_id', 'entity_id', 'trading_code', 'signal_date']
    if (not isinstance(features, list) or not features or len(features) != len(set(features))
            or any(not isinstance(f, str) or not f.strip() for f in features)
            or set(features) & set(identity+['ts_code', 'trade_date', 'feature_row_present', 'source_path', 'source_sha256'])):
        raise ValueError('explicit unique feature allowlist required')
    if not set(identity).issubset(candidates) or candidates[identity].isna().any().any() or candidates.sample_id.duplicated().any():
        raise ValueError('unique candidate identity required')
    required = {'ts_code', 'trade_date', 'source_path', 'source_sha256', *features}
    if set(factors.columns) != required or factors.columns.duplicated().any():
        raise ValueError('exact feature source schema required')
    if factors[['ts_code','trade_date']].isna().any().any() or factors.duplicated(['ts_code','trade_date']).any():
        raise ValueError('duplicate or missing feature stock-date identity')
    left = candidates[identity].copy()
    right = factors.rename(columns={'ts_code':'trading_code', 'trade_date':'signal_date'}).copy()
    left['signal_date'] = left.signal_date.astype(str)
    right['signal_date'] = right.signal_date.astype(str)
    if right.duplicated(['trading_code','signal_date']).any():
        raise ValueError('duplicate normalized feature identity')
    for values in (left.signal_date, right.signal_date):
        if not values.str.fullmatch(r'\d{8}').all():
            raise ValueError('feature dates require YYYYMMDD')
        pd.to_datetime(values, format='%Y%m%d', errors='raise')
    right['feature_row_present'] = True
    result = left.merge(right, on=['trading_code','signal_date'], how='left', sort=False, validate='many_to_one')
    result['feature_row_present'] = result.feature_row_present.eq(True)
    if result.sample_id.tolist() != candidates.sample_id.tolist() or len(result) != len(candidates):
        raise ValueError('feature join changed candidates')
    for f in features:
        if not pd.api.types.is_numeric_dtype(result[f]):
            raise ValueError('numeric feature values required')
    return result


def build(root, candidate_path, candidate_sha, factor_directory, features):
    root, candidate_path, factor_directory = Path(root).resolve(), Path(candidate_path).resolve(), Path(factor_directory).resolve()
    if not candidate_path.is_relative_to(root) or not factor_directory.is_relative_to(root):
        raise ValueError('feature sources outside workspace')
    if digest(candidate_path) != candidate_sha:
        raise ValueError('candidate input pin mismatch')
    candidates = pd.read_parquet(candidate_path)
    if not 1 <= len(candidates) <= 100000:
        raise ValueError('bounded nonempty candidate set required')
    dates = sorted(candidates.signal_date.astype(str).unique())
    files = sorted(factor_directory.glob('group_*.parquet'))
    if not 1 <= len(files) <= 100:
        raise ValueError('bounded feature group inventory required')
    pins = {candidate_path: candidate_sha, Path(__file__).resolve(): digest(Path(__file__))}
    frames = []
    for path in files:
        pins[path] = digest(path)
        schema = set(pq.read_schema(path).names)
        if not {'ts_code','trade_date',*features}.issubset(schema):
            raise ValueError('feature group missing declared fields')
        frame = pd.read_parquet(path, columns=['ts_code','trade_date',*features])
        frame = frame.loc[frame.trade_date.astype(str).isin(dates)].copy()
        frame['source_path'], frame['source_sha256'] = str(path), pins[path]
        frames.append(frame)
    joined = attach(candidates, pd.concat(frames, ignore_index=True), features)
    if sorted(factor_directory.glob('group_*.parquet')) != files or any(digest(p) != h for p,h in pins.items()):
        raise ValueError('feature source changed during join')
    out = root/'output/experiments/s20_safe_v4/sources'/('candidate-features-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    joined.to_parquet(out/'features.parquet', index=False)
    atomic_json(out/'inputs.json', {str(p):h for p,h in pins.items()})
    report = dict(directory=str(out), at=now(), candidates=len(candidates), output_rows=len(joined),
                  matched_rows=int(joined.feature_row_present.sum()),
                  missing_rows=int((~joined.feature_row_present).sum()), features=features,
                  missing_values_by_feature={f:int(joined[f].isna().sum()) for f in features},
                  all_candidates_retained=True, current_metadata_filter_used=False, outcome_filter_used=False,
                  source_feature_semantics_verified=False, historical_feature_availability_proven=False,
                  formal_training_authorized=False, source_files=len(files),
                  table_sha256=digest(out/'features.parquet'), inputs_sha256=digest(out/'inputs.json'))
    atomic_json(out/'summary.json', report)
    return report
