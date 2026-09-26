"""Compare indexed event routing with the independent dataframe selection path.

Checks supplied-record selection only: neither implementation proves that the
vendor supplied every event or that review decisions are economically correct.
"""
from dataclasses import asdict
import json
from pathlib import Path
import uuid

import pandas as pd

from .distribution_adapter import window_events
from .event_index import EventIndex
from .runtime import atomic_json, digest, load_plan, now


def compare(candidates, frame, events, decisions, start, end):
    if candidates.sample_id.isna().any() or candidates.sample_id.duplicated().any():
        raise ValueError('invalid event audit candidate identity')
    index = EventIndex(frame, events, decisions)
    groups = {code: group for code, group in frame.groupby('ts_code', sort=False)}
    results = []
    cache = {}
    for row in candidates.itertuples(index=False):
        codes = tuple(row.event_codes)
        if not codes:
            raise ValueError('explicit event codes required')
        if codes not in cache:
            # Restrict by code only, never by dates; the reference independently
            # parses dates and keeps unknown dates. Both retain payload order.
            parts = [groups[code] for code in set(codes) if code in groups]
            subset = pd.concat(parts) if parts else frame.iloc[:0]
            expected = window_events(subset, events, decisions, list(codes), start, end)
            actual = index.window(list(codes), start, end)
            def serialize(selection):
                return dict(selection, events=[asdict(e) for e in selection['events']])
            cache[codes] = (serialize(expected), serialize(actual))
        expected, actual = cache[codes]
        results.append(dict(sample_id=row.sample_id, matched=expected == actual,
            reference_status=expected['status'],
            reference_event_ids=json.dumps([e['event_id'] for e in expected['events']]),
            unresolved_event_ids=json.dumps(expected['unresolved_event_ids']),
            reference_json=json.dumps(expected, sort_keys=True),
            indexed_json=json.dumps(actual, sort_keys=True)))
    return pd.DataFrame(results)


def build(root, partition, partition_sha):
    from .label_partition_verify import verify
    from .distribution_adapter import adapt
    from . import distribution_adapter, event_index, corporate_actions, cash_unit_adapter
    root, partition = Path(root).resolve(), Path(partition).resolve()
    verify(partition, partition_sha)
    summary = load_plan(partition/'summary.json')
    sources = load_plan(partition/'inputs.json')
    pins = {p: h for p, h in sources.items() if Path(p).suffix != '.py'}
    for module in [distribution_adapter, event_index, corporate_actions, cash_unit_adapter]:
        path = Path(module.__file__).resolve(); pins[str(path)] = digest(path)
    pins[str(Path(__file__).resolve())] = digest(Path(__file__))
    def check():
        if any(digest(Path(p)) != h for p, h in pins.items()):
            raise ValueError('event audit input changed')
    def source(name):
        paths = [Path(p) for p in sources if Path(p).name == name]
        if len(paths) != 1:
            raise ValueError('unique event audit source required')
        return paths[0]
    check()
    candidates = pd.read_parquet(partition/'candidates.parquet')
    signal_dates = candidates.signal_date.astype(str).unique()
    if len(signal_dates) != 1:
        raise ValueError('one signal date required')
    calendar = pd.read_parquet(source('trade_cal.parquet'))
    dates = sorted(calendar.loc[calendar.exchange.eq('SSE') & calendar.is_open.eq(1), 'cal_date'].astype(str))
    start = dates.index(signal_dates[0]); window = dates[start+1:start+21]
    if len(window) != 20:
        raise ValueError('twenty market sessions required')
    frame = pd.read_parquet(source('normalized_distributions.parquet'))
    review = Path(summary['review_path'])
    if pins.get(str(review)) != summary['review_sha256']:
        raise ValueError('unbound event review')
    events, decisions = adapt(frame, load_plan(review)['reviews'],
        rate_policy='gross_reference_diagnostic',
        unit_reviews=load_plan(source('s20_v4_cash_unit_reviews.json'))['reviews'])
    result = compare(candidates, frame, events, decisions, window[0], window[-1])
    verify(partition, partition_sha); check()
    out = root/'output/experiments/s20_safe_v4/sources'/('event-selection-audit-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    result.to_parquet(out/'comparisons.parquet', index=False)
    atomic_json(out/'inputs.json', dict(partition=str(partition), partition_sha256=partition_sha, source_pins=pins))
    report = dict(at=now(), directory=str(out), rows=len(result),
        matched=int(result.matched.sum()), mismatched=int((~result.matched).sum()),
        reference_status_counts=result.reference_status.value_counts().to_dict(),
        all_candidates_retained=True, source_event_coverage_proven=False,
        review_semantics_independently_verified=False, formal_training_authorized=False,
        artifacts={n: digest(out/n) for n in ['comparisons.parquet', 'inputs.json']})
    atomic_json(out/'summary.json', report)
    return report
