"""Rebuild announcement-gated name proxies; never certify an official ST registry."""
from pathlib import Path
import re

import pandas as pd

from . import name_timeline, name_history
from .metadata_source import NAME_FIELDS
from .runtime import digest, load_plan


def verify(root, path, table_sha):
    root, path = Path(root).resolve(), Path(path).resolve()
    directory, source = path.parent, path.parent.parent
    pins = {}

    def pin(p, expected=None):
        p = Path(p).resolve()
        if not p.is_relative_to(root):
            raise ValueError("name evidence escapes root")
        sha = digest(p)
        if (expected is not None and sha != expected) or (p in pins and pins[p] != sha):
            raise ValueError("name evidence hash mismatch")
        pins[p] = sha
        return p

    saved = pd.read_parquet(pin(path, table_sha))
    summary = load_plan(pin(directory/'summary.json'))
    if summary['timeline_sha256'] != table_sha:
        raise ValueError('name timeline summary binding mismatch')
    for module, key in ((name_timeline, 'code_sha256'), (name_history, 'asof_code_sha256')):
        code = Path(module.__file__).resolve()
        if digest(code) != summary[key]:
            raise ValueError('name timeline code changed')
        pins[code] = summary[key]
    audit_directory = Path(summary['audit_directory']).resolve()
    if audit_directory.parent != source:
        raise ValueError('name audit source mismatch')
    inputs = load_plan(pin(audit_directory/'inputs.json', summary['audit_inputs_sha256']))
    plan = load_plan(pin(source/'collection_plan.json', inputs['plan_sha256']))
    codes = plan['codes']
    if (not codes or len(codes) != len(set(codes))
            or any(not isinstance(c, str) or not re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', c) for c in codes)
            or [x['code'] for x in inputs['receipts']] != codes):
        raise ValueError('incomplete or invalid name inventory')
    if {p.name for p in source.glob('*.parquet')} != {c+'.parquet' for c in codes}:
        raise ValueError('name source inventory changed')
    rows, empty, missing_ann, source_rows = [], [], 0, 0
    for item in inputs['receipts']:
        code = item['code']
        receipt = load_plan(pin(source/(code+'.json'), item['receipt_sha256']))
        if receipt['code'] != code or receipt['sha256'] != item['data_sha256']:
            raise ValueError('name receipt identity mismatch')
        events = pd.read_parquet(pin(source/(code+'.parquet'), item['data_sha256']))
        if (not set(NAME_FIELDS.split(',')).issubset(events.columns) or not events.ts_code.eq(code).all()
                or len(events) != receipt['rows'] or len(events) >= 1000):
            raise ValueError('name source schema or count mismatch')
        source_rows += len(events)
        if events.empty:
            empty.append(code)
        missing_ann += int(pd.to_datetime(events.ann_date.astype('string'), format='%Y%m%d', errors='coerce').isna().sum())
        states = name_timeline.timeline(events, [code], summary['start_date'], summary['end_date'])
        for state in states:
            state.update(source_code=code, source_sha256=item['data_sha256'])
        rows.extend(states)
    rebuilt = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(saved.sort_index(axis=1), rebuilt.sort_index(axis=1),
                                  check_dtype=False, check_exact=True)
    unknown = sum(r['status'] == 'unknown' for r in rows)
    if summary['codes'] != len(codes) or summary['timeline_rows'] != len(rows) or summary['unknown_intervals'] != unknown:
        raise ValueError('name reconstructed summary mismatch')
    for p, sha in pins.items():
        if digest(p) != sha:
            raise ValueError('name evidence changed during reconstruction')
    return dict(full_timeline_reconstructed=True, codes=len(codes), source_rows=source_rows,
                timeline_rows=len(rows), unknown_intervals=unknown, empty_unknown_codes=empty,
                invalid_or_missing_announcement_rows=missing_ann, official_ST_status_proven=False,
                historical_revision_availability_proven=False, formal_training_eligible=False,
                evidence_files=[dict(path=str(p), sha256=sha, role='fresh_name_timeline_reconstruction')
                                for p, sha in pins.items()])
