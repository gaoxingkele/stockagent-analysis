from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness.metadata_source import NAME_FIELDS
from research.s20_harness.name_timeline import build
from research.s20_harness.name_timeline_verify import verify
from research.s20_harness.runtime import atomic_json, digest, load_plan


def fixture(root):
    source = root/'names'
    source.mkdir()
    codes = ['000001.SZ', '000002.SZ']
    atomic_json(source/'collection_plan.json', dict(codes=codes))
    for i, code in enumerate(codes):
        row = dict.fromkeys(NAME_FIELDS.split(','))
        row.update(ts_code=code, name='STtest', start_date='20240102', ann_date='20240102')
        frame = pd.DataFrame([row] if i == 0 else [], columns=NAME_FIELDS.split(','))
        path = source/(code+'.parquet')
        frame.to_parquet(path, index=False)
        atomic_json(source/(code+'.json'), dict(code=code, rows=len(frame), sha256=digest(path)))
    summary = build(source, '20240101', '20240105')
    return Path(summary['directory'])/'name_timeline.parquet'


def test_unknown_and_announcement_gate_retained_in_formal_evidence(tmp_path):
    from research.s20_harness.source_evidence import inspect_sources
    path = fixture(tmp_path)
    result = verify(tmp_path, path, digest(path))
    assert result['full_timeline_reconstructed']
    assert result['empty_unknown_codes'] == ['000002.SZ'] and result['unknown_intervals'] == 2
    assert not result['official_ST_status_proven']
    records, pins = inspect_sources(tmp_path, {'sources': [{'role': 'per_code_historical_names',
        'path': str(path.parent.parent), 'timeline': str(path), 'timeline_sha256': digest(path)}]},
        revalidate_names=True)
    assert records[0]['fresh_name_validation']['codes'] == 2 and pins
    assert not records[0]['semantic_acceptance']


def test_rehashed_false_timeline_rejected(tmp_path):
    path = fixture(tmp_path)
    frame = pd.read_parquet(path)
    frame.loc[frame.status.eq('known_name_proxy'), 'name'] = 'invented'
    frame.to_parquet(path, index=False)
    summary = load_plan(path.parent/'summary.json')
    summary['timeline_sha256'] = digest(path)
    atomic_json(path.parent/'summary.json', summary)
    with pytest.raises(AssertionError):
        verify(tmp_path, path, digest(path))


def test_rehashed_false_summary_rejected(tmp_path):
    path = fixture(tmp_path)
    summary = load_plan(path.parent/'summary.json')
    summary['unknown_intervals'] = 0
    atomic_json(path.parent/'summary.json', summary)
    with pytest.raises(ValueError, match='summary mismatch'):
        verify(tmp_path, path, digest(path))
