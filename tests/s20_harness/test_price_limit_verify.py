from pathlib import Path
import json

import pandas as pd
import pytest

from research.s20_harness.price_limit_source import collect
from research.s20_harness.price_limit_audit import audit
from research.s20_harness.price_limit_verify import verify
from research.s20_harness.runtime import digest, load_plan, atomic_json


def fixture(root):
    daily = root / 'output/tushare_cache/daily'
    daily.mkdir(parents=True)
    pd.DataFrame({'ts_code': ['000001.SZ']}).to_parquet(daily / '20240102.parquet')
    def query(date):
        return pd.DataFrame({'ts_code': ['000001.SZ'], 'trade_date': [date],
                             'up_limit': [0.], 'down_limit': [0.]})
    result = collect(root, query, pause=0)
    source = root / 'output/experiments/s20_safe_v4/sources' / result['source_id']
    directory = Path(audit(root, source)['directory'])
    return source, directory


def test_invalid_limits_retained_and_inventory_integration(tmp_path):
    from research.s20_harness.source_evidence import inspect_sources
    source, directory = fixture(tmp_path)
    result = verify(tmp_path, directory, digest(directory/'summary.json'), source)
    assert result['full_coverage_reconstructed'] and result['invalid_limit_rows'] == 1
    assert result['usable_stock_dates'] == 0 and not result['effective_rule_semantics_proven']
    records, pins = inspect_sources(tmp_path, {'sources': [{'role': 'daily_price_limits_fresh',
        'path': str(source), 'audit': str(directory/'summary.json')}]}, revalidate_limits=True)
    assert records[0]['fresh_limit_validation']['invalid_limit_rows'] == 1 and pins
    assert not records[0]['semantic_acceptance']


def test_rehashed_false_summary_and_diagnostics_rejected(tmp_path):
    source, directory = fixture(tmp_path)
    summary = load_plan(directory/'summary.json')
    summary['usable_stock_dates'] = 1
    atomic_json(directory/'summary.json', summary)
    with pytest.raises(ValueError, match='summary mismatch'):
        verify(tmp_path, directory, digest(directory/'summary.json'), source)
    summary['usable_stock_dates'] = 0
    atomic_json(directory/'summary.json', summary)
    diagnostics = json.loads((directory/'date_diagnostics.json').read_text(encoding='utf-8'))
    diagnostics[0]['missing_or_invalid_expected_codes'] = []
    atomic_json(directory/'date_diagnostics.json', diagnostics)
    with pytest.raises(ValueError, match='diagnostics mismatch'):
        verify(tmp_path, directory, digest(directory/'summary.json'), source)


def test_extra_source_file_rejected(tmp_path):
    source, directory = fixture(tmp_path)
    pd.DataFrame().to_parquet(source/'20240103.parquet')
    with pytest.raises(ValueError, match='inventory changed'):
        verify(tmp_path, directory, digest(directory/'summary.json'), source)
