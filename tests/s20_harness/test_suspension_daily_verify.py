from pathlib import Path
import pandas as pd
import pytest

from research.s20_harness.runtime import atomic_json, digest, load_plan
from research.s20_harness.suspension_daily_source import collect
from research.s20_harness.suspension_daily_audit import audit
from research.s20_harness.suspension_daily_verify import verify
from research.s20_harness.source_evidence import inspect_sources


def fixture(tmp_path):
    daily = tmp_path/'output/tushare_cache/daily'
    daily.mkdir(parents=True)
    pd.DataFrame({'ts_code':['000046.SZ']}).to_parquet(daily/'20240102.parquet')
    def query(date):
        return pd.DataFrame(dict(ts_code=['000046.SZ'], trade_date=[date], suspend_type=['S'], suspend_timing=[None]))
    result = collect(tmp_path, query, max_requests=1, pause=0)
    source = tmp_path/'output/experiments/s20_safe_v4/sources'/result['source_id']
    report = audit(tmp_path, source, expected_plan_sha256=digest(source/'collection_plan.json'))
    return source, Path(report['directory'])


def test_source_integration_replays_without_acceptance(tmp_path):
    source, directory = fixture(tmp_path)
    inventory = dict(sources=[dict(role='daily_suspension_events', path=str(source),
        audit=str(directory/'summary.json'), audit_sha256=digest(directory/'summary.json'))])
    records, artifacts = inspect_sources(tmp_path, inventory, revalidate_suspensions=True)
    check = records[0]['fresh_suspension_validation']
    assert check['full_receipt_reconstruction_verified']
    assert check['full_day_candidates_with_quotes'] == 1
    assert not check['tradability_semantics_accepted'] and not records[0]['semantic_acceptance']
    assert any(a['path'].endswith('20240102.parquet') for a in artifacts)


def test_rehashed_false_overlay_rejected_by_reconstruction(tmp_path):
    source, directory = fixture(tmp_path)
    rows = pd.read_parquet(directory/'rows.parquet')
    rows['quote_present'] = False
    rows.to_parquet(directory/'rows.parquet', index=False)
    summary = load_plan(directory/'summary.json')
    summary['rows_sha256'] = digest(directory/'rows.parquet')
    atomic_json(directory/'summary.json', summary)
    with pytest.raises(AssertionError):
        verify(tmp_path, directory, digest(directory/'summary.json'), source)
