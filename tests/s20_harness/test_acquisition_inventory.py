import pytest
from research.s20_harness.acquisition_inventory import inspect
from research.s20_harness.runtime import atomic_json, digest


def test_native_receipts_are_not_historical_coverage(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    atomic_json(data / 'raw.json', {'price': 10})
    receipt = dict(file='raw.json', sha256=digest(data / 'raw.json'),
                   requested_at='2026-01-01T00:00:00Z', received_at='2026-01-01T00:01:00Z')
    atomic_json(data / 'receipt.json', receipt)
    atomic_json(data / 'aggregate.json', [receipt])
    inv = {'sources': [dict(role='test', path='data')]}
    result = inspect(tmp_path, inv)
    assert result['verified_native_receipts'] == 1
    assert result['earliest_recorded_receipt'] == '2026-01-01T00:01:00+00:00'
    assert not result['historical_dependency_coverage_proven']
    assert not result['formal_training_authorized']
    atomic_json(data / 'raw.json', {'price': 11})
    result = inspect(tmp_path, inv)
    assert result['verified_native_receipts'] == 0
    assert result['rows'][0]['status'] == 'invalid_binding'


def test_outside_scope_rejected(tmp_path):
    with pytest.raises(ValueError, match='escapes'):
        inspect(tmp_path, {'sources': [dict(role='test', path='../elsewhere')]})


@pytest.mark.parametrize('role,field,identity', [
    ('daily_suspension_events', 'date', '20240102'),
    ('daily_price_limits_fresh', 'date', '20240102'),
    ('corporate_action_distributions', 'date', '20240102'),
    ('per_code_historical_names', 'code', '000001.SZ'),
])
def test_collector_binding_and_tampering(tmp_path, role, field, identity):
    import pandas as pd
    data = tmp_path / 'data'
    data.mkdir()
    raw = data / (identity + '.parquet')
    pd.DataFrame({'value': []}).to_parquet(raw, index=False)
    receipt = {field: identity, 'sha256': digest(raw),
               'requested_at': '2026-01-01T00:00:00Z', 'received_at': '2026-01-01T00:01:00Z'}
    receipt_path = data / (identity + '.json')
    atomic_json(receipt_path, receipt)
    inv = {'sources': [dict(role=role, path='data')]}
    result = inspect(tmp_path, inv)
    assert result['verified_collector_receipts'] == 1
    assert result['verified_native_receipts'] == 0
    assert not result['historical_dependency_coverage_proven']
    receipt[field] = '20240103' if field == 'date' else '000002.SZ'
    atomic_json(receipt_path, receipt)
    assert inspect(tmp_path, inv)['rows'][0]['status'] == 'invalid_binding'
    receipt[field] = identity
    receipt['received_at'] = '2025-01-01T00:00:00Z'
    atomic_json(receipt_path, receipt)
    assert inspect(tmp_path, inv)['rows'][0]['status'] == 'invalid_binding'
    receipt['received_at'] = '2026-01-01T00:01:00Z'
    atomic_json(receipt_path, receipt)
    pd.DataFrame({'value': [1]}).to_parquet(raw, index=False)
    assert inspect(tmp_path, inv)['rows'][0]['status'] == 'invalid_binding'
