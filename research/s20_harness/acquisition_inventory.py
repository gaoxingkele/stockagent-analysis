"""Bounded native receipt inventory; never certifies historical PIT coverage."""
from pathlib import Path
import argparse
import json
import uuid
import re

from .dependency_receipts import bind, KEYS, BINDINGS
from .runtime import atomic_json, digest, load_plan, now
import pandas as pd
from .label_availability import _instant


# Explicit mappings from the actual collectors; never infer names for unknown roles.
COLLECTORS = {
    'daily_suspension_events': ('date', r'\d{8}', 'suspension_daily_source.py'),
    'daily_price_limits_fresh': ('date', r'\d{8}', 'price_limit_source.py'),
    'corporate_action_distributions': ('date', r'\d{8}', 'dividend_source.py'),
    'per_code_historical_names': ('code', r'\d{6}\.(SH|SZ|BJ)', 'name_history_collect.py'),
}


def bind_collector(root, role, receipt_path, receipt_sha, value):
    field, pattern, producer = COLLECTORS[role]
    identity = value.get(field)
    if not isinstance(identity, str) or not re.fullmatch(pattern, identity):
        raise ValueError('invalid collector receipt identity')
    if receipt_path.name != identity + '.json':
        raise ValueError('collector receipt filename identity mismatch')
    if field == 'date':
        pd.to_datetime(identity, format='%Y%m%d', errors='raise')
    artifact = receipt_path.with_suffix('.parquet').resolve()
    if not artifact.is_relative_to(root):
        raise ValueError('collector artifact escapes root')
    expected = value.get('sha256')
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError('collector artifact SHA256 required')
    requested, received = _instant(value['requested_at']), _instant(value['received_at'])
    if received < requested or received > _instant(now()):
        raise ValueError('collector receipt timing reversed or future')
    if digest(artifact) != expected or digest(receipt_path) != receipt_sha:
        raise ValueError('collector receipt or artifact hash mismatch')
    # These are byte/time bindings only. Semantic checks and issuer authenticity
    # are separate gates; a zero-row response is not a normal-status assertion.
    return artifact, {str(artifact): expected, str(receipt_path): receipt_sha}, producer


def inspect(root, inventory):
    root = Path(root).resolve()
    rows, scanned = [], {}
    for source in inventory['sources']:
        base = (root / source['path']).resolve()
        if not base.is_relative_to(root):
            raise ValueError('inventory source escapes root')
        if not base.is_dir():
            continue
        # Native acquisition files are immediate children, not arbitrary nested
        # JSON payloads or aggregate receipts in unrelated research runs.
        for path in sorted(base.glob('*.json')):
            if path.stat().st_size > 8_000_000:
                rows.append(dict(role=source['role'], receipt=str(path), status='oversize_uninspected'))
                continue
            sha = digest(path)
            try:
                value = json.loads(path.read_text(encoding='utf-8-sig'))
            except (ValueError, UnicodeError) as exc:
                rows.append(dict(role=source['role'], receipt=str(path), status='invalid_json', error=str(exc)))
                value = None
            scanned[str(path)] = sha
            if digest(path) != sha:
                raise ValueError('receipt changed while reading')
            if not isinstance(value, dict) or not {'requested_at', 'received_at'} <= value.keys():
                continue
            row = dict(role=source['role'], receipt=str(path), receipt_sha256=sha,
                       requested_at=value['requested_at'], received_at=value['received_at'])
            if 'file' not in value and source['role'] in COLLECTORS:
                try:
                    artifact, pins, producer = bind_collector(root, source['role'], path, sha, value)
                    row.update(status='local_binding_verified', artifact=str(artifact),
                               artifact_sha256=value['sha256'], binding_schema=source['role'],
                               producer_mapping=producer)
                    scanned.update(pins)
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    row.update(status='invalid_binding', error=str(exc))
            elif not {'file', 'sha256'} <= value.keys():
                row.update(status='unsupported_receipt_schema')
            else:
                artifact = base / str(value['file'])
                dependency = pd.DataFrame([dict(sample_id='inventory', role='feature',
                    dependency_id=str(path), artifact_path=str(artifact), artifact_sha256=value['sha256'],
                    receipt_path=str(path), receipt_sha256=sha)], columns=KEYS+BINDINGS)
                try:
                    _, evidence = bind(root, dependency)
                    row.update(status='local_binding_verified', artifact=str(artifact.resolve()),
                               artifact_sha256=value['sha256'], binding_schema='native_file')
                    scanned.update(evidence['source_pins'])
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    row.update(status='invalid_binding', error=str(exc))
            rows.append(row)
    for path, sha in scanned.items():
        if digest(Path(path)) != sha:
            raise ValueError('inventory input changed during inspection')
    verified = [r for r in rows if r['status'] == 'local_binding_verified']
    from .label_availability import _instant
    times = [_instant(r['received_at']) for r in verified]
    return dict(checked_at=now(), scope='registered source directories, immediate JSON children only',
        rows=rows, scanned_files=scanned,
        verified_native_receipts=sum(r.get('binding_schema') == 'native_file' for r in verified),
        verified_collector_receipts=sum(r.get('binding_schema') != 'native_file' for r in verified),
        verified_total_receipts=len(verified),
        earliest_recorded_receipt=min(times).isoformat() if times else None,
        latest_recorded_receipt=max(times).isoformat() if times else None,
        external_timestamp_authenticity_proven=False, historical_dependency_coverage_proven=False,
        formal_training_authorized=False)


def build(root):
    root = Path(root).resolve()
    registry = root / 'config/s20_v4_data_sources.json'
    sha = digest(registry)
    sources = load_plan(registry)['sources']
    local = [s for s in sources if (root / s['path']).resolve().is_relative_to(root)]
    result = inspect(root, {'sources': local})
    result['external_sources_not_inspected'] = [s['path'] for s in sources if s not in local]
    if digest(registry) != sha:
        raise ValueError('source registry changed')
    result.update(registry_sha256=sha, code_sha256=digest(Path(__file__)),
                  producer_schema_code_pins={p: digest(Path(__file__).parent / p) for _, _, p in COLLECTORS.values()})
    output = root / 'output/experiments/s20_safe_v4/sources' / ('acquisition-inventory-' + uuid.uuid4().hex)
    output.mkdir(parents=True)
    atomic_json(output / 'inventory.json', result)
    summary = {k: v for k, v in result.items() if k not in {'rows', 'scanned_files'}}
    summary.update(directory=str(output), inventory_sha256=digest(output / 'inventory.json'),
                   statuses=pd.Series([r['status'] for r in result['rows']], dtype=str).value_counts().to_dict())
    atomic_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[2]), indent=2))
