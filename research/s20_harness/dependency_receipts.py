"""Bind declared dependencies to pinned local acquisition receipts and raw bytes."""
from pathlib import Path
import re

import pandas as pd

from .label_availability import _instant
from .runtime import digest, load_plan, now


KEYS = ['sample_id', 'role', 'dependency_id']
BINDINGS = ['artifact_path', 'artifact_sha256', 'receipt_path', 'receipt_sha256']


def bind(root, dependencies):
    """Only the native {file,sha256,requested_at,received_at} receipt schema.

    Pins must come from the consumer's frozen input manifest. Matching local
    bytes cannot authenticate historical timestamp issuance or completeness.
    """
    root = Path(root).resolve()
    if dependencies.columns.duplicated().any() or set(dependencies.columns) != set(KEYS+BINDINGS):
        raise ValueError('exact acquisition dependency binding schema required')
    if dependencies.isna().any().any() or dependencies.duplicated(KEYS).any():
        raise ValueError('unique nonmissing acquisition dependencies required')
    if not dependencies.role.isin(['feature', 'label']).all():
        raise ValueError('unsupported dependency role')
    for key in KEYS:
        if not dependencies[key].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('named dependency identities required')
    checked_at = _instant(now())
    pins, rows, receipt_cache = {}, [], {}

    def pin(value, expected):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', expected or ''):
            raise ValueError('explicit path and SHA256 required')
        path = (root/ value).resolve()
        if not path.is_relative_to(root):
            raise ValueError('dependency source escapes root')
        if path in pins:
            if pins[path] != expected:
                raise ValueError('dependency receipt or artifact hash mismatch')
            return path
        if digest(path) != expected:
            raise ValueError('dependency receipt or artifact hash mismatch')
        pins[path] = expected
        return path

    for r in dependencies.itertuples(index=False):
        artifact = pin(r.artifact_path, r.artifact_sha256)
        receipt_path = pin(r.receipt_path, r.receipt_sha256)
        if receipt_path not in receipt_cache:receipt_cache[receipt_path]=load_plan(receipt_path)
        receipt = receipt_cache[receipt_path]
        if (not isinstance(receipt.get('file'), str) or Path(receipt['file']).name != receipt['file']
                or (receipt_path.parent/receipt['file']).resolve() != artifact
                or receipt.get('sha256') != r.artifact_sha256):
            raise ValueError('receipt does not identify bound artifact')
        requested, received = _instant(receipt['requested_at']), _instant(receipt['received_at'])
        if received < requested or received > checked_at:
            raise ValueError('receipt timing reversed or future')
        rows.append(dict(sample_id=r.sample_id, role=r.role, dependency_id=r.dependency_id,
                         available_at=received.isoformat(), basis='historical_receipt'))
    for path, sha in pins.items():
        if digest(path) != sha:
            raise ValueError('dependency source changed during binding')
    report = dict(checked_at=checked_at.isoformat(), dependencies=len(rows),
                  local_receipt_artifact_bindings_verified=True, external_timestamp_authenticity_proven=False,
                  dependency_completeness_proven=False, formal_training_authorized=False,
                  basis='recorded local acquisition time; no event-date or file-mtime fallback',
                  source_pins={str(p): h for p,h in pins.items()})
    return pd.DataFrame(rows, columns=KEYS+['available_at', 'basis']), report


def prepare_bound(root, samples, values, boundaries, feature_contract, dependency_contract,
                  bindings, *, evaluation_at=None):
    """Prepare features using bound receipt times, preserving missing declarations."""
    from .feature_pipeline import prepare
    receipts, evidence = bind(root, bindings)
    matrix, assigned, report = prepare(samples, values, boundaries, feature_contract,
        evaluation_at=evaluation_at, dependency_contract=dependency_contract, dependency_receipts=receipts)
    # Protect the fit-time consumption interval as well as the initial read.
    for path, sha in evidence['source_pins'].items():
        if digest(Path(path)) != sha:
            raise ValueError('dependency source changed during preprocessing')
    report['bound_receipt_evidence'] = evidence
    return matrix, assigned, report
