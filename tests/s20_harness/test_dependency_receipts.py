import pandas as pd
import pytest

from research.s20_harness.dependency_receipts import bind, prepare_bound
from research.s20_harness.dependency_availability import resolve
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_feature_pipeline import fixture


def binding(root, sample, role, at, number):
    path = root/f'data{number}.json'
    receipt = root/f'receipt{number}.json'
    atomic_json(path, dict(value=number))
    atomic_json(receipt, dict(file=path.name, sha256=digest(path), requested_at=at, received_at=at))
    return dict(sample_id=sample, role=role, dependency_id='raw', artifact_path=str(path),
                artifact_sha256=digest(path), receipt_path=str(receipt), receipt_sha256=digest(receipt))


def test_real_receipt_times_override_optimistic_sample_summary(tmp_path):
    samples, values, boundaries, contract = fixture()
    rows = []
    for r in samples.itertuples():
        for role, at in [('feature', r.feature_available_at), ('label', r.label_available_at)]:
            rows.append(binding(tmp_path, r.sample_id, role, at, len(rows)))
    bindings = pd.DataFrame(rows)
    deps = bindings[['sample_id','role','dependency_id']]
    matrix, _, report = prepare_bound(tmp_path, samples, values, boundaries, contract, deps, bindings)
    assert len(matrix) == len(samples) and report['fit_sample_ids'] == ['fit0','fit1']
    assert report['bound_receipt_evidence']['local_receipt_artifact_bindings_verified']
    assert not report['bound_receipt_evidence']['external_timestamp_authenticity_proven']


def test_retrospective_acquisition_does_not_backdate_features(tmp_path):
    samples, _, _, _ = fixture()
    rows = [binding(tmp_path, 'fit0', role, '2026-01-01T00:00:00Z', i) for i,role in enumerate(['feature','label'])]
    frame = pd.DataFrame(rows)
    receipts, _ = bind(tmp_path, frame)
    derived, _, report = resolve(samples.iloc[:1], frame[['sample_id','role','dependency_id']], receipts)
    assert derived.feature_available_at.iloc[0].startswith('2026')
    assert report['role_diagnostics'][0]['feature_prior_to_prediction'] is False


@pytest.mark.parametrize('kind', ['wrong_file', 'reverse', 'future', 'changed_bytes'])
def test_invalid_receipt_bindings_rejected(tmp_path, kind):
    from research.s20_harness.runtime import load_plan
    row = binding(tmp_path, 'a', 'feature', '2024-01-01T00:00:00Z', 0)
    from pathlib import Path
    path = Path(row['receipt_path'])
    receipt = load_plan(path)
    if kind == 'wrong_file': receipt['file'] = '../other.json'
    if kind == 'reverse': receipt['received_at'] = '2023-01-01T00:00:00Z'
    if kind == 'future': receipt['received_at'] = '2099-01-01T00:00:00Z'
    if kind == 'changed_bytes': atomic_json(Path(row['artifact_path']), dict(changed=True))
    atomic_json(path, receipt)
    row['receipt_sha256'] = digest(path)
    with pytest.raises(ValueError): bind(tmp_path, pd.DataFrame([row]))


def test_shared_sources_hashed_once_then_rechecked(tmp_path,monkeypatch):
    from research.s20_harness import dependency_receipts as module
    row=binding(tmp_path,'a','feature','2024-01-01T00:00:00Z',0)
    counts={};original=module.digest
    def counted(path):
        counts[str(path)]=counts.get(str(path),0)+1
        return original(path)
    monkeypatch.setattr(module,'digest',counted)
    receipts,_=bind(tmp_path,pd.DataFrame([dict(row,sample_id=str(i)) for i in range(20)]))
    assert len(receipts)==20 and sorted(counts.values())==[2,2]


def test_shared_receipt_cache_does_not_hide_mid_bind_mutation(tmp_path,monkeypatch):
    from pathlib import Path
    from research.s20_harness import dependency_receipts as module
    row=binding(tmp_path,'a','feature','2024-01-01T00:00:00Z',0)
    original=module.load_plan
    def mutate(path):
        receipt=original(path)
        atomic_json(Path(row['artifact_path']),{'changed':True})
        return receipt
    monkeypatch.setattr(module,'load_plan',mutate)
    with pytest.raises(ValueError,match='source changed during binding'):
        bind(tmp_path,pd.DataFrame([row,dict(row,sample_id='b')]))
