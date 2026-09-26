from pathlib import Path

import pytest

from research.s20_harness import baseline_run
from research.s20_harness.runtime import atomic_json, digest, load_plan
from tests.s20_harness.test_baseline_run import plan
from tests.s20_harness.test_dependency_receipts import binding


def fixture(root):
    value = plan()
    value['schema_version'] = '2'
    value['dependency_contract'], value['dependency_bindings'] = [], []
    for sample in value['samples']:
        for role, field in [('feature', 'feature_available_at'), ('label', 'label_available_at')]:
            row = binding(root, sample['sample_id'], role, sample[field], len(value['dependency_bindings']))
            value['dependency_bindings'].append(row)
            value['dependency_contract'].append({k: row[k] for k in ('sample_id','role','dependency_id')})
    path = root/'plan.json'
    atomic_json(path, value)
    return path


def test_v2_fit_calibrate_select_and_replay(tmp_path):
    from research.s20_harness.baseline_replay import replay
    path = fixture(tmp_path)
    result = baseline_run.build(tmp_path, path, digest(path))
    out = Path(result['directory'])
    assert 'dependency_evidence.json' in result['artifacts']
    assert baseline_run.verify(out, digest(out/'summary.json'))['artifact_bytes_verified']
    evidence = load_plan(out/'dependency_evidence.json')
    assert evidence['receipt_evidence']['dependencies'] == 20
    assert not evidence['formal_data_acceptance']
    assert replay(tmp_path, out, digest(out/'summary.json'))['semantic_replay_performed']


def test_optimistic_summary_rejected_before_fitting(tmp_path, monkeypatch):
    path = fixture(tmp_path)
    value = load_plan(path)
    value['samples'][0]['feature_available_at'] = '2020-01-01T00:00:00Z'
    atomic_json(path, value)
    def no_fit(*args, **kwargs):
        pytest.fail('fit must not run')
    monkeypatch.setattr(baseline_run, 'fit_baseline', no_fit)
    with pytest.raises(ValueError, match='differs from bound'):
        baseline_run.build(tmp_path, path, digest(path))
    assert not (tmp_path/'output').exists()


def test_source_changes_during_fit_are_not_published(tmp_path, monkeypatch):
    path = fixture(tmp_path)
    original = baseline_run.fit_baseline
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        atomic_json(tmp_path/'data0.json', dict(changed=True))
        return result
    monkeypatch.setattr(baseline_run, 'fit_baseline', mutate)
    with pytest.raises(ValueError, match='source changed'):
        baseline_run.build(tmp_path, path, digest(path))
    out = next((tmp_path/'output/experiments/s20_safe_v4/sources').iterdir())
    assert not (out/'summary.json').exists()


def test_extra_dependency_fields_not_silently_discarded(tmp_path):
    path = fixture(tmp_path)
    value = load_plan(path)
    value['dependency_bindings'][0]['override_time'] = '2020-01-01'
    atomic_json(path, value)
    with pytest.raises(ValueError, match='row schema'):
        baseline_run.build(tmp_path, path, digest(path))
