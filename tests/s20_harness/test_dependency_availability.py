import pandas as pd
import pytest

from research.s20_harness.dependency_availability import resolve
from research.s20_harness.feature_pipeline import prepare
from tests.s20_harness.test_feature_pipeline import fixture


def setup():
    samples, values, boundaries, contract = fixture()
    dependencies, receipts = [], []
    for r in samples.itertuples():
        for role, at in [('feature', r.feature_available_at), ('label', r.label_available_at)]:
            item = dict(sample_id=r.sample_id, role=role, dependency_id='source')
            dependencies.append(item)
            receipts.append(dict(item, available_at=at, basis='historical_receipt'))
    return samples, values, boundaries, contract, pd.DataFrame(dependencies), pd.DataFrame(receipts)


def test_label_after_scoring_is_valid_but_late_feature_not_imputed():
    samples, values, boundaries, contract, deps, receipts = setup()
    mask = receipts.sample_id.eq('calibration0') & receipts.role.eq('feature')
    receipts.loc[mask, 'available_at'] = samples.loc[samples.sample_id.eq('calibration0'), 'prediction_at'].iloc[0]
    receipts.loc[mask, 'basis'] = 'observed_now'
    matrix, assigned, report = prepare(samples, values, boundaries, contract,
                                      dependency_contract=deps, dependency_receipts=receipts)
    assert report['fit_sample_ids'] == ['fit0', 'fit1']
    assert pd.isna(matrix.loc[matrix.sample_id.eq('calibration0'), 'x']).all()
    assert len(matrix) == len(samples)
    assert not report['dependency_availability']['receipt_provenance_independently_verified']


def test_missing_declared_dependency_and_late_label_cannot_use_summary_time():
    samples, _, boundaries, _, deps, receipts = setup()
    from research.s20_harness.splits import assign_segments
    extra = pd.DataFrame([dict(sample_id='fit0', role='feature', dependency_id='missing')])
    deps = pd.concat([deps, extra], ignore_index=True)
    receipts.loc[receipts.sample_id.eq('fit1') & receipts.role.eq('label'), 'available_at'] = boundaries[1]['start_at']
    derived, ledger, report = resolve(samples, deps, receipts)
    assigned, _ = assign_segments(derived, boundaries)
    assert not assigned.loc[assigned.segment.eq('fit'), 'supervised_eligible'].any()
    assert len(ledger) == len(deps) and len(derived) == len(samples)


def test_roles_and_unknown_receipts_fail_closed():
    samples, _, _, _, deps, receipts = setup()
    receipts.loc[0, 'dependency_id'] = 'undeclared'
    with pytest.raises(ValueError, match='not declared'):
        resolve(samples, deps, receipts)
    receipts.loc[0, 'dependency_id'] = 'source'
    receipts.loc[0, 'basis'] = 'unknown'
    with pytest.raises(ValueError, match='unknown receipt'):
        resolve(samples, deps, receipts)
