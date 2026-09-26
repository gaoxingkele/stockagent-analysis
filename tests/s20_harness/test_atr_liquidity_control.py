import json

import pandas as pd
import pytest

from research.s20_harness.recommendation_policy import apply
from research.s20_harness import baseline_run, baseline_replay
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_recommendation_policy import fixture
from tests.s20_harness.test_baseline_run import plan


def control_fixture():
    frame, policy, calendar = fixture()
    policy.update(mode='atr_liquidity_control', risk_target_id=None, max_risk=None,
                  max_atr_fraction=.04, min_traded_value_cny=1000000.)
    controls = pd.DataFrame(dict(sample_id=['c', 'a', 'b'],
        available_at=['2024-05-02T15:01:00+08:00'] * 3,
        atr_fraction=[.1, .04, .02], traded_value_cny=[2000000., 1000000., 900000.]))
    return frame, policy, calendar, controls


def test_filter_preserves_scores_candidates_and_calendar():
    frame, policy, calendar, controls = control_fixture()
    original = frame.copy(deep=True)
    rows, report = apply(frame, policy, calendar, controls=controls)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_series_equal(rows.score, frame.score)
    assert rows.sample_id.tolist() == frame.sample_id.tolist()
    assert rows.loc[rows.selected, 'sample_id'].tolist() == ['a']
    assert rows.control_passed.tolist() == [False, True, False]
    assert report['daily'][1]['selected'] == 0
    assert not report['risk_gate_applied']
    assert report['control_model_fits'] == 0
    assert not report['filtered_score_is_new_probability']


def test_missing_controls_abstain_not_delete():
    frame, policy, calendar, controls = control_fixture()
    controls['atr_fraction'] = float('nan')
    rows, report = apply(frame, policy, calendar, controls=controls)
    assert len(rows) == 3 and not rows.selected.any()
    assert report['daily'][0]['precision'] is None
    assert set(rows.reject_reason) == {'atr_liquidity_unavailable'}


@pytest.mark.parametrize('kind', ['future', 'no_time', 'duplicate', 'extra', 'outcome',
                                 'infinity', 'negative', 'boolean', 'bad_threshold', 'risk_claim'])
def test_invalid_control_inputs(kind):
    frame, policy, calendar, controls = control_fixture()
    if kind == 'future':
        controls.loc[0, 'available_at'] = '2024-05-03T00:00:00+08:00'
    elif kind == 'no_time':
        controls.loc[0, 'available_at'] = None
    elif kind == 'duplicate':
        controls.loc[0, 'sample_id'] = 'a'
    elif kind == 'extra':
        controls.loc[0, 'sample_id'] = 'not_candidate'
    elif kind == 'outcome':
        controls['target'] = True
    elif kind == 'infinity':
        controls.loc[0, 'atr_fraction'] = float('inf')
    elif kind == 'negative':
        controls.loc[0, 'traded_value_cny'] = -1.
    elif kind == 'boolean':
        controls['atr_fraction'] = True
    elif kind == 'bad_threshold':
        policy['max_atr_fraction'] = True
    else:
        policy['max_risk'] = .1
    with pytest.raises(ValueError):
        apply(frame, policy, calendar, controls=controls)


def test_saved_baseline_policy_and_semantic_replay(tmp_path):
    value = plan()
    value['policy'].update(mode='atr_liquidity_control', max_atr_fraction=.04,
                           min_traded_value_cny=1000000.)
    value['policy_controls'] = [dict(sample_id='outer-test'+str(i),
        available_at='2024-05-02T20:00:00Z', atr_fraction=.02 if i == 0 else .1,
        traded_value_cny=2000000.) for i in range(2)]
    path = tmp_path/'input.json'
    atomic_json(path, value)
    result = baseline_run.build(tmp_path, path, digest(path))
    out = baseline_run.Path(result['directory'])
    ledger = pd.read_parquet(out/'candidate_ledger.parquet')
    assert ledger.loc[ledger.selected, 'sample_id'].tolist() == ['outer-test0']
    assert len(ledger) == 2 and result['model_level_fits'] == 1
    assert baseline_replay.replay(tmp_path, out, digest(out/'summary.json'))['semantic_replay_performed']
    # Rehashing modified saved values cannot pass semantic reconstruction.
    ledger.loc[0, 'atr_fraction'] = .03
    ledger.to_parquet(out/'candidate_ledger.parquet', index=False)
    checkpoint = json.loads((out/'checkpoint.json').read_text())
    checkpoint['artifacts']['candidate_ledger.parquet'] = digest(out/'candidate_ledger.parquet')
    atomic_json(out/'checkpoint.json', checkpoint)
    summary = json.loads((out/'summary.json').read_text())
    summary['artifacts'].update(checkpoint['artifacts'])
    summary['artifacts']['checkpoint.json'] = digest(out/'checkpoint.json')
    atomic_json(out/'summary.json', summary)
    with pytest.raises(ValueError, match='semantic replay failed'):
        baseline_replay.replay(tmp_path, out, digest(out/'summary.json'))


def test_owned_same_fold_control_budget_and_full_ledger(tmp_path):
    from research.s20_harness.multifold_run import run
    from tests.s20_harness.test_multifold_run import same_fold_models
    path = same_fold_models(tmp_path)
    manifest = json.loads(path.read_text())
    job = manifest['jobs'][1]
    input_path = baseline_run.Path(job['input_path'])
    value = json.loads(input_path.read_text())
    value['policy'].update(mode='atr_liquidity_control', max_atr_fraction=.04,
                           min_traded_value_cny=1000000.)
    value['policy_controls'] = [dict(sample_id='outer-test'+str(i),
        available_at='2024-05-02T20:00:00Z', atr_fraction=.02 if i == 0 else .1,
        traded_value_cny=2000000.) for i in range(2)]
    atomic_json(input_path, value)
    job['input_sha256'] = digest(input_path)
    manifest['budget']['trials'][1]['input_sha256'] = digest(input_path)
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match='comparison scope mismatch'):
        run(tmp_path, path, digest(path))
    # Existing model-family competition freezes the policy. Both models must
    # share the same controls; this is not a silently enabled policy search.
    first = manifest['jobs'][0]
    first_path = baseline_run.Path(first['input_path'])
    first_value = json.loads(first_path.read_text())
    first_value['policy'] = value['policy']
    first_value['policy_controls'] = value['policy_controls']
    atomic_json(first_path, first_value)
    first['input_sha256'] = digest(first_path)
    manifest['budget']['trials'][0]['input_sha256'] = digest(first_path)
    atomic_json(path, manifest)
    result = run(tmp_path, path, digest(path))
    assert result['budget']['reserved_counts']['policy_evaluations'] == 2
    assert result['budget']['reserved_counts']['model_fits'] == 2
    ledger = pd.read_parquet(result['outer_prediction_ledger']['path'])
    assert len(ledger) == 4
    out = baseline_run.Path(result['jobs'][1]['directory'])
    selected = pd.read_parquet(out/'candidate_ledger.parquet')
    assert selected.loc[selected.selected, 'sample_id'].tolist() == ['outer-test0']
    assert selected.score.notna().all()
