from research.s20_harness import execution_golden as module
import pytest
from research.s20_harness.runtime import load_plan,atomic_json,digest


def test_saved_cases_include_expected_actual_and_limits(tmp_path):
    report=module.build(tmp_path)
    cases=load_plan(module.Path(report['directory'])/'golden_cases.json')['cases']
    assert report['all_passed'] and report['passed']==report['cases']==16
    assert len({c['case_id'] for c in cases})==16
    assert 'sale_preserves_dividend_claim' in {c['case_id'] for c in cases}
    assert all(c['actual']==c['expected'] for c in cases)
    assert not report['real_fill_evidence'] and not report['full_H02_coverage']


def test_case_failure_is_persisted_not_hidden(tmp_path,monkeypatch):
    monkeypatch.setattr(module.execution,'ohlc_path_bounds',lambda *args:(_ for _ in ()).throw(ValueError('injected')))
    report=module.build(tmp_path)
    assert report['cases']==16 and report['passed']==15 and not report['all_passed']
    cases=load_plan(module.Path(report['directory'])/'golden_cases.json')['cases']
    failed=[c for c in cases if not c['passed']]
    assert failed[0]['case_id']=='same_day_touch_bounds' and failed[0]['error']['message']=='injected'


@pytest.mark.parametrize('tamper',[None,'case','summary','code_scope'])
def test_consumer_replays_not_just_hashes(tmp_path,tamper):
    report=module.build(tmp_path);directory=module.Path(report['directory'])
    if tamper=='case':
        saved=load_plan(directory/'golden_cases.json')
        saved['cases'][0]['actual']['remaining']=99
        atomic_json(directory/'golden_cases.json',saved)
        report['artifacts']['golden_cases.json']=digest(directory/'golden_cases.json')
    if tamper=='summary': report['real_fill_evidence']=True
    if tamper=='code_scope':
        inputs=load_plan(directory/'inputs.json')
        inputs['code_pins'].pop(next(iter(inputs['code_pins'])))
        atomic_json(directory/'inputs.json',inputs)
        report['artifacts']['inputs.json']=digest(directory/'inputs.json')
    atomic_json(directory/'summary.json',report)
    if tamper:
        with pytest.raises(ValueError): module.verify(directory,digest(directory/'summary.json'))
    else:
        checked=module.verify(directory,digest(directory/'summary.json'))
        assert checked['replayed'] and checked['all_passed'] and not checked['full_H02_coverage']


def test_changed_live_case_cannot_reuse_saved_green(tmp_path,monkeypatch):
    report=module.build(tmp_path);directory=module.Path(report['directory'])
    monkeypatch.setattr(module,'evaluate',lambda *args:[])
    with pytest.raises(ValueError,match='replay differs'):
        module.verify(directory,digest(directory/'summary.json'))


def test_current_replay_explicitly_separate_from_historical_code(tmp_path):
    report=module.build(tmp_path);directory=module.Path(report['directory'])
    inputs=load_plan(directory/'inputs.json')
    inputs['code_pins']={p:'0'*64 for p in inputs['code_pins']}
    atomic_json(directory/'inputs.json',inputs)
    report['artifacts']['inputs.json']=digest(directory/'inputs.json')
    atomic_json(directory/'summary.json',report)
    sha=digest(directory/'summary.json')
    with pytest.raises(ValueError,match='code changed'): module.verify(directory,sha)
    result=module.replay_current(directory,sha)
    assert result['all_passed'] and result['cases']==16

    assert not result['historical_code_provenance_authenticated']
    assert not result['full_H02_coverage']


def test_current_replay_rejects_rehashed_result(tmp_path):
    report=module.build(tmp_path);directory=module.Path(report['directory'])
    saved=load_plan(directory/'golden_cases.json');saved['cases'][0]['actual']['remaining']=99
    atomic_json(directory/'golden_cases.json',saved)
    report['artifacts']['golden_cases.json']=digest(directory/'golden_cases.json')
    atomic_json(directory/'summary.json',report)
    with pytest.raises(ValueError,match='replay differs'):
        module.replay_current(directory,digest(directory/'summary.json'))
