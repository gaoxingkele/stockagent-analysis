from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.joint_policy_run import build,verify
from research.s20_harness.joint_run import build as train
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_joint_run import plan


def setup(tmp_path):
    value=plan();path=tmp_path/'model.json';atomic_json(path,value)
    result=train(tmp_path,path,digest(path));directory=Path(result['directory'])
    samples=pd.DataFrame(value['samples']);selection=samples.loc[samples.sample_id.str.startswith('selection-policy')]
    base=dict(value['policy']['selection'],frozen_at='2024-03-01T00:00:00Z',policy_id='direct')
    registry=dict(registry_id='ablation',target_id='P.joint.v4',registered_at='2024-03-02T00:00:00Z',policies=[
        dict(selection=base,ranking='safe_probability'),
        dict(selection=dict(base,policy_id='penalty'),weights=value['policy']['weights'])])
    payload=dict(registry=registry,selection_calendar=sorted(selection.signal_date.unique()),
        selection_outcomes=[dict(sample_id=r.sample_id,target='A',label_available_at=r.label_available_at) for r in selection.itertuples()])
    source=tmp_path/'policies.json';atomic_json(source,payload)
    return directory,source


def test_saved_policy_comparison_recomputes_without_training(tmp_path,monkeypatch,capsys):
    import json
    from research.s20_harness import cli
    monkeypatch.setattr(cli,'_repo_root',lambda:tmp_path)
    directory,source=setup(tmp_path)
    before={p.name:digest(p) for p in directory.iterdir() if p.is_file()}
    assert cli.main(['build-joint-policy-diagnostic','--run-directory',str(directory),'--run-sha256',digest(directory/'summary.json'),
                     '--input',str(source),'--input-sha256',digest(source)])==0
    result=json.loads(capsys.readouterr().out);assert result.pop('valid');out=Path(result['directory'])
    assert result['policy_evaluations']==2 and result['model_fits']==result['calibrator_fits']==0
    comparison=load_plan(out/'comparison.json')
    diagnostic=comparison['fixed_selection_calibration']
    assert diagnostic['model_fits']==diagnostic['calibrator_fits']==diagnostic['policy_evaluations']==0
    assert diagnostic['identical_selected_membership']
    for panel,trial in zip(diagnostic['panels'],comparison['trials']):
        raw=panel['raw_metrics']['joint_all']['multiclass_brier_sum_known_only']
        calibrated=trial['metrics']['joint_all']['multiclass_brier_sum_known_only']
        assert panel['calibrated_minus_raw_known_only']['joint_all']['multiclass_brier_sum_known_only']==pytest.approx(calibrated-raw)
    assert verify(tmp_path,out,digest(out/'summary.json'))['policy_comparison_recomputed']
    assert cli.main(['verify-joint-policy-diagnostic','--directory',str(out),'--summary-sha256',digest(out/'summary.json')])==0
    assert json.loads(capsys.readouterr().out)['policy_comparison_recomputed']
    assert before=={p.name:digest(p) for p in directory.iterdir() if p.is_file()}
    coverage_path=out/'risk_coverage.csv';original=coverage_path.read_bytes()
    coverage=pd.read_csv(coverage_path)
    assert set(coverage.scope)=={'all_candidates','selected','rejected'}
    coverage.loc[0,'positive']+=1;coverage.to_csv(coverage_path,index=False)
    original_hash=result['artifacts']['risk_coverage.csv']
    result['artifacts']['risk_coverage.csv']=digest(coverage_path);atomic_json(out/'summary.json',result)
    with pytest.raises(ValueError,match='coverage reconstruction'):verify(tmp_path,out,digest(out/'summary.json'))
    coverage_path.write_bytes(original);result['artifacts']['risk_coverage.csv']=original_hash
    atomic_json(out/'summary.json',result)
    rows=pd.read_parquet(out/'policy_00.parquet')
    assert set(rows.sample_id)=={'selection-policy0','selection-policy1'}
    rows.loc[0,'p_A']+=.01;rows.to_parquet(out/'policy_00.parquet',index=False)
    result['artifacts']['policy_00.parquet']=digest(out/'policy_00.parquet');atomic_json(out/'summary.json',result)
    with pytest.raises(AssertionError):verify(tmp_path,out,digest(out/'summary.json'))


def test_outer_outcome_and_promoted_claim_rejected(tmp_path):
    directory,source=setup(tmp_path)
    result=build(tmp_path,directory,digest(directory/'summary.json'),source,digest(source));out=Path(result['directory'])
    result['formal_H05_accepted']=True;atomic_json(out/'summary.json',result)
    with pytest.raises(ValueError,match='unsupported'):verify(tmp_path,out,digest(out/'summary.json'))
    payload=load_plan(source);payload['selection_outcomes'][0]['sample_id']='outer-test0';atomic_json(source,payload)
    with pytest.raises(ValueError,match='exact mature'):
        build(tmp_path,directory,digest(directory/'summary.json'),source,digest(source))


def test_saved_threshold_frontier_retains_empty_policy(tmp_path):
    directory,source=setup(tmp_path)
    payload=load_plan(source);payload['registry']['comparison_kind']='registered_gate_frontier'
    payload['registry']['policies'][1]['selection']['max_risk']=0
    atomic_json(source,payload)
    result=build(tmp_path,directory,digest(directory/'summary.json'),source,digest(source));out=Path(result['directory'])
    report=load_plan(out/'comparison.json')
    assert report['comparison_kind']=='registered_gate_frontier'
    assert not report['matched_coverage_gain_proven']
    rows=pd.read_parquet(out/'policy_01.parquet');assert not rows.selected.any() and len(rows)==2
    coverage=pd.read_csv(out/'risk_coverage.csv')
    assert set(coverage.comparison_kind)=={'registered_gate_frontier'}
    assert verify(tmp_path,out,digest(out/'summary.json'))['policy_comparison_recomputed']


@pytest.mark.parametrize('artifact',['calibration_states.jsonl','policy_candidates.json','selected_reliability.csv'])
def test_h05_diagnostic_artifacts_rehashed_tampering_rejected(tmp_path,artifact):
    import json
    directory,source=setup(tmp_path)
    payload=load_plan(source);payload['probability_source']='raw';atomic_json(source,payload)
    result=build(tmp_path,directory,digest(directory/'summary.json'),source,digest(source));out=Path(result['directory'])
    state=json.loads((out/'calibration_states.jsonl').read_text(encoding='utf-8'))
    assert state['effective_method']=='identity_raw' and not state['source_calibration_applied']
    assert not state['online_update_history'] and state['additional_calibrator_fits']==0
    reliability=pd.read_csv(out/'selected_reliability.csv')
    assert set(reliability.probability_source)=={'raw'} and set(reliability.policy_id)=={'direct','penalty'}
    assert not reliability.formal_H05_accepted.any()
    # Even harmless-looking appended whitespace is not the canonical snapshot.
    (out/artifact).write_bytes((out/artifact).read_bytes()+b' ')
    result['artifacts'][artifact]=digest(out/artifact);atomic_json(out/'summary.json',result)
    with pytest.raises(ValueError,match='evidence reconstruction'):verify(tmp_path,out,digest(out/'summary.json'))
