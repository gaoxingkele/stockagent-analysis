import json

import pytest

from research.s20_harness import baseline_run as runner
from research.s20_harness.runtime import atomic_json, digest
from tests.s20_harness.test_baseline_model import inputs


def plan():
    samples, features, fit_labels, boundaries, contract = inputs()
    features["x"] = [1., 3.] * 5
    samples["entity_id"] = ["A", "B"] * 5
    samples["signal_date"] = [f"2024{i:02d}03" for i in range(1, 6) for _ in range(2)]
    return dict(schema_version="1", evidence_mode="synthetic", target_id="P.safe.v4",
                samples=samples.to_dict("records"), features=features.to_dict("records"),
                fit_labels=fit_labels.to_dict("records"),
                calibration_labels=[dict(sample_id="calibration0", target=False), dict(sample_id="calibration1", target=True)],
                boundaries=boundaries, feature_contract=contract,
                policy=dict(policy_id="control1", target_id="P.safe.v4", risk_target_id=None,
                            mode="score_only_control", frozen_at="2024-05-01T00:00:00Z",
                            n_cap=1, min_score=0., max_risk=None), calendar=["20240503", "20240506"])


def test_full_saved_chain_and_tamper(tmp_path):
    path = tmp_path/"plan.json"
    atomic_json(path, plan())
    report = runner.build(tmp_path, path, digest(path))
    out = runner.Path(report["directory"])
    assert report["outer_candidates"] == 2 and report["selected"] == 1
    assert runner.verify(out, digest(out/"summary.json"))["artifact_bytes_verified"]
    atomic_json(out/"baseline_card.json", {})
    with pytest.raises(ValueError, match="artifact pin"):
        runner.verify(out, digest(out/"summary.json"))


def test_shallow_tree_saved_chain_replay_and_actual_split(tmp_path):
    from research.s20_harness.baseline_replay import replay
    value=plan();value['model_family']='shallow_tree';value['random_seed']=71
    for i in range(8):
        key='extra_fit'+str(i)
        value['samples'].append(dict(value['samples'][i%2],sample_id=key))
        value['features'].append(dict(sample_id=key,x=1. if i%2==0 else 3.))
        value['fit_labels'].append(dict(sample_id=key,target=bool(i%2)))
    path=tmp_path/'tree.json';atomic_json(path,value)
    report=runner.build(tmp_path,path,digest(path));out=runner.Path(report['directory'])
    card=json.loads((out/'baseline_card.json').read_text())
    assert card['model']=='fixed_shallow_tree_reference'
    assert card['parameters']['random_state']==71
    assert card['randomness']['requested_seed']==71
    assert len(card['tree_state']['feature'])==3
    assert card['fit_positive_count']==5 and len(card['fit_sample_ids'])==10
    assert report['outer_candidates']==2 and not report['formal_H03_H05_accepted']
    assert replay(tmp_path,out,digest(out/'summary.json'))['semantic_replay_performed']


def test_failed_calibration_keeps_baseline_only(tmp_path):
    value = plan()
    value["calibration_labels"][1]["target"] = False
    path = tmp_path/"plan.json"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="two resolved"):
        runner.build(tmp_path, path, digest(path))
    out = next((tmp_path/"output/experiments/s20_safe_v4/sources").iterdir())
    checkpoint = json.loads((out/"checkpoint.json").read_text())
    assert checkpoint["status"] == "FAILED" and checkpoint["completed_steps"] == ["baseline"]
    assert not (out/"summary.json").exists()


def test_source_mutation_is_not_published(tmp_path, monkeypatch):
    path = tmp_path/"plan.json"
    atomic_json(path, plan())
    original = runner.fit_baseline
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        atomic_json(path, {})
        return result
    monkeypatch.setattr(runner, "fit_baseline", mutate)
    with pytest.raises(ValueError, match="source changed"):
        runner.build(tmp_path, path, digest(path))
    out = next((tmp_path/"output/experiments/s20_safe_v4/sources").iterdir())
    assert not (out/"summary.json").exists()
    assert json.loads((out/"checkpoint.json").read_text())["status"] == "FAILED"
