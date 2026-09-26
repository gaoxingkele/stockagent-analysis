import json
from pathlib import Path
import pytest
from research.s20_harness.joint_run import build as train
from research.s20_harness.joint_endpoints import build as evaluate
from research.s20_harness.joint_paired_comparison import aggregate_bound,build,verify
from research.s20_harness.runtime import atomic_json,digest
from tests.s20_harness.test_joint_binary_scope import joint_plan


def test_two_real_saved_folds_aggregate_and_reconstruct(tmp_path):
    folds=[]
    for year in [2024,2025]:
        plan=json.loads(json.dumps(joint_plan()).replace('2024',str(year)))
        source=tmp_path/f'plan{year}.json';atomic_json(source,plan)
        trained=train(tmp_path,source,digest(source));parent=Path(trained['directory'])
        labels=tmp_path/f'labels{year}.json'
        atomic_json(labels,dict(evaluation_at='2025-06-01T00:00:00Z',
            joint=dict(target_id='P.joint.v4',outcomes=[]),risk10=dict(target_id='P.B10.v4',outcomes=[])))
        evaluated=evaluate(tmp_path,parent,digest(parent/'summary.json'),labels,digest(labels))
        out=Path(evaluated['directory']);ref=dict(directory=str(out),summary_sha256=digest(out/'summary.json'))
        folds.append(dict(fold_id=str(year),candidate=ref,baseline=ref))
    result=aggregate_bound(tmp_path,folds,draws=200)
    assert result['aggregation']['outer_folds']==2 and result['aggregation']['unique_calendar_days']==4
    assert result['aggregation']['pooled_confidence_interval'] is None
    assert not result['formal_G3_passed'] and result['models_refit']==0
    with pytest.raises(ValueError,match='overlapping outer dates'):
        aggregate_bound(tmp_path,[folds[0],dict(folds[0],fold_id='duplicate-seed')],draws=200)
    with pytest.raises(ValueError,match='chronologically'):
        aggregate_bound(tmp_path,list(reversed(folds)),draws=200)
    source=tmp_path/'aggregate.json'
    atomic_json(source,dict(schema_version='2',folds=folds,draws=200,seed=20))
    saved=build(tmp_path,source,digest(source));out=Path(saved['directory'])
    assert verify(tmp_path,out,digest(out/'summary.json'))['comparison_recomputed']
