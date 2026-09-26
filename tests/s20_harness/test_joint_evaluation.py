from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research.s20_harness.joint_evaluation import evaluate,build
from research.s20_harness.joint_run import build as train
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_baseline_evaluation import fixture
from tests.s20_harness.test_joint_run import plan


def inputs():
    c,s,o=fixture()
    for col,values in zip(['p_A','p_B','p_C','p_D'],[[.6,.1],[.2,.5],[.1,.1],[.1,.3]]):c[col]=values
    o['target']=['A','B']
    return c,s,o


def assess(args):
    return evaluate(*args,target_id='P.joint.v4',evaluation_at='2024-06-01T00:00:00Z',calendar=plan()['calendar'])


def test_joint_maturity_event_meanings_and_unknown_bounds():
    args=inputs();rows,m=assess(args)
    assert rows.evaluation_class.iloc[0]=='A' and pd.isna(rows.evaluation_class.iloc[1])
    assert m['events']['down5']['selected_candidates']['event_bounds']['rate_upper']==.5
    for event in ('safe_profit', 'up', 'down5'):
        panel = m['events'][event]['selected_candidates']
        assert sum(b['scored_count'] for b in panel['reliability_bins']) == 2
        assert sum(b['known_count'] for b in panel['reliability_bins']) == 1
        assert sum(b['event_bounds']['unknown'] for b in panel['reliability_bins']) == 1
    args[2]['label_available_at']='2024-05-29T00:00:00Z'
    rows,m=assess(args)
    assert rows.safe_profit_target.tolist()==[True,False]
    assert rows.up_target.tolist()==[True,True] and rows.down5_target.tolist()==[False,True]
    assert m['joint_selected']['class_counts']=={'A':1,'B':1}
    assert m['joint_selected']['multiclass_brier_sum_known_only']==pytest.approx(.29)
    assert not m['risk10_evaluated'] and not m['probability_reliability_proven']


def test_joint_empty_selection_missing_predictions_and_outcomes():
    c,s,o=inputs();c['selected']=False
    c[['p_A','p_B','p_C','p_D']]=np.nan
    rows,m=assess((c,s,o.iloc[:0]))
    assert len(rows)==2 and m['joint_all']['unknown_outcomes']==2
    assert m['joint_selected']['log_loss_known_only'] is None
    assert m['events']['up']['selected_candidates']['event_bounds']['known_only_rate'] is None


@pytest.mark.parametrize('kind',['class','partial','simplex','early','foreign','selected_missing'])
def test_invalid_joint_evaluation(kind):
    c,s,o=inputs()
    if kind=='class':o['target']=True
    elif kind=='partial':c.loc[c.index[0],'p_A']=np.nan
    elif kind=='simplex':c['p_A']=.9
    elif kind=='early':o['label_available_at']='2024-05-10T00:00:00Z'
    elif kind=='foreign':o.loc[o.index[0],'sample_id']='fit0'
    else:c[['p_A','p_B','p_C','p_D']]=np.nan
    with pytest.raises(ValueError):assess((c,s,o))


def test_saved_joint_evaluation_is_separate_no_fit(tmp_path):
    source=tmp_path/'input.json';atomic_json(source,plan())
    run=train(tmp_path,source,digest(source));directory=Path(run['directory'])
    outcome=tmp_path/'outcomes.json'
    atomic_json(outcome,dict(target_id='P.joint.v4',evaluation_at='2024-06-01T00:00:00Z',outcomes=[]))
    result=build(tmp_path,directory,digest(directory/'summary.json'),outcome,digest(outcome))
    assert result['rows']==2 and result['model_fits']==result['calibrator_fits']==0
    metrics=load_plan(Path(result['directory'])/'metrics.json')
    assert metrics['joint_all']['unknown_outcomes']==2
