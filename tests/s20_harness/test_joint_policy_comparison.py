import copy
import pandas as pd
import pytest
from research.s20_harness.joint_policy_comparison import compare
from tests.s20_harness.test_policy_search import fixture as binary_fixture


def fixture():
    samples,candidates,outcomes,boundaries,calendar,_=binary_fixture()
    candidates=candidates.drop(columns=['score','risk'])
    for col,values in zip(['p_A','p_B','p_C','p_D'],[[.7,.72],[.05,.03],[.23,.05],[.02,.2]]):
        candidates[col]=values
    # Same-day candidates make the ranking contrast observable.
    candidates.loc[:,'signal_date']=calendar[0]
    candidates.loc[:,'prediction_at']=candidates.prediction_at.iloc[0]
    samples.loc[7,'signal_date']=samples.loc[6,'signal_date']
    samples.loc[7,'prediction_at']=samples.loc[6,'prediction_at']
    selection=dict(policy_id='direct',target_id='P.safe.v4',risk_target_id='P.down5.v4',mode='risk_gated',
        frozen_at='2024-03-01T00:00:00Z',n_cap=1,min_score=.1,max_risk=.3)
    policies=[dict(selection=selection,ranking='safe_probability'),
              dict(selection=dict(selection,policy_id='penalty'),weights={'lambda':1.,'mu':3.,'nu':.1})]
    outcomes['target']=['A','D']
    registry=dict(registry_id='joint-ablation',target_id='P.joint.v4',registered_at='2024-03-02T00:00:00Z',policies=policies)
    return samples,candidates,outcomes,boundaries,calendar,registry


def test_fixed_prediction_ranking_ablation_keeps_all_rows_and_empty_days():
    ledgers,report=compare(*fixture())
    assert report['policy_evaluations']==2 and report['model_fits']==report['calibrator_fits']==0
    assert report['same_predictions_and_daily_coverage']
    assert ledgers['direct'].loc[lambda x:x.selected,'evaluation_class'].tolist()==['D']
    assert ledgers['penalty'].loc[lambda x:x.selected,'evaluation_class'].tolist()==['A']
    assert all(len(frame)==2 for frame in ledgers.values())
    assert all(t['selection']['daily'][1]['selected']==0 for t in report['trials'])
    assert report['selected_policy_id'] is None and not report['shared_budget_enforced']


@pytest.mark.parametrize('bad',['outer','gate','late','missing','future_feature','many','no_control','maturity'])
def test_comparison_firewall(bad):
    args=list(fixture())
    if bad=='outer':args[2].loc[args[2].index[0],'sample_id']='outer-test0'
    elif bad=='gate':args[5]['policies'][1]['selection']['max_risk']=.2
    elif bad=='late':args[5]['registered_at']='2024-05-01T00:00:00Z'
    elif bad=='missing':args[1]=args[1].iloc[:1]
    elif bad=='future_feature':args[0].loc[6,'feature_available_at']='2025-01-01T00:00:00Z'
    elif bad=='many':args[5]['policies']*=4
    elif bad=='no_control':args[5]['policies'][0]=copy.deepcopy(args[5]['policies'][1])
    else:args[2].loc[args[2].index[0],'label_available_at']='2024-04-29T00:00:00Z'
    with pytest.raises(ValueError):compare(*args)


def test_unmatured_outcome_retained_not_false_failure():
    args=list(fixture());args[0].loc[7,'label_available_at']=None
    args[2]=args[2].iloc[:1]
    _,report=compare(*args)
    direct=report['trials'][0]['metrics']['events']['safe_profit']['selected_candidates']['event_bounds']
    assert direct['denominator']==direct['unknown']==1 and direct['known']==0


def test_registered_threshold_frontier_can_abstain_without_claiming_matched_gain():
    args=list(fixture());args[-1]['comparison_kind']='registered_gate_frontier'
    args[-1]['policies'][1]['selection']['max_risk']=.01
    ledgers,report=compare(*args)
    assert report['same_predictions'] and not report['same_predictions_and_daily_coverage']
    assert not report['matched_coverage_gain_proven'] and report['selected_policy_id'] is None
    assert ledgers['direct'].selected.sum()==1 and ledgers['penalty'].selected.sum()==0
    bound=report['trials'][1]['metrics']['events']['down5']['selected_candidates']['event_bounds']
    assert bound['rate_lower'] is None and bound['denominator']==0
    assert len(ledgers['penalty'])==2 and report['policy_evaluations']==2


@pytest.mark.parametrize('bad',['unknown_kind','target','freeze','invalid_threshold'])
def test_frontier_only_allows_explicit_decision_parameters(bad):
    args=list(fixture());args[-1]['comparison_kind']='registered_gate_frontier'
    selection=args[-1]['policies'][1]['selection']
    if bad=='unknown_kind':args[-1]['comparison_kind']='automatic_search'
    elif bad=='target':selection['target_id']='O.touch'
    elif bad=='freeze':selection['frozen_at']='2024-02-01T00:00:00Z'
    else:selection['max_risk']=-.1
    with pytest.raises(ValueError):compare(*args)
