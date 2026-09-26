import pandas as pd
import pytest
from research.s20_harness.joint_policy import apply


def fixture():
    rows=pd.DataFrame(dict(sample_id=['volatile','safe','risky','missing'],entity_id=['A','B','C','D'],
        signal_date=['20240102']*4,prediction_at=['2024-01-02T21:00:00+08:00']*4,
        p_A=[.3,.7,.72,float('nan')],p_B=[.65,.05,.03,float('nan')],
        p_C=[.04,.23,.05,float('nan')],p_D=[.01,.02,.20,float('nan')]))
    policy=dict(selection=dict(policy_id='joint',target_id='P.safe.v4',risk_target_id='P.down5.v4',
        mode='risk_gated',frozen_at='2024-01-01T00:00:00+08:00',n_cap=1,min_score=.1,max_risk=.3),
        weights={'lambda':1.,'mu':3.,'nu':.1})
    return rows,policy,['20240102','20240103']


def test_high_upside_cannot_offset_risk_and_utility_changes_rank():
    rows,report=apply(*fixture())
    assert rows.loc[rows.selected,'sample_id'].tolist()==['safe']
    assert rows.loc[0,'p_up']==pytest.approx(.95) and rows.loc[0,'reject_reason']=='risk_high'
    assert rows.loc[2,'score']>rows.loc[1,'score'] and rows.loc[2,'utility_rank']<rows.loc[1,'utility_rank']
    assert len(rows)==4 and pd.isna(rows.loc[3,'utility_rank'])
    assert report['daily'][1]['selected']==0 and report['daily'][1]['precision'] is None
    assert not report['utility_is_probability'] and not report['orders_created']


def test_direct_safe_probability_is_distinct_control_with_identical_gate():
    candidates,policy,calendar=fixture()
    penalized,_=apply(candidates,policy,calendar)
    direct=dict(selection=policy['selection'],ranking='safe_probability')
    rows,report=apply(candidates,direct,calendar)
    assert rows.loc[rows.selected,'sample_id'].tolist()==['risky']
    assert rows.loc[0,'reject_reason']=='risk_high'
    assert rows.utility_rank.isna().all()
    assert rows.score.equals(penalized.score) and rows.risk.equals(penalized.risk)
    assert rows.selected.sum()==penalized.selected.sum()==1
    assert report['selection_order']=='hard_down5_gate_then_safe_probability'
    assert report['policy_sha256']!=penalized.policy_sha256.iloc[0]


@pytest.mark.parametrize('bad',['weights','ranking','gate'])
def test_direct_control_rejects_ambiguous_or_ungated_policy(bad):
    rows,policy,calendar=fixture()
    direct=dict(selection=policy['selection'],ranking='safe_probability')
    if bad=='weights':direct['weights']=policy['weights']
    elif bad=='ranking':direct['ranking']='unknown'
    else:direct['selection']['mode']='score_only'
    with pytest.raises(ValueError):apply(rows,direct,calendar)


@pytest.mark.parametrize('bad',['weights','partial','simplex','outcome'])
def test_invalid_joint_policy_rejected(bad):
    rows,policy,calendar=fixture()
    if bad=='weights': policy['weights']['mu']=.5
    elif bad=='partial': rows.loc[0,'p_A']=float('nan')
    elif bad=='simplex': rows.loc[0,'p_A']=.4
    else: rows['future_return']=1.
    with pytest.raises(ValueError):apply(rows,policy,calendar)
