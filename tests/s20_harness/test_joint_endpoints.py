from pathlib import Path
import pandas as pd
import pytest
from research.s20_harness.joint_endpoints import evaluate,build
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_joint_evaluation import inputs
from tests.s20_harness.test_joint_run import plan


def assess(c,s,j,r,**kwargs):
    return evaluate(c,s,j,r,joint_target_id='P.joint.v4',risk_target_id=kwargs.get('target','P.B10.v4'),
        evaluation_at='2024-06-01T00:00:00Z',calendar=plan()['calendar'])


def test_risk10_has_own_maturity_and_no_probability():
    c,s,j=inputs();j['target']=['B','D'];j['label_available_at']='2024-05-29T00:00:00Z'
    r=j.copy();r['target']=[True,False];r.loc[r.index[1],'label_available_at']='2024-06-02T00:00:00Z'
    rows,m=assess(c,s,j,r)
    assert rows.risk10_status.tolist()==['mature','not_mature_at_evaluation']
    assert rows.selected.tolist()==c.selected.tolist()
    bounds=m['risk10']['selected_candidates']['event_bounds']
    assert bounds['denominator']==2 and bounds['unknown']==1 and bounds['rate_lower']==.5
    assert m['risk10']['selected_candidates']['brier_known_only'] is None
    assert not m['risk10_probability_available'] and not m['formal_G3_passed']


def test_contradiction_and_wrong_endpoint_rejected():
    c,s,j=inputs();r=j.copy();r['target']=[True,False]
    with pytest.raises(ValueError,match='contradicts'):assess(c,s,j,r)
    with pytest.raises(ValueError,match='independent P.B10'):assess(c,s,j,r,target='P.down5.v4')


def test_unknown_joint_not_inferred_from_risk10():
    c,s,j=inputs();r=j.copy();r['target']=[True,False]
    rows,m=assess(c,s,j.iloc[:0],r)
    assert rows.evaluation_class.isna().all() and rows.down5_target.isna().all()
    assert rows.risk10_target.iloc[0]
    c['selected']=False
    _,m=assess(c,s,j.iloc[:0],r.iloc[:0])
    assert m['risk10']['selected_candidates']['event_bounds']['known_only_rate'] is None


@pytest.mark.parametrize('known',[False,True])
def test_saved_joint_endpoints_preserve_all_candidates(tmp_path,known):
    from research.s20_harness.joint_run import build as train
    source=tmp_path/'joint.json';atomic_json(source,plan())
    run=train(tmp_path,source,digest(source));directory=Path(run['directory'])
    data=tmp_path/'endpoints.json'
    joint_labels=[];risk_labels=[]
    if known:
        ids=pd.read_parquet(directory/'candidate_ledger.parquet').sample_id.tolist()
        joint_labels=[dict(sample_id=sid,target=c,label_available_at='2024-05-29T00:00:00Z') for sid,c in zip(ids,['A','D'])]
        risk_labels=[dict(sample_id=sid,target=v,label_available_at='2024-05-29T00:00:00Z') for sid,v in zip(ids,[False,True])]
    atomic_json(data,dict(evaluation_at='2024-06-01T00:00:00Z',joint=dict(target_id='P.joint.v4',outcomes=joint_labels),
                         risk10=dict(target_id='P.B10.v4',outcomes=risk_labels)))
    result=build(tmp_path,directory,digest(directory/'summary.json'),data,digest(data))
    assert result['rows']==2 and result['models_refit']==0
    m=load_plan(Path(result['directory'])/'metrics.json')
    assert m['risk10']['all_candidates']['event_bounds']['unknown']==(0 if known else 2)
    from research.s20_harness.joint_endpoints import verify_evaluation
    out=Path(result['directory'])
    assert verify_evaluation(tmp_path,out,digest(out/'summary.json'))['endpoints_and_metrics_recomputed']
    m['risk10_probability_available']=True
    atomic_json(out/'metrics.json',m)
    result['artifacts']['metrics.json']=digest(out/'metrics.json');atomic_json(out/'summary.json',result)
    with pytest.raises(ValueError,match='metrics reconstruction'):
        verify_evaluation(tmp_path,out,digest(out/'summary.json'))
