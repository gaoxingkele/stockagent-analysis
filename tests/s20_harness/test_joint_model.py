import numpy as np
import pandas as pd
import pytest
from research.s20_harness.joint_model import run
from tests.s20_harness.test_feature_pipeline import fixture


def inputs():
    samples,features,bounds,contract=fixture()
    extra=[]
    for i in range(2,12):
        extra.append(dict(samples.iloc[0],sample_id=f'fit{i}'))
    samples=pd.concat([samples,pd.DataFrame(extra)],ignore_index=True)
    features=pd.concat([features,pd.DataFrame(dict(sample_id=[f'fit{i}' for i in range(2,12)],x=[float(i%4) for i in range(2,12)]))],ignore_index=True)
    labels=pd.DataFrame(dict(sample_id=[f'fit{i}' for i in range(12)],target=[['A','B','C','D'][i%4] for i in range(12)]))
    return samples,features,labels,bounds,contract


def test_joint_simplex_future_firewall_and_missing_candidates():
    args=inputs();args[0].loc[9,'feature_available_at']=None
    out,card=run(*args,target_id='P.joint.v4')
    ready=out.prediction_status.eq('uncalibrated_joint_reference')
    p=out.loc[ready,['p_A','p_B','p_C','p_D']]
    assert np.allclose(p.sum(axis=1),1) and len(out)==20 and ready.sum()==7
    assert np.allclose(out.loc[ready,'p_up'],p.p_A+p.p_B)
    assert np.allclose(out.loc[ready,'p_down5'],p.p_B+p.p_D)
    assert out.loc[out.segment.eq('fit'),'p_A'].isna().all() and pd.isna(out.loc[9,'p_A'])
    assert card['fit_class_counts']==dict(A=3,B=3,C=3,D=3)
    args[1].loc[args[1].sample_id.str.startswith('outer'),'x']=-1e6
    _,second=run(*args,target_id='P.joint.v4')
    assert card['coefficients']==second['coefficients']
    assert not card['absolute_probability_validated']


def test_joint_labels_are_exact_and_versioned():
    args=list(inputs())
    args[2]=pd.concat([args[2],pd.DataFrame(dict(sample_id=['outer-test0'],target=['A']))])
    with pytest.raises(ValueError,match='exact eligible'):run(*args,target_id='P.joint.v4')
    args=list(inputs());args[2].loc[args[2].target.eq('D'),'target']='C'
    with pytest.raises(ValueError,match='all four'):run(*args,target_id='P.joint.v4')
    with pytest.raises(ValueError,match='P.joint.v4'):run(*inputs(),target_id='O.opportunity')
