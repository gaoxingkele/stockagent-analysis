import numpy as np
import pandas as pd
import pytest
from research.s20_harness.conditional_joint_model import run,combine
from tests.s20_harness.test_joint_model import inputs


def test_factorization_not_marginal_independence():
    p=combine([.9],[.8],[.1])
    np.testing.assert_allclose(p,[[.18,.72,.09,.01]])
    assert np.allclose(p.sum(axis=1),1)
    for args in [([.5],[1.1],[.5]),([.5],[],[.5]),([np.nan],[.5],[.5])]:
        with pytest.raises(ValueError):combine(*args)


def test_conditional_heads_membership_and_future_firewall():
    args=inputs();args[0].loc[9,'feature_available_at']=None
    output,card=run(*args,target_id='P.joint.v4')
    assert len(output)==20 and output.p_A.notna().sum()==7
    assert output.loc[output.segment.eq('fit'),'p_A'].isna().all()
    counts={k:len(v['fit_sample_ids']) for k,v in card['heads'].items()}
    assert counts=={'up':12,'down_if_up':6,'down_if_not_up':6}
    assert set(card['heads']['down_if_up']['fit_sample_ids']).isdisjoint(card['heads']['down_if_not_up']['fit_sample_ids'])
    p=output.loc[output.p_A.notna(),['p_A','p_B','p_C','p_D']]
    np.testing.assert_allclose(p.sum(axis=1),1)
    assert card['underlying_fits']==3 and card['model_level_fits']==1
    args[1].loc[args[1].sample_id.str.startswith('outer'),'x']=-100000.
    _,changed=run(*args,target_id='P.joint.v4')
    assert changed['heads']==card['heads']
    assert not card['conditional_sample_sufficiency_proven']
    from research.s20_harness.joint_calibration import run as calibrate
    labels=pd.DataFrame([dict(sample_id='calibration0',target='A'),dict(sample_id='calibration1',target='D')])
    calibrated,calcard=calibrate(args[0],output,labels,args[3],card,target_id='P.joint.v4')
    available=calibrated.cal_p_A.notna()
    np.testing.assert_allclose(calibrated.loc[available,['cal_p_A','cal_p_B','cal_p_C','cal_p_D']].sum(axis=1),1)
    assert calcard['calibrator_fits']==1 and not calcard['absolute_probability_validated']


def test_conditional_rejects_foreign_and_missing_classes():
    args=list(inputs());args[2]=pd.concat([args[2],pd.DataFrame([dict(sample_id='outer-test0',target='A')])])
    with pytest.raises(ValueError,match='exact eligible'):run(*args,target_id='P.joint.v4')
    args=list(inputs());args[2].loc[args[2].target.eq('D'),'target']='C'
    with pytest.raises(ValueError,match='all four'):run(*args,target_id='P.joint.v4')
