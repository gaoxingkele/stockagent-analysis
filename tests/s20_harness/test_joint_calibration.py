import numpy as np
import pandas as pd
import pytest
from research.s20_harness.joint_model import run as train
from research.s20_harness.joint_calibration import run
from tests.s20_harness.test_joint_model import inputs


def test_temperature_forward_only_and_outer_not_fitted():
    args=inputs();args[0].loc[9,'feature_available_at']=None
    predictions,card=train(*args,target_id='P.joint.v4')
    labels=pd.DataFrame(dict(sample_id=['calibration0','calibration1'],target=['A','D']))
    out,cal=run(args[0],predictions,labels,args[3],card,target_id='P.joint.v4')
    cols=['cal_p_A','cal_p_B','cal_p_C','cal_p_D']
    assert out.loc[:5,cols].isna().all().all() and out.loc[9,cols].isna().all()
    assert np.allclose(out.loc[6:8,cols].sum(axis=1),1) and cal['calibrated_rows']==3
    altered=predictions.copy();mask=altered.segment.eq('outer-test')&altered.feature_ready
    altered.loc[mask,['p_A','p_B','p_C','p_D']]=[.1,.2,.3,.4]
    altered.loc[mask,'p_up']=.3;altered.loc[mask,'p_down5']=.6
    _,second=run(args[0],altered,labels,args[3],card,target_id='P.joint.v4')
    assert cal['temperature']==second['temperature']
    leaked=pd.concat([labels,pd.DataFrame(dict(sample_id=['outer-test0'],target=['A']))])
    with pytest.raises(ValueError,match='eligible joint calibration'):
        run(args[0],predictions,leaked,args[3],card,target_id='P.joint.v4')
    altered.loc[0,'p_A']=.1
    with pytest.raises(ValueError,match='availability/simplex'):
        run(args[0],altered,labels,args[3],card,target_id='P.joint.v4')


def test_identity_no_optimizer_no_labels_exact_zero_one(monkeypatch):
    from research.s20_harness import joint_calibration
    args=inputs();predictions,card=train(*args,target_id='P.joint.v4')
    later=predictions.segment.isin(['selection-policy','outer-test'])&predictions.feature_ready
    predictions.loc[later,['p_A','p_B','p_C','p_D']]=[1.,0.,0.,0.]
    predictions.loc[later,'p_up']=1.;predictions.loc[later,'p_down5']=0.
    monkeypatch.setattr(joint_calibration,'minimize_scalar',lambda *a,**k:pytest.fail('identity must not fit'))
    empty=pd.DataFrame(columns=['sample_id','target'])
    result,cal=run(args[0],predictions,empty,args[3],card,target_id='P.joint.v4',method='identity_raw')
    assert cal['calibrator_fits']==0 and cal['calibration_sample_ids']==[]
    assert result.loc[later,'cal_p_A'].eq(1).all() and result.loc[later,'cal_p_B'].eq(0).all()
    assert result.loc[~later,'cal_p_A'].isna().all()
    labels=pd.DataFrame([dict(sample_id='calibration0',target='A')])
    with pytest.raises(ValueError,match='empty labels'):
        run(args[0],predictions,labels,args[3],card,target_id='P.joint.v4',method='identity_raw')
