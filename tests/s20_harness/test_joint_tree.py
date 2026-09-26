import copy
import numpy as np
import pytest
from research.s20_harness.joint_model import run
from research.s20_harness.joint_tree_inference import predict
from research.s20_harness.feature_pipeline import prepare
from tests.s20_harness.test_joint_model import inputs


def test_actual_tree_split_replayed_and_future_features_do_not_leak():
    args=inputs();out,card=run(*args,target_id='P.joint.v4',model_family='shallow_joint',random_seed=71)
    assert card['model']=='fixed_shallow_joint_tree' and len(card['tree_state']['left'])>1
    matrix,assignment,pre=prepare(args[0],args[1],args[3],args[4])
    ready=out.prediction_status.eq('uncalibrated_joint_reference')
    x=matrix.loc[ready,pre['feature_columns']].to_numpy()
    np.testing.assert_allclose(predict(card['tree_state'],x),out.loc[ready,['p_A','p_B','p_C','p_D']])
    args[1].loc[~args[1].sample_id.str.startswith('fit'),'x']=-999.
    _,other=run(*args,target_id='P.joint.v4',model_family='shallow_joint',random_seed=71)
    assert other['tree_state']==card['tree_state']
    assert card['randomness']['seed_effect']=='feature_permutation_and_split_ties'
    for kind in ['cycle','value','feature']:
        tree=copy.deepcopy(card['tree_state'])
        if kind=='cycle':tree['left'][0]=0
        elif kind=='value':tree['values'][0][0]=-1
        else:tree['feature'][0]=999
        with pytest.raises(ValueError):predict(tree,x)
