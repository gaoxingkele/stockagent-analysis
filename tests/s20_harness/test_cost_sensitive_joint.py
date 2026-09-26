from pathlib import Path
import numpy as np
import pytest
from research.s20_harness.joint_model import run
from research.s20_harness.joint_run import build,pipeline_costs
from research.s20_harness.joint_replay import replay
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_joint_model import inputs
from tests.s20_harness.test_joint_binary_scope import joint_plan


def test_weights_normalize_without_changing_total_regularization_scale():
    args=inputs();_,base=run(*args,target_id='P.joint.v4')
    _,equal=run(*args,target_id='P.joint.v4',class_weights=dict(A=3.,B=3.,C=3.,D=3.))
    np.testing.assert_allclose(base['coefficients'],equal['coefficients'],rtol=0,atol=0)
    weights=dict(A=1.,B=2.,C=1.,D=3.)
    _,weighted=run(*args,target_id='P.joint.v4',class_weights=weights)
    assert weighted['training_weight_evidence']['fit_mean_weight']==1.75
    assert not np.allclose(base['coefficients'],weighted['coefficients'])
    args[1].loc[args[1].sample_id.str.startswith('outer'),'x']=-1e6
    _,changed=run(*args,target_id='P.joint.v4',class_weights=weights)
    assert changed['coefficients']==weighted['coefficients']


@pytest.mark.parametrize('weights',[{},dict(A=0,B=1,C=1,D=1),dict(A=True,B=1,C=1,D=1),dict(A=1,B=1,C=1,D=float('inf'))])
def test_invalid_weights_reject(weights):
    with pytest.raises(ValueError,match='weights'):run(*inputs(),target_id='P.joint.v4',class_weights=weights)


def test_saved_cost_sensitive_chain_replays_and_calibrates(tmp_path):
    value=joint_plan();value.update(model_family='cost_sensitive_joint',class_weights=dict(A=1.,B=2.,C=1.,D=3.))
    source=tmp_path/'weighted.json';atomic_json(source,value)
    result=build(tmp_path,source,digest(source));out=Path(result['directory'])
    assert result['underlying_fits']==2 and pipeline_costs(value)['model_fits']==1
    assert replay(out,digest(out/'summary.json'))['model_fits']==0
    assert load_plan(out/'joint_card.json')['model']=='fixed_cost_sensitive_multinomial'
    wrong=dict(value,model_family='multinomial')
    with pytest.raises(ValueError,match='require cost-sensitive'):pipeline_costs(wrong)
