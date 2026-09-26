import copy
from pathlib import Path
import pytest
from research.s20_harness.feature_group_ablation import plans
from research.s20_harness.joint_run import build
from research.s20_harness.joint_replay import replay
from research.s20_harness.runtime import atomic_json,digest,load_plan
from tests.s20_harness.test_joint_run import plan


def source():
    value=plan();value['feature_contract']['columns']=['x','z']
    for i,row in enumerate(value['features']):row['z']=float(i%3)
    return value


def test_three_controls_train_and_replay_on_identical_scope(tmp_path):
    original=source();before=copy.deepcopy(original)
    variants,report=plans(original,{'base':['x'],'addition':['z']},['base'],'addition')
    assert original==before and report['total_initial_costs']==dict(model_fits=3,underlying_fits=6,calibrator_fits=3,policy_evaluations=3)
    assert not report['budget_reserved'] and not report['independent_information_proven']
    for name,value in variants.items():
        for key in original:
            if key not in ['features','feature_contract']:assert value[key]==original[key]
        path=tmp_path/(name+'.json');atomic_json(path,value)
        result=build(tmp_path,path,digest(path));out=Path(result['directory'])
        assert replay(out,digest(out/'summary.json'))['saved_inference_recomputed']
        card=load_plan(out/'joint_card.json')
        assert set(card['preprocessing']['feature_columns'])==({'x'} if name=='base' else {'x','z'})


def test_noise_does_not_depend_on_future_values_or_order():
    original=source();a,_=plans(original,{'base':['x'],'addition':['z']},['base'],'addition')
    for row in original['features']:row['z']=999999.
    original['features'].reverse()
    b,_=plans(original,{'base':['x'],'addition':['z']},['base'],'addition')
    keyed=lambda v:{r['sample_id']:r['z'] for r in v['noise_control']['features']}
    assert keyed(a)==keyed(b)


@pytest.mark.parametrize('bad',['overlap','missing','empty_base','same_group','live','seed'])
def test_invalid_ablation_contract(bad):
    value=source();groups={'base':['x'],'addition':['z']};base=['base'];added='addition';seed=20
    if bad=='overlap':groups['addition']=['x','z']
    elif bad=='missing':groups['addition']=['other']
    elif bad=='empty_base':base=[]
    elif bad=='same_group':added='base'
    elif bad=='live':value['evidence_mode']='supplied_reference'
    else:seed=True
    with pytest.raises(ValueError):plans(value,groups,base,added,noise_seed=seed)
