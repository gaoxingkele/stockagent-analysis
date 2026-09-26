import pandas as pd
import pytest
from research.s20_harness.fixed_selection_calibration import assess
from research.s20_harness.joint_policy_comparison import compare
from tests.s20_harness.test_joint_policy_comparison import fixture


def test_identity_deltas_zero_and_same_membership():
    args=fixture();ledgers,comparison=compare(*args)
    raw=args[1].copy();raw['segment']='selection-policy'
    for col in ['p_A','p_B','p_C','p_D']:raw['cal_'+col]=raw[col]
    result=assess(args[0],raw,args[2],ledgers,comparison,args[4])
    assert result['policy_evaluations']==0
    for panel in result['panels']:
        assert all(v==0 for scope in panel['calibrated_minus_raw_known_only'].values() for v in scope.values())
        expected=ledgers[panel['policy_id']]
        assert panel['selected_sample_ids']==expected.loc[expected.selected,'sample_id'].tolist()


def test_unknown_scores_not_used_as_known_error_and_foreign_segment_rejected():
    args=list(fixture());args[0].loc[7,'label_available_at']=None;args[2]=args[2].iloc[:1]
    ledgers,comparison=compare(*args);raw=args[1].copy();raw['segment']='selection-policy'
    for col in ['p_A','p_B','p_C','p_D']:raw['cal_'+col]=raw[col]
    result=assess(args[0],raw,args[2],ledgers,comparison,args[4])
    assert result['panels'][0]['calibrated_minus_raw_known_only']['joint_selected']['log_loss_known_only'] is None
    raw.loc[raw.index[0],'segment']='outer-test'
    with pytest.raises(ValueError,match='selection-only'):assess(args[0],raw,args[2],ledgers,comparison,args[4])
