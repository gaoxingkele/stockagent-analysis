import copy
import pytest
from research.s20_harness.calibration_policy_contrast import compare_reports
from research.s20_harness.joint_policy_comparison import compare
from tests.s20_harness.test_joint_policy_comparison import fixture


def reports():
    _,a=compare(*fixture());b=copy.deepcopy(a)
    a['probability_source']='raw';b['probability_source']='calibrated'
    return a,b


def test_same_coverage_bounds_and_no_winner():
    result=compare_reports(*reports())
    assert all(p['same_daily_coverage'] for p in result['policies'])
    assert result['policies'][0]['events']['safe_profit']['calibrated_minus_raw_bounds']==[0,0]
    assert not result['automatic_winner_selected'] and not result['bounds_are_confidence_intervals']


def test_different_coverage_is_not_matched_gain():
    a,b=reports();b['trials'][0]['selection']['daily'][0]['selected']=0
    result=compare_reports(a,b)
    assert result['policies'][0]['status']=='COVERAGE_TRADEOFF'
    assert all(e['calibrated_minus_raw_bounds'] is None for e in result['policies'][0]['events'].values())


@pytest.mark.parametrize('bad',['source','registry','split','cutoff','policy'])
def test_incompatible_reports_rejected(bad):
    a,b=reports()
    if bad=='source':b['probability_source']='raw'
    elif bad=='registry':b['registry']['registry_id']='changed'
    elif bad=='split':b['split']={}
    elif bad=='cutoff':b['label_cutoff']='2025-01-01T00:00:00Z'
    else:b['trials'][0]['policy_sha256']='0'*64
    with pytest.raises(ValueError):compare_reports(a,b)
