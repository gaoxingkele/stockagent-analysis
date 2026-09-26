import pytest
from research.s20_harness.feature_strata import summarize,validate
from tests.s20_harness.test_joint_run import plan
from tests.s20_harness.test_joint_evaluation import inputs,assess


def test_unavailable_and_unknown_rows_retained_without_future_grouping():
    value=plan();rows,_=assess(inputs());controls=[dict(feature='x',cuts=[1.5,2.])]
    result=summarize(value,rows,controls)
    assert sum(r['candidates'] for r in result['strata'])==2
    assert sum(r['events']['safe_profit']['unknown'] for r in result['strata'])==1
    unavailable=next(r for r in result['strata'] if r['stratum']=='unavailable')
    assert unavailable['candidates']==1
    for row in value['features']:
        if row['sample_id']=='outer-test1':row['x']=9999
    assert summarize(value,rows,controls)==result


def test_strict_prior_availability_and_cutpoint_edges():
    value=plan();rows,_=assess(inputs());controls=[dict(feature='x',cuts=[1.5])]
    for feature in value['features']:
        if feature['sample_id']=='outer-test0':feature['x']=1.5
    assert any(s['stratum']=='x=1' for s in summarize(value,rows,controls)['strata'])
    for sample in value['samples']:
        if sample['sample_id']=='outer-test0':sample['feature_available_at']=sample['prediction_at']
    result=summarize(value,rows,controls)
    assert len(result['strata'])==1 and result['strata'][0]['stratum']=='unavailable'
    assert result['strata'][0]['candidates']==2


@pytest.mark.parametrize('controls',[[dict(feature='z',cuts=[1])],[dict(feature='x',cuts=[2,1])],
    [dict(feature='x',cuts=[True])],[dict(feature='x',cuts=[float('inf')])],[]])
def test_invalid_or_nonbase_controls(controls):
    with pytest.raises(ValueError):validate(controls,['x'])
