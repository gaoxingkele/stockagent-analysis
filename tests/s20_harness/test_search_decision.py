import pytest
from research.s20_harness.search_decision import decide
from tests.s20_harness.test_configuration_comparison import row


def summary(values,cap=3):
    cells=[]
    for cid,(safe,risk) in values.items():
        for seed in [20,71]:cells.append(dict(row(seed,safe,risk),candidate_id=cid))
    return dict(registered_candidate_ids=list(values),baseline_id='base',cells=cells,
        registered_rule='matched_cell_robust_dominance_no_automatic_tiebreak',registered_finalist_cap=cap)


def test_shortlist_dominance_no_gain_and_no_tiebreak():
    value=summary({'base':([False],[True]),'a':([True],[False]),'b':([True],[False])},cap=1)
    result=decide(value)
    assert result['status']=='UNRESOLVED_CAPACITY' and result['diagnostic_shortlist']==[]
    value['registered_finalist_cap']=2
    assert decide(value)['diagnostic_shortlist']==['a','b']
    assert not decide(value)['formal_finalists_selected']
    value=summary({'base':([False,False],[True,True]),'a':([True,True],[False,False]),'b':([True,False],[False,True])},cap=1)
    assert decide(value)['diagnostic_shortlist']==['a']
    value=summary({'base':([True],[False]),'a':([True],[False])})
    assert decide(value)['status']=='NO_ROBUST_CANDIDATE'


def test_unknown_empty_coverage_and_missing_cells():
    for safe,risk in [([None],[None]),([],[])]:
        result=decide(summary({'base':(safe,risk),'a':(safe,risk)}))
        assert result['diagnostic_shortlist']==[]
    value=summary({'base':([False],[True]),'a':([True],[False])})
    value['cells'][-1]['daily_selected']=[dict(signal_date='20240504',count=1)]
    assert decide(value)['coverage_unresolved']==['a']
    value['cells'].pop()
    with pytest.raises(ValueError,match='grid'):decide(value)


def test_registered_frontier_retains_baseline_and_abstains_on_capacity():
    value=summary({'base':([True],[False]),'a':([True],[False])},cap=2)
    value['registered_rule']='all_registered_nondominated_no_automatic_tiebreak'
    result=decide(value)
    assert result['diagnostic_shortlist']==['base','a'] and result['baseline_in_comparison']
    value['registered_finalist_cap']=1
    assert decide(value)['status']=='UNRESOLVED_CAPACITY'
    value['cells'][-1]['daily_selected']=[dict(signal_date='different',count=1)]
    assert decide(value)['status']=='UNRESOLVED_COMPARABILITY'
