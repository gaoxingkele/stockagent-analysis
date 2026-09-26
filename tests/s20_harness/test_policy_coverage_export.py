import pandas as pd
from research.s20_harness.policy_coverage_export import table
from research.s20_harness.joint_policy_comparison import compare
from tests.s20_harness.test_joint_policy_comparison import fixture


def test_rejected_opportunities_and_empty_calendar_days_are_visible():
    result=table(*compare(*fixture()))
    assert len(result)==2*3*3*3
    overall=result.query("signal_date == 'all' and event == 'safe_profit'")
    direct=overall.query("policy_id == 'direct'").set_index('scope')
    assert direct.loc['selected','positive']==0 and direct.loc['rejected','positive']==1
    assert direct.loc['all_candidates','denominator']==2
    assert direct.loc['selected','active_day_coverage']==.5
    empty=result.query("signal_date == '20240404'")
    assert empty.denominator.eq(0).all() and empty.event_rate_lower.isna().all()
    assert empty.empty_days.eq(1).all()
    assert not result.risk10_evaluated.any() and not result.formal_H05_accepted.any()


def test_unknown_accounting_conserves_full_denominator():
    args=list(fixture());args[0].loc[7,'label_available_at']=None;args[2]=args[2].iloc[:1]
    result=table(*compare(*args))
    for _,group in result.groupby(['policy_id','signal_date','event']):
        groups=group.set_index('scope')
        for metric in ['denominator','positive','unknown','known']:
            assert groups.loc['all_candidates',metric]==groups.loc['selected',metric]+groups.loc['rejected',metric]
    direct=result.query("signal_date == 'all' and policy_id == 'direct' and scope == 'selected'")
    assert direct.unknown.eq(1).all() and direct.event_rate_known_only.isna().all()
