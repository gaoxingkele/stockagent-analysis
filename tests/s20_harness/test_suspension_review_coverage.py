import pandas as pd
import pytest
from research.s20_harness.suspension_review_coverage import attach


def test_retains_multiple_provider_rows_and_unknown_coverage():
    overlay=pd.DataFrame(dict(ts_code=['a','a','b'],trade_date=['20240101']*3,
        suspend_type=['S','R','S'],review_state=['full_day_suspended']*2+['not_event_reviewed']))
    case=dict(ts_code='a',trade_date='20240101',original_review_state='full_day_suspended',
        historical_prediction_eligible=False,executable_fill_proven=False,
        support_status='direct_disclosure_supported',retrospective_direct_support=True,documents=[])
    result=attach(overlay,[case])
    pd.testing.assert_frame_equal(result[overlay.columns],overlay)
    assert result.notice_direct_support.iloc[:2].all()
    assert pd.isna(result.notice_direct_support.iloc[2])
    assert result.notice_support_status.iloc[2]=='not_notice_reviewed'
    with pytest.raises(ValueError,match='denominator'):
        attach(overlay,[])
    with pytest.raises(ValueError,match='contradicts'):
        attach(overlay,[dict(case,original_review_state='resumption_announced_quote_observed')])
