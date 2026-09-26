import pytest
from research.s20_harness.suspension_evidence_merge import merge


def test_later_direct_support_does_not_rewrite_original_mirror():
    row=dict(ts_code='a',trade_date='20240101',source_url='mirror',original_review_state='full_day_suspended',
        basis='mirror_pending_primary',host_role='mirror',page_evidence_bound=True,
        historical_prediction_eligible=False,executable_fill_proven=False)
    later=dict(row,source_url='later_notice',basis='direct_disclosure',host_role='disclosure_host')
    result=merge([row],[[later]])[0]
    assert result['retrospective_direct_support']
    assert result['original_document_basis']=='mirror_pending_primary'
    assert result['documents']==[row,later] and not result['historical_prediction_eligible']
    with pytest.raises(ValueError,match='conflicting'):
        merge([row],[[dict(later,original_review_state='resumption_announced_quote_observed')]])
    with pytest.raises(ValueError,match='duplicate source'):
        merge([row],[[row]])
