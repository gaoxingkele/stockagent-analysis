import pytest
from research.s20_harness.suspension_notice_findings import validate


def test_mirror_anchor_and_candidate_denominator_guards():
    case=dict(ts_code='a',trade_date='20240101',source_url='url',host_role='mirror',document_identity_matched=True)
    pages=[dict(url='url',pages=[dict(page=1,text='continues suspension')])]
    finding=dict(ts_code='a',trade_date='20240101',basis='mirror_pending_primary',
        evidence=[dict(page=1,anchor='continuessuspension')],finding='review',limit='mirror only')
    result=validate([case],pages,[finding])
    assert result[0]['page_evidence_bound'] and not result[0]['retrospective_direct_disclosure_supported']
    with pytest.raises(ValueError,match='mirror'):
        validate([case],pages,[dict(finding,basis='direct_disclosure')])
    with pytest.raises(ValueError,match='anchor'):
        validate([case],pages,[dict(finding,evidence=[dict(page=2,anchor='continuessuspension')])])
    with pytest.raises(ValueError,match='denominator'):
        validate([case],pages,[])
