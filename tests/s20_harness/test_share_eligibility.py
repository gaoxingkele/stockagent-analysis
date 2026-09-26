from dataclasses import asdict,replace

import pytest

from research.s20_harness.corporate_actions import Distribution
from research.s20_harness.share_eligibility import register
from research.s20_harness.portfolio_cash_actions import record_cash_claim
from research.s20_harness.portfolio_share_actions import record_share_claim,allocate_share_basis,transfer_share_position
from research.s20_harness.portfolio_replay import step
from tests.s20_harness.test_share_basis_allocation import credit,basis
from tests.s20_harness.test_share_position_transfer import transfer


def book(): return transfer_share_position(allocate_share_basis(credit(),**basis()),**transfer())


def target(cash=True):
    return Distribution('next','20240730','20240731','20240701',cash_per_share=.2 if cash else 0,
        bonus_per_share=0 if cash else .1,pay_date='20240801' if cash else None,
        bonus_list_date=None if cash else '20240801',beneficiary_scope='existing_shareholders_verified')


def eligibility(**changes):
    a=dict(event_id='follow-on',source_distribution_id='share1',target_distribution=target(),eligible=True,
        available_at='2024-07-30T10:30:00+08:00',processing_at='2024-07-30T11:00:00+08:00',
        evidence_id='synthetic-follow-on',evidence_sha256='1'*64)
    a.update(changes);return a


def record(b,dist):
    a=dict(event_id='next-record',ts_code='stock',distribution=dist,processing_at='2024-07-30T15:05:00+08:00',
        terms_available_at='2024-07-01T12:00:00+08:00',source_id='synthetic')
    if dist.cash_per_share: return record_cash_claim(b,**a)
    return record_share_claim(b,**a,source_sha256='2'*64)


@pytest.mark.parametrize('cash',[True,False])
@pytest.mark.parametrize('allowed,quantity',[(True,110),(False,100)])
def test_event_specific_eligibility_includes_or_excludes_only_transferred_lot(cash,allowed,quantity):
    original=book();dist=target(cash)
    with pytest.raises(ValueError,match='unresolved share eligibility'): record(original,dist)
    b=register(original,**eligibility(target_distribution=dist,eligible=allowed))
    b=record(b,dist)
    claim=b.cash_claims[-1] if cash else b.share_claims[-1]
    assert claim.record_quantity==quantity
    if not cash:
        assert sum(q for _,q in claim.record_lots)==quantity
        assert ('bonus',10) in claim.record_lots if allowed else ('bonus',10) not in claim.record_lots
    assert original.follow_on_eligibility==()


def test_evidence_cannot_be_reused_for_changed_target_terms():
    b=register(book(),**eligibility())
    with pytest.raises(ValueError,match='unresolved share eligibility'):
        record(b,replace(target(),cash_per_share=.3))
    with pytest.raises(ValueError): register(b,**eligibility(event_id='other',evidence_id='other',eligible=False))


@pytest.mark.parametrize('changes',[
    dict(eligible=1),dict(evidence_id=''),dict(evidence_sha256='bad'),dict(source_distribution_id='missing'),
    dict(available_at='2024-07-31T10:30:00+08:00'),dict(processing_at='2024-07-31T11:00:00+08:00')])
def test_invalid_decisions_rejected(changes):
    b=book()
    with pytest.raises(ValueError): register(b,**eligibility(**changes))
    assert b.follow_on_eligibility==()


def test_replay_dispatch_preserves_complete_target_terms():
    a=eligibility();a['target_distribution']=asdict(a['target_distribution'])
    b=step(book(),dict(kind='follow_on_eligibility',args=a))
    assert b.follow_on_eligibility[0].target_distribution==target()
    assert record(b,target()).cash_claims[-1].gross_amount==22
