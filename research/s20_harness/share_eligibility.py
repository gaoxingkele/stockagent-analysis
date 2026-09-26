"""Event-scoped supplied eligibility assertions, not independent legal proof."""
import re
from .portfolio_book import FollowOnEligibility,append,clock
from .label_availability import _instant


def register(book, *, event_id, source_distribution_id, target_distribution, eligible,
             available_at, processing_at, evidence_id, evidence_sha256):
    at=clock(book,event_id,processing_at);instant=_instant(at);available=_instant(available_at)
    day=instant.tz_convert('Asia/Shanghai').strftime('%Y%m%d')
    if available>instant or day>target_distribution.record_date:
        raise ValueError('follow-on eligibility unavailable or record date passed')
    if type(eligible) is not bool: raise ValueError('explicit boolean eligibility required')
    if (not isinstance(evidence_id,str) or not evidence_id.strip()
            or not isinstance(evidence_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',evidence_sha256)):
        raise ValueError('follow-on eligibility evidence required')
    if any(e.get('evidence_id')==evidence_id for e in book.journal): raise ValueError('duplicate eligibility evidence')
    claims=[c for c in book.share_claims if c.distribution.event_id==source_distribution_id]
    if len(claims)!=1 or claims[0].transferred_position_id is None:
        raise ValueError('transferred source shares required')
    c=claims[0]
    if target_distribution.record_date<=c.distribution.record_date or target_distribution.quote_unit!='ordinary_share':
        raise ValueError('invalid follow-on target distribution')
    if any(x.distribution.event_id==target_distribution.event_id for x in (*book.cash_claims,*book.share_claims)):
        raise ValueError('target entitlement already recorded')
    if any(e.source_distribution_id==source_distribution_id and e.target_distribution.event_id==target_distribution.event_id
           for e in book.follow_on_eligibility): raise ValueError('duplicate/conflicting eligibility decision')
    decision=FollowOnEligibility(source_distribution_id,target_distribution,c.transferred_position_id,
        eligible,available.isoformat(),evidence_id,evidence_sha256)
    return append(book,event_id,at,dict(kind='follow_on_eligibility',source_distribution_id=source_distribution_id,
        target_distribution_id=target_distribution.event_id,record_date=target_distribution.record_date,
        position_id=c.transferred_position_id,eligible=eligible,available_at=available.isoformat(),
        evidence_id=evidence_id,evidence_sha256=evidence_sha256,eligibility_independently_verified=False,cash_delta=0.),
        follow_on_eligibility=book.follow_on_eligibility+(decision,))
