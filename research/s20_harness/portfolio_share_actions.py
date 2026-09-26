"""Record ordinary-share entitlements without inventing receipt or tradability.

Contractual fractional quantities remain explicit; no rounding, price, cost
allocation, tax, CDR conversion or automatic listing-date delivery is inferred.
Explicit credit receipts are retained separately from tradable positions.
Cost allocation, fractional settlement and sellability remain separate work.
"""
import re
from dataclasses import replace
from decimal import Decimal

import pandas as pd

from .exit_book import positive
from .label_availability import _instant
from .portfolio_book import CashClaim, ShareClaim, ShareReceipt, FractionalSettlement, append, clock, record_date_lots, share_contract_residual


def record_share_claim(book, *, event_id, ts_code, distribution, processing_at,
                       terms_available_at, source_id, source_sha256):
    at = clock(book, event_id, processing_at)
    instant, available = _instant(at), _instant(terms_available_at)
    day = instant.tz_convert('Asia/Shanghai').strftime('%Y%m%d')
    if day != distribution.record_date or instant < pd.Timestamp(
            distribution.record_date+' 15:00', tz='Asia/Shanghai'):
        raise ValueError('record-date close snapshot required')
    if available > instant:
        raise ValueError('share terms unavailable')
    if (not isinstance(source_id, str) or not source_id.strip()
            or not isinstance(source_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', source_sha256)):
        raise ValueError('share terms source identity/hash required')
    if not isinstance(ts_code, str) or not ts_code.strip():
        raise ValueError('share identity required')
    if distribution.quote_unit != 'ordinary_share':
        raise ValueError('CDR requires separate unit adapter')
    positive(distribution.bonus_per_share, 'share distribution rate')
    if any(c.distribution.event_id == distribution.event_id for c in book.share_claims):
        raise ValueError('duplicate share claim')
    record_lots=record_date_lots(book,ts_code,instant,distribution)
    quantity = sum(p.remaining_quantity for p in record_lots)
    contractual = quantity * distribution.bonus_per_share
    positive(contractual, 'contractual shares', zero=True)
    claim = ShareClaim(ts_code, distribution, quantity, contractual,
                       source_id, source_sha256, available.isoformat(),
                       record_lots=tuple((p.position_id,p.remaining_quantity) for p in record_lots))
    return append(book, event_id, at, dict(kind='share_record', ts_code=ts_code,
        distribution_id=distribution.event_id, record_quantity=quantity,
        contractual_quantity=contractual, source_id=source_id, source_sha256=source_sha256,
        terms_available_at=available.isoformat(), cash_delta=0.,
        source_bytes_verified=False, received_quantity=None, tradable_quantity=None),
        share_claims=book.share_claims+(claim,))


def receive_share_claim(book, *, event_id, distribution_id, credited_quantity,
                        credited_at, received_at, processing_at, evidence_id,
                        evidence_sha256):
    """Retain a supplied whole-share credit, not authorize trading or infer basis.

    Partial credits are supported. Any fractional residual stays unresolved;
    no rounding-up entitlement or cash-in-lieu policy is manufactured.
    """
    at = clock(book, event_id, processing_at)
    credited, received = _instant(credited_at), _instant(received_at)
    if not credited <= received <= _instant(at):
        raise ValueError('invalid share receipt chronology')
    if (not isinstance(evidence_id, str) or not evidence_id.strip()
            or not isinstance(evidence_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', evidence_sha256)):
        raise ValueError('share receipt evidence identity/hash required')
    if type(credited_quantity) is not int or credited_quantity <= 0:
        raise ValueError('positive whole credited shares required')
    found = [c for c in book.share_claims if c.distribution.event_id == distribution_id]
    if len(found) != 1:
        raise ValueError('recorded share claim required')
    claim = found[0]
    if claim.fractional_settlement:
        raise ValueError('share entitlement already finally settled')
    if credited < _instant(claim.terms_available_at) or (
            credited.tz_convert('Asia/Shanghai').strftime('%Y%m%d') < claim.distribution.ex_date):
        raise ValueError('share credit predates entitlement')
    if any(r.evidence_id == evidence_id for c in book.share_claims for r in c.receipts):
        raise ValueError('duplicate share receipt evidence')
    total = sum(r.quantity for r in claim.receipts) + credited_quantity
    contractual = Decimal(claim.record_quantity) * Decimal(str(claim.distribution.bonus_per_share))
    if Decimal(total) > contractual:
        raise ValueError('share credits exceed contractual entitlement; settlement adapter required')
    receipt = ShareReceipt(credited_quantity, credited.isoformat(), received.isoformat(),
                           evidence_id, evidence_sha256)
    updated = replace(claim, receipts=claim.receipts+(receipt,))
    return append(book, event_id, at, dict(kind='share_receipt', distribution_id=distribution_id,
        ts_code=claim.ts_code, credited_quantity=credited_quantity,
        total_credited_quantity=total, contractual_residual=str(contractual-Decimal(total)),
        credited_at=credited.isoformat(), received_at=received.isoformat(),
        evidence_id=evidence_id, evidence_sha256=evidence_sha256, source_bytes_verified=False,
        cost_basis_allocated=False, tradable_quantity=None, cash_delta=0.),
        share_claims=tuple(updated if c is claim else c for c in book.share_claims))


def settle_fractional_share(book, *, event_id, distribution_id, quantity, net_cash,
                            settled_at, received_at, processing_at, evidence_id, evidence_sha256):
    """Consume an explicitly evidenced fractional residual for supplied net cash.

    No inferred cash-in-lieu rate, tax basis, withholding or market rule.
    Whole-share cash-outs and subsequent receipt corrections are not this event.
    """
    at=clock(book,event_id,processing_at)
    settled,received=_instant(settled_at),_instant(received_at)
    if not settled<=received<=_instant(at): raise ValueError('fractional settlement chronology invalid')
    if not isinstance(quantity,str): raise ValueError('fraction quantity requires decimal string')
    try: fraction=Decimal(quantity)
    except Exception as exc: raise ValueError('invalid fractional quantity') from exc
    if not fraction.is_finite() or not 0<fraction<1: raise ValueError('fraction must be between zero and one')
    positive(net_cash,'net fractional cash',zero=True)
    if (not isinstance(evidence_id,str) or not evidence_id.strip()
            or not isinstance(evidence_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',evidence_sha256)):
        raise ValueError('fractional settlement evidence required')
    if any(e.get('evidence_id')==evidence_id for e in book.journal):
        raise ValueError('duplicate settlement evidence')
    found=[c for c in book.share_claims if c.distribution.event_id==distribution_id]
    if len(found)!=1 or found[0].fractional_settlement: raise ValueError('unsettled share claim required')
    claim=found[0]
    if settled.tz_convert('Asia/Shanghai').strftime('%Y%m%d')<claim.distribution.ex_date:
        raise ValueError('fractional settlement predates entitlement')
    if claim.receipts and settled<max(_instant(r.credited_at) for r in claim.receipts):
        raise ValueError('fractional settlement predates whole-share credits')
    if share_contract_residual(claim)!=fraction:
        raise ValueError('fractional settlement must exhaust exact residual')
    updated=replace(claim,fractional_settlement=FractionalSettlement(str(fraction),float(net_cash),
        settled.isoformat(),received.isoformat(),evidence_id,evidence_sha256))
    return append(book,event_id,at,dict(kind='fractional_share_settlement',distribution_id=distribution_id,
        fractional_quantity=str(fraction),cash_delta=net_cash,net_cash=net_cash,
        settled_at=settled.isoformat(),received_at=received.isoformat(),evidence_id=evidence_id,
        evidence_sha256=evidence_sha256,final_tax_verified=False,source_bytes_verified=False),
        share_claims=tuple(updated if c is claim else c for c in book.share_claims),
        cash=book.cash+net_cash,share_settlement_net_income=book.share_settlement_net_income+net_cash)


def record_mixed_distribution(book, *, event_id, ts_code, distribution, processing_at,
                              terms_available_at, source_id, source_sha256):
    """Atomically return both claims from one unchanged record snapshot.

    Two journal entries form one replay packet. Immutable intermediate state
    is not exposed if either leg fails. Both claims retain the full terms;
    this does not settle either leg or establish investor tax/sale rules.
    """
    at=clock(book,event_id,processing_at)
    positive(distribution.cash_per_share,'mixed cash rate')
    positive(distribution.bonus_per_share,'mixed share rate')
    if any(c.distribution.event_id==distribution.event_id for c in book.cash_claims):
        raise ValueError('duplicate mixed cash claim')
    proposed=record_share_claim(book,event_id=event_id+'::share',ts_code=ts_code,
        distribution=distribution,processing_at=at,terms_available_at=terms_available_at,
        source_id=source_id,source_sha256=source_sha256)
    share=proposed.share_claims[-1]
    amount=share.record_quantity*distribution.cash_per_share
    positive(amount,'mixed cash entitlement',zero=True)
    cash=CashClaim(ts_code,distribution,share.record_quantity,amount,source_id)
    return append(proposed,event_id,at,dict(kind='cash_record',distribution_id=distribution.event_id,
        ts_code=ts_code,record_quantity=share.record_quantity,gross_amount=amount,
        source_id=source_id,source_sha256=source_sha256,terms_available_at=share.terms_available_at,
        cash_delta=0.,mixed_packet_id=event_id,paired_share_event_id=event_id+'::share'),
        cash_claims=proposed.cash_claims+(cash,))


def allocate_share_basis(book, *, event_id, distribution_id, allocations,
                         available_at, processing_at, evidence_id, evidence_sha256):
    """Apply supplied currency basis transfers, including post-record sold units.

    This is a reference accounting allocation, not inferred tax basis. Prior
    journal valuations/costs remain untouched; adjustments are booked now.
    """
    from .exit_book import BasisTransfer, allocated_exit_cost
    from .portfolio_book import accounting_snapshot
    at=clock(book,event_id,processing_at)
    if _instant(available_at)>_instant(at): raise ValueError('basis evidence unavailable')
    if (not isinstance(evidence_id,str) or not evidence_id.strip()
            or not isinstance(evidence_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',evidence_sha256)):
        raise ValueError('basis evidence required')
    if any(e.get('evidence_id')==evidence_id for e in book.journal): raise ValueError('duplicate basis evidence')
    found=[c for c in book.share_claims if c.distribution.event_id==distribution_id]
    if len(found)!=1 or found[0].allocated_basis is not None: raise ValueError('unallocated share claim required')
    claim=found[0]
    if not claim.receipts or share_contract_residual(claim)!=0:
        raise ValueError('fully resolved credited share quantity required before basis allocation')
    if not isinstance(allocations,dict) or set(allocations)!=set(k for k,q in claim.record_lots):
        raise ValueError('exact recorded source lots required')
    if any(o.lot.position_id in allocations for o in book.exits): raise ValueError('source exit result pending')
    positions=list(book.positions);adjustments=[];total=0.
    for position_id,record_qty in claim.record_lots:
        matches=[i for i,p in enumerate(positions) if p.position_id==position_id and p.ts_code==claim.ts_code]
        if len(matches)!=1: raise ValueError('recorded source lot missing')
        i=matches[0];old=positions[i];amount=allocations[position_id]
        positive(amount,'allocated share basis',zero=True)
        original=old.entry_cost+sum(t.amount for t in old.basis_transfers)
        available=(original*record_qty/old.initial_quantity
            -sum(t.amount*min(record_qty,t.record_quantity)/t.record_quantity for t in old.basis_transfers))
        if amount>available or amount>old.entry_cost: raise ValueError('basis allocation exceeds eligible source cost')
        new=replace(old,entry_cost=old.entry_cost-amount,
                    basis_transfers=old.basis_transfers+(BasisTransfer(record_qty,float(amount)),))
        cost=allocated_exit_cost(new,new.remaining_quantity)
        new=replace(new,realized_cost=cost,realized_pnl=new.net_exit_cash-cost)
        positions[i]=new;total+=amount
        adjustments.append(dict(position_id=position_id,transferred_basis=amount,
            realized_pnl_adjustment=new.realized_pnl-old.realized_pnl,
            previous_realized_cost=old.realized_cost,new_realized_cost=cost))
    updated=replace(claim,allocated_basis=total)
    result=append(book,event_id,at,dict(kind='share_basis_allocation',distribution_id=distribution_id,
        allocations=adjustments,allocated_basis=total,cash_delta=0.,evidence_id=evidence_id,
        evidence_sha256=evidence_sha256,available_at=_instant(available_at).isoformat(),
        tax_basis_verified=False,source_bytes_verified=False),positions=tuple(positions),
        share_claims=tuple(updated if c is claim else c for c in book.share_claims))
    accounting_snapshot(result)
    return result


def transfer_share_position(book, *, event_id, distribution_id, position_id, due_date,
                            sellable_from, eligibility_available_at, processing_at,
                            evidence_id, evidence_sha256):
    """Move resolved credits and their basis once into the existing exit engine.

    Explicit eligibility is a supplied assertion requiring external validation,
    not inferred from listing day or a new buy's T+1 rule. No orders are sent.
    """
    from .exit_book import ExitLot
    from .portfolio_book import accounting_snapshot
    at=clock(book,event_id,processing_at)
    if _instant(eligibility_available_at)>_instant(at): raise ValueError('sellability evidence unavailable')
    if (not isinstance(evidence_id,str) or not evidence_id.strip()
            or not isinstance(evidence_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',evidence_sha256)):
        raise ValueError('sellability evidence required')
    if any(e.get('evidence_id')==evidence_id for e in book.journal): raise ValueError('duplicate eligibility evidence')
    if not isinstance(position_id,str) or not position_id.strip() or any(p.position_id==position_id for p in book.positions):
        raise ValueError('unique transferred position ID required')
    found=[c for c in book.share_claims if c.distribution.event_id==distribution_id]
    if len(found)!=1: raise ValueError('share claim required')
    c=found[0]
    if c.transferred_position_id is not None or c.allocated_basis is None or not c.receipts or share_contract_residual(c)!=0:
        raise ValueError('resolved allocated untransferred credit required')
    acquisition=max(_instant(r.credited_at) for r in c.receipts).tz_convert('Asia/Shanghai').strftime('%Y%m%d')
    day=_instant(at).tz_convert('Asia/Shanghai').strftime('%Y%m%d')
    if (any(d not in book.calendar for d in [acquisition,sellable_from,due_date])
            or sellable_from<max(acquisition,c.distribution.bonus_list_date)
            or due_date<max(sellable_from,day)):
        raise ValueError('invalid credited-share eligibility/exit dates')
    q=sum(r.quantity for r in c.receipts)
    lot=ExitLot(position_id,c.ts_code,acquisition,due_date,q,q,c.allocated_basis,
                explicit_sellable_from=sellable_from)
    updated=replace(c,transferred_position_id=position_id)
    result=append(book,event_id,at,dict(kind='share_position_transfer',distribution_id=distribution_id,
        position_id=position_id,quantity=q,allocated_basis=c.allocated_basis,cash_delta=0.,
        sellable_from=sellable_from,eligibility_available_at=_instant(eligibility_available_at).isoformat(),
        evidence_id=evidence_id,evidence_sha256=evidence_sha256,eligibility_independently_verified=False),
        positions=book.positions+(lot,),share_claims=tuple(updated if x is c else x for x in book.share_claims))
    accounting_snapshot(result)
    return result
