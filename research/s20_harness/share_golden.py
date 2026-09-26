"""Deterministic mixed-distribution portfolio scenario, not real market proof."""
from .portfolio_replay import step
from .portfolio_book import accounting_snapshot


def lifecycle(book, *, settle_fraction):
    def run(kind,**args):
        nonlocal book
        book=step(book,dict(kind=kind,args=args))
        accounting_snapshot(book)
    run('mixed_distribution_record',event_id='mixed',ts_code='stock',
        distribution=dict(event_id='dist',record_date='20240702',ex_date='20240729',known_date='20240701',
            cash_per_share=.2,bonus_per_share=.105,pay_date='20240730',bonus_list_date='20240730',
            beneficiary_scope='existing_shareholders_verified'),
        processing_at='2024-07-02T15:02:00+08:00',terms_available_at='2024-07-01T12:00:00+08:00',
        source_id='synthetic',source_sha256='a'*64)
    run('cash_accrual',event_id='accrue',distribution_id='dist',processing_at='2024-07-29T09:00:00+08:00')
    def sell(position,day,quantity):
        run('exit_request',order_id='sell-'+position,position_id=position,
            policy=dict(policy_id='synthetic',frozen_at='2024-07-01T10:00:00+08:00'),
            submitted_at=day+'T14:50:00+08:00')
        run('exit_result',event_id='sold-'+position,order_id='sell-'+position,
            outcome_at=day+'T15:00:00+08:00',received_at=day+'T15:01:00+08:00',
            processing_at=day+'T15:02:00+08:00',filled_quantity=quantity,price=6.,fee=1.,
            reason='synthetic',evidence_id='sale-'+position)
    sell('first','2024-07-29',100)
    run('share_receipt',event_id='credit',distribution_id='dist',credited_quantity=10,
        credited_at='2024-07-30T08:00:00+08:00',received_at='2024-07-30T08:01:00+08:00',
        processing_at='2024-07-30T08:02:00+08:00',evidence_id='credit',evidence_sha256='b'*64)
    run('cash_receipt',event_id='cash',distribution_id='dist',cash_received=18.,tax_withheld=2.,
        received_at='2024-07-30T09:00:00+08:00',processing_at='2024-07-30T09:01:00+08:00',evidence_id='cash')
    if settle_fraction:
        run('fractional_share_settlement',event_id='fraction',distribution_id='dist',quantity='0.5',net_cash=2.,
            settled_at='2024-07-30T10:00:00+08:00',received_at='2024-07-30T10:01:00+08:00',
            processing_at='2024-07-30T10:02:00+08:00',evidence_id='fraction',evidence_sha256='c'*64)
        run('share_basis_allocation',event_id='basis',distribution_id='dist',allocations={'first':70.1},
            available_at='2024-07-30T10:30:00+08:00',processing_at='2024-07-30T11:00:00+08:00',
            evidence_id='basis',evidence_sha256='d'*64)
        run('share_position_transfer',event_id='transfer',distribution_id='dist',position_id='bonus',due_date='20240730',
            sellable_from='20240730',eligibility_available_at='2024-07-30T11:30:00+08:00',
            processing_at='2024-07-30T12:00:00+08:00',evidence_id='eligibility',evidence_sha256='e'*64)
        sell('bonus','2024-07-30',10)
    from .portfolio_valuation import value_portfolio,ShareCreditMark
    marks=[] if settle_fraction else [ShareCreditMark('dist','2024-07-30T15:02:00+08:00',
        '2024-07-30T15:03:00+08:00','observed',6.,'synthetic','f'*64)]
    nav=value_portfolio(book,[],share_marks=marks,valuation_at='2024-07-30T15:02:00+08:00',
        knowledge_at='2024-07-30T15:03:00+08:00')
    return dict(cash=book.cash,nav=nav['nav'],known_value=nav['known_component_value'],
        held_units=sum(p.remaining_quantity for p in book.positions),
        pending_rights=nav['unresolved_share_distribution_ids'],
        conservation_error=round(accounting_snapshot(book)['conservation_error'],8))
