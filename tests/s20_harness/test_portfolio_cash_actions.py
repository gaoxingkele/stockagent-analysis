import pytest

from research.s20_harness.corporate_actions import Distribution
from research.s20_harness.portfolio_book import create, accounting_snapshot, reserve_exit, settle_portfolio_exit
from research.s20_harness.portfolio_cash_actions import record_cash_claim, accrue_cash_claim, receive_cash_claim
from tests.s20_harness.test_portfolio_book import buy, fill, CAL, POLICY


def event():
    return Distribution('cash1', '20240702', '20240729', '20240701', cash_per_share=.2,
                        pay_date='20240730', beneficiary_scope='existing_shareholders_verified')


def record(book, **kwargs):
    args = dict(event_id='record', ts_code='stock', distribution=event(),
                processing_at='2024-07-02T15:02:00+08:00',
                terms_available_at='2024-07-01T12:00:00+08:00', source_id='synthetic')
    args.update(kwargs)
    return record_cash_claim(book, **args)


def accrue(book):
    return accrue_cash_claim(book, event_id='accrue', distribution_id='cash1', processing_at='2024-07-29T09:00:00+08:00')


def pay(book, **kwargs):
    args=dict(event_id='paid', distribution_id='cash1', received_at='2024-07-30T09:00:00+08:00',
              processing_at='2024-07-30T09:01:00+08:00', cash_received=18., tax_withheld=2., evidence_id='synthetic')
    args.update(kwargs)
    return receive_cash_claim(book, **args)


def adjust(book, **kwargs):
    from research.s20_harness.portfolio_cash_actions import settle_dividend_tax
    args=dict(event_id='tax',distribution_id='cash1',tax_delta=2.,
        settled_at='2024-07-31T09:00:00+08:00',received_at='2024-07-31T09:01:00+08:00',
        processing_at='2024-07-31T09:02:00+08:00',evidence_id='tax-receipt')
    args.update(kwargs)
    return settle_dividend_tax(book,**args)


def test_explicit_tax_debit_refund_preserve_conservation():
    original=pay(accrue(record(fill(buy(create(1000,CAL))))))
    debit=adjust(original)
    assert original.cash==317 and debit.cash==315
    assert debit.distribution_tax_paid==4
    assert accounting_snapshot(debit)['conservation_error']==0
    refund=adjust(debit,event_id='refund',evidence_id='refund-receipt',tax_delta=-3.)
    assert refund.cash==318 and refund.distribution_tax_paid==1
    assert accounting_snapshot(refund)['conservation_error']==0
    assert not refund.journal[-1]['final_personal_tax_proven']
    with pytest.raises(ValueError,match='duplicate tax receipt'):
        adjust(debit,event_id='different-id')


@pytest.mark.parametrize('kwargs',[
    dict(tax_delta=-3),dict(tax_delta=19),dict(tax_delta=0),dict(tax_delta=True),dict(tax_delta=float('nan')),
    dict(evidence_id=''),dict(received_at='2024-07-31T08:00:00+08:00'),
    dict(settled_at='2024-07-29T09:00:00+08:00'),dict(received_at='2024-08-01T09:00:00+08:00')])
def test_invalid_tax_receipts_rejected(kwargs):
    b=pay(accrue(record(fill(buy(create(1000,CAL))))))
    with pytest.raises(ValueError): adjust(b,**kwargs)
    assert b.cash==317 and b.distribution_tax_paid==2


def test_tax_requires_paid_claim():
    with pytest.raises(ValueError,match='paid dividend claim'):
        adjust(accrue(record(fill(buy(create(1000,CAL))))))


def test_tax_does_not_spend_reserved_order_cash():
    from research.s20_harness.portfolio_book import reserve_buy
    b=pay(accrue(record(fill(buy(create(1000,CAL))))))
    b=reserve_buy(b,order_id='reserve-other',ts_code='other',buy_date='20240731',due_date='20240731',
        quantity=100,price_cap=3.15,fee_cap=0.,submitted_at='2024-07-31T08:50:00+08:00')
    assert b.cash==2
    with pytest.raises(ValueError,match='remaining cash'): adjust(b,tax_delta=3.)
    assert b.cash==2 and accounting_snapshot(b)['reserved_buy_cash']==315


def test_record_accrual_payment_preserve_cash_and_accounting():
    b = record(fill(buy(create(1000,CAL))))
    assert b.cash==299 and accounting_snapshot(b)['gross_dividend_receivable']==0
    b=accrue(b)
    assert b.cash==299 and accounting_snapshot(b)['gross_dividend_receivable']==20
    b=pay(b)
    s=accounting_snapshot(b)
    assert b.cash==317 and s['gross_dividend_receivable']==0
    assert s['distribution_income']==20 and s['distribution_tax_paid']==2 and s['conservation_error']==0


def test_selling_after_record_does_not_erase_receivable():
    b=accrue(record(fill(buy(create(1000,CAL)))))
    b=reserve_exit(b,order_id='sell',position_id='buy1',policy=POLICY,submitted_at='2024-07-29T14:50:00+08:00')
    b=settle_portfolio_exit(b,event_id='sell-result',order_id='sell',outcome_at='2024-07-29T15:00:00+08:00',
        received_at='2024-07-29T15:01:00+08:00',processing_at='2024-07-29T15:02:00+08:00',
        filled_quantity=100,price=6,fee=1,reason='synthetic',evidence_id='synthetic')
    assert accounting_snapshot(b)['gross_dividend_receivable']==20
    b=pay(b)
    assert b.cash==916 and accounting_snapshot(b)['conservation_error']==0
    b=adjust(b)
    assert b.cash==914 and accounting_snapshot(b)['conservation_error']==0


def test_unreported_record_day_execution_blocks_entitlement():
    with pytest.raises(ValueError,match='unsettled'):
        record(buy(create(1000,CAL)))


def test_duplicate_record_accrual_and_payment_rejected():
    b=record(fill(buy(create(1000,CAL))))
    with pytest.raises(ValueError,match='duplicate distribution'):record(b,event_id='again')
    b=accrue(b)
    with pytest.raises(ValueError):accrue_cash_claim(b,event_id='again',distribution_id='cash1',processing_at='2024-07-29T09:01:00+08:00')
    b=pay(b)
    with pytest.raises(ValueError):pay(b,event_id='again')


@pytest.mark.parametrize('kwargs',[dict(cash_received=20,tax_withheld=2),dict(cash_received=-1),
    dict(evidence_id=''),dict(received_at='2024-07-29T09:00:00+08:00'),
    dict(received_at='2024-07-31T09:00:00+08:00')])
def test_invalid_payments_rejected(kwargs):
    with pytest.raises(ValueError):pay(accrue(record(fill(buy(create(1000,CAL))))),**kwargs)


def test_early_accrual_and_unavailable_terms_rejected():
    b=fill(buy(create(1000,CAL)))
    with pytest.raises(ValueError):record(b,terms_available_at='2024-07-03T12:00:00+08:00')
    b=record(b)
    with pytest.raises(ValueError):accrue_cash_claim(b,event_id='early',distribution_id='cash1',processing_at='2024-07-02T15:03:00+08:00')


def test_ex_date_valuation_requires_accrual_and_keeps_gross_value_continuity():
    from research.s20_harness.portfolio_valuation import PriceMark,value_portfolio
    b=record(fill(buy(create(1000,CAL))))
    mark=PriceMark('stock','2024-07-29T15:00:00+08:00','2024-07-29T15:01:00+08:00',
                   'observed',6.8,'synthetic','a'*64)
    def value(book):return value_portfolio(book,[mark],valuation_at=mark.price_at,knowledge_at=mark.available_at)
    assert value(b)['nav'] is None and value(b)['unprocessed_cash_distribution_ids']==['cash1']
    r=value(accrue(b))
    assert r['nav']==999 and r['gross_dividend_receivable']==20
    assert not r['final_personal_dividend_tax_verified']
