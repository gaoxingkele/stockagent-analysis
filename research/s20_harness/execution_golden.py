"""Persist deterministic H02 execution cases; never evidence of real fills."""
from dataclasses import asdict
from pathlib import Path
import uuid

from . import execution, exit_book, portfolio_book
from .runtime import atomic_json,digest,now,load_plan


def verify(directory,summary_sha):
    """Replay pinned current-code cases; green summary text alone is insufficient."""
    directory=Path(directory).resolve()
    if digest(directory/'summary.json')!=summary_sha: raise ValueError('golden summary pin mismatch')
    summary=load_plan(directory/'summary.json')
    if set(summary['artifacts'])!={'golden_cases.json','inputs.json'}:
        raise ValueError('unexpected golden artifacts')
    pins=load_plan(directory/'inputs.json')['code_pins']
    required={str(p.resolve()) for p in Path(__file__).parent.glob('*.py')}
    if set(pins)!=required:
        raise ValueError('missing golden code provenance')
    def check():
        if digest(directory/'summary.json')!=summary_sha or any(
            digest(directory/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('golden artifacts changed')
        if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('golden code changed')
    check()
    saved=load_plan(directory/'golden_cases.json')
    if saved.get('version') not in ('3','4','5','6') or saved.get('evidence_mode')!='synthetic':
        raise ValueError('unsupported golden contract')
    rebuilt=evaluate(saved['version'])
    if saved['cases']!=rebuilt: raise ValueError('golden replay differs from saved cases')
    passed=sum(c['passed'] for c in rebuilt)
    if (summary['cases']!=len(rebuilt) or summary['passed']!=passed
        or summary['all_passed']!=all(c['passed'] for c in rebuilt)
        or summary['real_fill_evidence'] is not False or summary['full_H02_coverage'] is not False
        or summary['formal_training_authorized'] is not False):
        raise ValueError('golden summary semantics mismatch')
    check()
    return dict(replayed=True,cases=len(rebuilt),passed=passed,all_passed=passed==len(rebuilt),
        real_fill_evidence=False,full_H02_coverage=False,formal_training_authorized=False)


def evaluate(version='6'):
    if version not in ('3','4','5','6'): raise ValueError('unsupported golden version')
    cases=[]
    def record(name,expected,fn):
        try:
            actual=fn();error=None
        except Exception as exc:
            actual=None;error=dict(type=type(exc).__name__,message=str(exc))
        cases.append(dict(case_id=name,expected=expected,actual=actual,error=error,
                          passed=error is None and actual==expected))
    def blocked_sale():
        cal=['20240102','20240103']
        lot=exit_book.open_lot('p','fixture.SH',cal[0],cal[0],100,1000.,cal)
        try:
            exit_book.apply_exit(lot,attempt_id='a',position_id='p',ts_code='fixture.SH',
                trade_date=cal[0],calendar=cal,filled_quantity=100,price=10.,fee=0.,
                reason='synthetic',evidence_id='fixture')
        except ValueError as exc:
            return dict(rejected=str(exc)=='T+1 sale forbidden',remaining=lot.remaining_quantity)
        return dict(rejected=False,remaining=lot.remaining_quantity)
    record('same_day_sale',dict(rejected=True,remaining=100),blocked_sale)
    for name,kwargs,reason in [('halt',dict(suspended=True),'suspended'),
                              ('upper_limit',dict(limit_up=10.),'limit_up_open')]:
        record(name,dict(filled=False,price=None,reason=reason,used_pct_chg=False),
               lambda kwargs=kwargs:asdict(execution.decide_open_buy(10.,**kwargs)))
    record('close_return_not_used',True,
        lambda:execution.decide_open_buy(10.,limit_up=11.,pct_chg=10.)==
               execution.decide_open_buy(10.,limit_up=11.,pct_chg=-10.))
    record('same_day_touch_bounds',dict(ambiguous=True,keep=True,optimistic='up_first',pessimistic='down_first'),
        lambda:(lambda b:dict(ambiguous=b['ambiguous'],keep=b['keep_ex_ante'],
                    optimistic=b['bound_optimistic'],pessimistic=b['bound_pessimistic']))(
                        execution.ohlc_path_bounds(100.,120.,89.)))
    def lifecycle():
        cal=['20240702','20240729','20240730','20240802']
        state=exit_book.open_lot('p','fixture.SH',cal[0],cal[1],200,1402.,cal)
        records=[]
        for i,(date,quantity,price,fee) in enumerate([(cal[1],0,None,0.),(cal[2],80,6.,5.),(cal[3],120,5.,5.)]):
            state,event=exit_book.apply_exit(state,attempt_id=str(i),position_id='p',ts_code='fixture.SH',
                trade_date=date,calendar=cal,filled_quantity=quantity,price=price,fee=fee,
                reason='synthetic',evidence_id='fixture')
            records.append(dict(remaining=state.remaining_quantity,cash=state.net_exit_cash,
                status=state.status,unknown_mark=event['remaining_market_value'] is None))
        return dict(events=records,fees=state.exit_fees,cost=state.realized_cost,pnl=state.realized_pnl)
    record('halt_partial_delayed_exit',dict(events=[
        dict(remaining=200,cash=0.,status='EXIT_PENDING',unknown_mark=True),
        dict(remaining=120,cash=475.,status='EXIT_PENDING',unknown_mark=True),
        dict(remaining=0,cash=1070.,status='CLOSED',unknown_mark=False)],fees=10.,cost=1402.,pnl=-332.),lifecycle)
    def portfolio(partial,return_book=False):
        cal=['20240702','20240729','20240730']
        book=portfolio_book.create(1000.,cal)
        def reserve(book,order,code):
            return portfolio_book.reserve_buy(book,order_id=order,ts_code=code,
                buy_date=cal[0],due_date=cal[1],quantity=100,price_cap=8.,fee_cap=1.,
                submitted_at='2024-07-02T09:20:00+08:00')
        book=reserve(book,'first','stock')
        reserved_cash=book.cash
        book=reserve(book,'other','other-stock'); insufficient=book.journal[-1]['reason']
        book=reserve(book,'duplicate','stock'); duplicate=book.journal[-1]['reason']
        book=portfolio_book.settle_buy(book,event_id='fill',order_id='first',
            outcome_at='2024-07-02T09:30:00+08:00',received_at='2024-07-02T09:31:00+08:00',
            processing_at='2024-07-02T09:32:00+08:00',filled_quantity=partial,
            price=7. if partial else None,fee=1. if partial else 0.,reason='synthetic',evidence_id='fixture')
        snap=portfolio_book.accounting_snapshot(book)
        if return_book: return book
        return dict(reserved_cash=reserved_cash,insufficient=insufficient,duplicate=duplicate,
            cash=book.cash,quantity=sum(p.remaining_quantity for p in book.positions),
            cost=snap['remaining_cost_basis'],conservation_error=snap['conservation_error'],
            open_buy_orders=len(book.buys))
    for quantity,cash,cost in [(0,1000.,0.),(40,719.,281.),(100,299.,701.)]:
        record('portfolio_buy_'+str(quantity),dict(reserved_cash=199.,insufficient='insufficient_cash',
            duplicate='buy_pending',cash=cash,quantity=quantity,cost=cost,conservation_error=0.,open_buy_orders=0),
            lambda quantity=quantity:portfolio(quantity))
    def dividend(sell_before_payment=False,return_book=False):
        from .corporate_actions import Distribution
        from .portfolio_cash_actions import record_cash_claim,accrue_cash_claim,receive_cash_claim
        book=portfolio(100,return_book=True)
        distribution=Distribution('cash1','20240702','20240729','20240701',cash_per_share=.2,
            pay_date='20240730',beneficiary_scope='existing_shareholders_verified')
        book=record_cash_claim(book,event_id='record',ts_code='stock',distribution=distribution,
            processing_at='2024-07-02T15:02:00+08:00',terms_available_at='2024-07-01T12:00:00+08:00',source_id='synthetic')
        stages=[]
        def snapshot():
            s=portfolio_book.accounting_snapshot(book)
            return dict(cash=book.cash,receivable=s['gross_dividend_receivable'],error=s['conservation_error'])
        stages.append(snapshot())
        book=accrue_cash_claim(book,event_id='accrue',distribution_id='cash1',processing_at='2024-07-29T09:00:00+08:00')
        stages.append(snapshot())
        if sell_before_payment:
            from .exit_policy import ExitPolicy
            policy=ExitPolicy('synthetic','2024-07-01T10:00:00+08:00')
            book=portfolio_book.reserve_exit(book,order_id='dividend-sell',position_id='first',policy=policy,
                submitted_at='2024-07-29T14:50:00+08:00')
            book=portfolio_book.settle_portfolio_exit(book,event_id='dividend-sold',order_id='dividend-sell',
                outcome_at='2024-07-29T15:00:00+08:00',received_at='2024-07-29T15:01:00+08:00',
                processing_at='2024-07-29T15:02:00+08:00',filled_quantity=100,price=6.,fee=1.,
                reason='synthetic',evidence_id='synthetic')
            stages.append(snapshot())
        book=receive_cash_claim(book,event_id='paid',distribution_id='cash1',received_at='2024-07-30T09:00:00+08:00',
            processing_at='2024-07-30T09:01:00+08:00',cash_received=18.,tax_withheld=2.,evidence_id='synthetic')
        stages.append(snapshot())
        s=portfolio_book.accounting_snapshot(book)
        if return_book: return book
        return dict(stages=stages,income=s['distribution_income'],tax=s['distribution_tax_paid'])
    record('dividend_record_accrual_payment',dict(stages=[dict(cash=299.,receivable=0.,error=0.),
        dict(cash=299.,receivable=20.,error=0.),dict(cash=317.,receivable=0.,error=0.)],income=20.,tax=2.),dividend)
    if version in ('4','5','6'):
        record('sale_preserves_dividend_claim',dict(stages=[dict(cash=299.,receivable=0.,error=0.),
            dict(cash=299.,receivable=20.,error=0.),dict(cash=898.,receivable=20.,error=0.),
            dict(cash=916.,receivable=0.,error=0.)],income=20.,tax=2.),
            lambda:dividend(sell_before_payment=True))
    if version in ('5','6'):
        def tax_path(backdate=False):
            from .portfolio_cash_actions import settle_dividend_tax
            from .portfolio_valuation import value_portfolio
            original=dividend(sell_before_payment=True,return_book=True)
            before=value_portfolio(original,[],valuation_at='2024-07-30T15:00:00+08:00',
                knowledge_at='2024-07-30T15:01:00+08:00')['nav']
            book=settle_dividend_tax(original,event_id='tax-debit',distribution_id='cash1',tax_delta=2.,
                settled_at='2024-07-30T16:00:00+08:00',received_at='2024-07-31T09:00:00+08:00',
                processing_at='2024-07-31T09:01:00+08:00',evidence_id='synthetic-debit')
            if backdate:
                try:
                    value_portfolio(book,[],valuation_at='2024-07-30T15:00:00+08:00',
                        knowledge_at='2024-07-31T09:02:00+08:00')
                except ValueError as exc:
                    return dict(rejected=str(exc)=='book snapshot contains later processing',old_cash=original.cash)
                return dict(rejected=False,old_cash=original.cash)
            after=book.cash
            book=settle_dividend_tax(book,event_id='tax-refund',distribution_id='cash1',tax_delta=-1.,
                settled_at='2024-07-31T10:00:00+08:00',received_at='2024-07-31T10:01:00+08:00',
                processing_at='2024-07-31T10:02:00+08:00',evidence_id='synthetic-refund')
            snap=portfolio_book.accounting_snapshot(book)
            return dict(before=before,after_debit=after,after_refund=book.cash,
                total_tax=snap['distribution_tax_paid'],error=snap['conservation_error'])
        record('explicit_tax_debit_refund',dict(before=916.,after_debit=914.,after_refund=915.,total_tax=3.,error=0.),tax_path)
        record('late_tax_cannot_backdate_nav',dict(rejected=True,old_cash=916.),lambda:tax_path(backdate=True))
    def pending_exit():
        from .exit_policy import ExitPolicy
        book=portfolio(100,return_book=True)
        policy=ExitPolicy('synthetic','2024-07-01T10:00:00+08:00')
        book=portfolio_book.reserve_exit(book,order_id='exit1',position_id='first',policy=policy,
            submitted_at='2024-07-29T14:50:00+08:00')
        book=portfolio_book.settle_portfolio_exit(book,event_id='halted',order_id='exit1',
            outcome_at='2024-07-29T15:00:00+08:00',received_at='2024-07-29T15:01:00+08:00',
            processing_at='2024-07-29T15:02:00+08:00',filled_quantity=0,reason='suspended',evidence_id='fixture')
        s=portfolio_book.accounting_snapshot(book)
        pending=dict(cash=s['cash'],cost=s['remaining_cost_basis'],pending=s['pending_exits'],nav=s['nav'])
        book=portfolio_book.reserve_buy(book,order_id='again',ts_code='stock',buy_date='20240730',due_date='20240730',
            quantity=100,price_cap=8.,fee_cap=1.,submitted_at='2024-07-30T09:10:00+08:00')
        reason=book.journal[-1]['reason']
        book=portfolio_book.reserve_exit(book,order_id='retry',position_id='first',policy=policy,
            submitted_at='2024-07-30T09:20:00+08:00')
        book=portfolio_book.settle_portfolio_exit(book,event_id='sold',order_id='retry',
            outcome_at='2024-07-30T09:30:00+08:00',received_at='2024-07-30T09:31:00+08:00',
            processing_at='2024-07-30T09:32:00+08:00',filled_quantity=100,price=6.,fee=1.,
            reason='synthetic',evidence_id='fixture')
        s=portfolio_book.accounting_snapshot(book)
        return dict(pending=pending,repeat_reason=reason,cash=s['cash'],pnl=s['realized_pnl'],
            nav=s['nav'],held=s['held_positions'],error=s['conservation_error'])
    record('portfolio_pending_exit',dict(pending=dict(cash=299.,cost=701.,pending=1,nav=None),
        repeat_reason='already_held',cash=898.,pnl=-102.,nav=898.,held=0,error=0.),pending_exit)
    if version=='6':
        from .share_golden import lifecycle as share_lifecycle
        record('mixed_share_fraction_basis_and_exit',dict(cash=977.,nav=977.,known_value=977.,
            held_units=0,pending_rights=[],conservation_error=0.),
            lambda:share_lifecycle(portfolio(100,return_book=True),settle_fraction=True))
        record('unsettled_fraction_prevents_complete_nav',dict(cash=916.,nav=None,known_value=976.,
            held_units=0,pending_rights=['dist'],conservation_error=0.),
            lambda:share_lifecycle(portfolio(100,return_book=True),settle_fraction=False))
    return cases


def replay_current(directory,summary_sha):
    """Recompute an older pinned case artifact without claiming old code replay.

    verify() keeps its strict same-code contract. This explicitly different
    entry point records current code and does not authenticate historical pins.
    """
    directory=Path(directory).resolve()
    if digest(directory/'summary.json')!=summary_sha:
        raise ValueError('golden summary pin mismatch')
    summary=load_plan(directory/'summary.json')
    if set(summary['artifacts'])!={'golden_cases.json','inputs.json'}:
        raise ValueError('unexpected golden artifacts')
    current={str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}
    def check():
        if digest(directory/'summary.json')!=summary_sha or any(
            digest(directory/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('golden artifact changed')
        if any(digest(Path(p))!=h for p,h in current.items()):
            raise ValueError('current golden code changed')
    check()
    saved=load_plan(directory/'golden_cases.json')
    if saved.get('version') not in ('3','4','5','6') or saved.get('evidence_mode')!='synthetic':
        raise ValueError('unsupported golden contract')
    rebuilt=evaluate(saved['version'])
    if saved.get('cases')!=rebuilt: raise ValueError('current golden replay differs')
    passed=sum(c['passed'] for c in rebuilt)
    if (summary.get('cases')!=len(rebuilt) or summary.get('passed')!=passed
        or summary.get('all_passed')!=all(c['passed'] for c in rebuilt)
        or any(summary.get(k) is not False for k in
               ('real_fill_evidence','full_H02_coverage','formal_training_authorized'))):
        raise ValueError('golden summary semantics mismatch')
    check()
    return dict(cases=len(rebuilt),passed=passed,all_passed=all(c['passed'] for c in rebuilt),
        current_code_recomputed=True,historical_code_replayed=False,
        historical_code_provenance_authenticated=False,current_code_pins=current,
        original_inputs_sha256=summary['artifacts']['inputs.json'],
        real_fill_evidence=False,full_H02_coverage=False,formal_training_authorized=False)


def build(root):
    code=list(Path(__file__).parent.glob('*.py'))
    pins={str(p.resolve()):digest(p) for p in code}
    cases=evaluate()
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('golden code changed')
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('execution-golden-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'golden_cases.json',dict(version='6',evidence_mode='synthetic',cases=cases))
    atomic_json(out/'inputs.json',dict(code_pins=pins))
    report=dict(directory=str(out),at=now(),cases=len(cases),passed=sum(c['passed'] for c in cases),
        all_passed=all(c['passed'] for c in cases),real_fill_evidence=False,full_H02_coverage=False,
        formal_training_authorized=False,
        remaining_case_families=['corporate actions and tax','portfolio cash/overlap/capacity','effective market rule provenance'],
        artifacts={n:digest(out/n) for n in ['golden_cases.json','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
