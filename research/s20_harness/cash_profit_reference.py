"""Independent scalar cash-only P reference. No bonus/tax/fill assumptions inferred."""
import math


def classify(dates,opens,highs,lows,closes,events,*,buy_cost=.001,sell_cost=.0015):
    if (not dates or dates!=sorted(set(dates)) or any(len(x)!=len(dates) for x in [opens,highs,lows,closes])
        or not math.isfinite(buy_cost) or buy_cost<0 or not math.isfinite(sell_cost) or not 0<=sell_cost<1):
        raise ValueError('invalid cash reference window/costs')
    if len({e.event_id for e in events})!=len(events): raise ValueError('duplicate cash event')
    relevant=[e for e in events if dates[0]<=e.record_date<=dates[-1]]
    if any(e.bonus_per_share or e.record_date not in dates for e in relevant):
        raise ValueError('cash-only reference requires observed record dates and no bonus')
    entry=float(opens[0]);entry_cost=entry*(1+buy_cost)
    peak=0.;minimum=float('inf');drawdown=0.;first_up=first_down=first_risk=None
    bars=[]
    for i,(date,o,h,l,c) in enumerate(zip(dates,opens,highs,lows,closes),1):
        if not all(math.isfinite(v) and v>0 for v in [o,h,l,c]) or h<max(o,l,c) or l>min(o,h,c):
            raise ValueError('unknown/invalid raw path')
        # One original share: recognized cash changes form at payment, not value.
        recognized=[e for e in relevant if e.ex_date<=date and e.cash_per_share>0]
        paid=sum(e.cash_per_share for e in recognized if e.pay_date<=date)
        receivable=sum(e.cash_per_share for e in recognized if e.pay_date>date)
        total=paid+receivable
        eh,el=h+total,l+total
        peak=max(peak,eh);minimum=min(minimum,el);drawdown=min(drawdown,el/peak-1)
        if first_up is None and eh>entry: first_up=i
        if first_down is None and el<entry: first_down=i
        if first_risk is None and el<entry_cost*.95: first_risk=i
        bars.append(dict(date=date,open=o+total,high=eh,low=el,close=c+total,cash=paid,receivable_cash=receivable))
    final=bars[-1]
    proceeds=closes[-1]*(1-sell_cost)+final['cash']+final['receivable_cash']
    net=proceeds/entry_cost-1;up=net>0;risk=first_risk is not None
    return dict(p_class=('B' if risk else 'A') if up else ('D' if risk else 'C'),up_event=up,b5=risk,
        terminal_net=net,mae=minimum/entry-1,max_drawdown=drawdown,first_up_day=first_up,
        first_down_day=first_down,time_to_b5=first_risk,bars=bars,
        cash=final['cash'],receivable_cash=final['receivable_cash'],formal_training_authorized=False)
