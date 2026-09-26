"""Full-calendar selection/rejection event accounting, not executable returns."""
import pandas as pd
from .metrics import binary_bounds


def table(ledgers,comparison):
    records=[]
    for trial in comparison['trials']:
        rows=ledgers[trial['policy_id']]
        calendar=[d['signal_date'] for d in trial['selection']['daily']]
        for day in ['all',*calendar]:
            frame=rows if day=='all' else rows.loc[rows.signal_date.eq(day)]
            days=len(calendar) if day=='all' else 1
            selected=frame.loc[frame.selected]
            active=selected.signal_date.nunique()
            for scope,part in [('all_candidates',frame),('selected',selected),('rejected',frame.loc[~frame.selected])]:
                for event in ['safe_profit','up','down5']:
                    # Convert nullable pandas bool to Python bool for the strict
                    # binary-bounds API; unresolved outcomes remain unknown.
                    values=[None if pd.isna(v) else bool(v) for v in part[event+'_target']]
                    bounds=binary_bounds(values)
                    records.append(dict(policy_id=trial['policy_id'],policy_sha256=trial['policy_sha256'],
                        target_id='P.joint.v4',evaluation_at=comparison['label_cutoff'],signal_date=day,
                        comparison_kind=comparison['comparison_kind'],
                        probability_source=comparison.get('probability_source','unspecified'),
                        same_daily_coverage=comparison['same_predictions_and_daily_coverage'],
                        scope=scope,event=event,calendar_days=days,active_days=active,empty_days=days-active,
                        active_day_coverage=active/days,group_candidates=len(frame),group_selected=len(selected),
                        selection_fraction=len(selected)/len(frame) if len(frame) else None,
                        denominator=bounds['denominator'],known=bounds['known'],positive=bounds['positive'],
                        unknown=bounds['unknown'],event_rate_lower=bounds['rate_lower'],event_rate_upper=bounds['rate_upper'],
                        event_rate_known_only=bounds['known_only_rate'],bounds_are_confidence_intervals=False,
                        risk10_evaluated=False,executed_returns_evaluated=False,formal_H05_accepted=False))
    if not records:raise ValueError('nonempty policy comparison required')
    return pd.DataFrame(records)


def csv_text(ledgers,comparison):
    return table(ledgers,comparison).to_csv(index=False,lineterminator='\n')
