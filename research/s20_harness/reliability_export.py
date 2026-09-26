"""Flatten reconstructed joint diagnostics; never pool seeds or fit a policy."""
import pandas as pd


def table(jobs):
    records = []
    for job in jobs:
        identity = {k: job[k] for k in ('candidate_id', 'job_id', 'fold_id', 'random_seed', 'input_sha256')}
        report = job['metrics']
        for event, panel in report['events'].items():
            groups = [('all_candidates', 'all', panel['all_candidates']),
                      ('selected_candidates', 'all', panel['selected_candidates'])]
            groups.extend(('selected_daily', d['signal_date'], d['selected']) for d in panel['daily'])
            for scope, day, group in groups:
                for i, bucket in enumerate(group['reliability_bins']):
                    bounds = bucket['event_bounds']
                    records.append(dict(identity, target_id=report['target_id'],
                        evaluation_at=report['evaluation_at'], event=event, scope=scope,
                        signal_date=day, bin_index=i, lower=bucket['lower'], upper=bucket['upper'],
                        group_candidates=group['event_bounds']['denominator'],
                        group_unknown_outcomes=group['event_bounds']['unknown'],
                        group_missing_predictions=group['missing_prediction_rows'],
                        scored_count=bucket['scored_count'], known_count=bucket['known_count'],
                        positive_count=bounds['positive'], unknown_count=bounds['unknown'],
                        mean_prediction_all_scored=bucket['mean_prediction_all_scored'],
                        mean_prediction_known_only=bucket['mean_prediction'],
                        observed_rate_known_only=bucket['observed_rate'],
                        known_fraction=bucket['known_fraction'],
                        event_rate_lower=bounds['rate_lower'], event_rate_upper=bounds['rate_upper'],
                        prediction_minus_rate_lower=bucket['prediction_minus_rate_lower'],
                        prediction_minus_rate_upper=bucket['prediction_minus_rate_upper'],
                        bounds_are_confidence_intervals=False, probability_reliability_proven=False,
                        formal_H05_accepted=False))
    if not records:
        raise ValueError('nonempty reconstructed evaluation jobs required')
    result = pd.DataFrame(records)
    if result.duplicated(['job_id', 'event', 'scope', 'signal_date', 'bin_index']).any():
        raise ValueError('duplicate reliability identity')
    return result


def csv_text(jobs):
    # Verification compares canonical bytes, avoiding lossy CSV type inference
    # (e.g. fold "0", empty null cells, float round-trips or boolean coercion).
    return table(jobs).to_csv(index=False, lineterminator='\n')
