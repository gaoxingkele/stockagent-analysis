"""Selection-only fixed-prediction ranking ablation, not a formal selector."""
import pandas as pd
from .joint_policy import apply, validate
from .joint_evaluation import evaluate
from .label_availability import _instant
from .splits import assign_segments
from .trial_budget import canonical


def compare(samples, candidates, outcomes, boundaries, calendar, registry):
    required={'registry_id','target_id','registered_at','policies'}|({'comparison_kind'} if 'comparison_kind' in registry else set())
    if set(registry)!=required or registry['target_id']!='P.joint.v4':
        raise ValueError('exact joint comparison registry required')
    kind=registry.get('comparison_kind','fixed_gate_ranking')
    if kind not in ['fixed_gate_ranking','registered_gate_frontier']:
        raise ValueError('unsupported policy comparison kind')
    if not isinstance(registry['registry_id'],str) or not registry['registry_id'].strip():
        raise ValueError('named registry required')
    policies=registry['policies']
    if not isinstance(policies,list) or not 2<=len(policies)<=6:
        raise ValueError('two to six explicit policies required')
    registered=_instant(registry['registered_at'])
    start=_instant(boundaries[3]['start_at']);end=_instant(boundaries[3]['end_at'])
    cutoff=_instant(boundaries[4]['start_at'])
    if registered>=start:raise ValueError('registry must precede selection segment')
    ids=[];gates=[];rankings=[]
    for policy in policies:
        rankings.append(validate(policy));selection=policy['selection']
        ids.append(selection['policy_id'])
        if _instant(selection['frozen_at'])>registered:raise ValueError('policy not frozen at registration')
        varying={'policy_id'} if kind=='fixed_gate_ranking' else {'policy_id','max_risk','min_score','n_cap'}
        gates.append(canonical({k:v for k,v in selection.items() if k not in varying}))
    if len(set(ids))!=len(ids):raise ValueError('unique policy IDs required')
    if len(set(gates))!=1:raise ValueError('identical fixed risk gates required for ranking ablation')
    if (rankings.count('safe_probability')!=1 if kind=='fixed_gate_ranking' else 'safe_probability' not in rankings) or 'penalized_utility' not in rankings:
        raise ValueError('one direct probability control and penalized comparator required')
    assignment,split=assign_segments(samples,boundaries)
    selected_segment=assignment.segment.eq('selection-policy')
    expected=assignment.loc[selected_segment,'sample_id']
    if candidates.sample_id.duplicated().any() or set(candidates.sample_id)!=set(expected):
        raise ValueError('exact complete selection candidate universe required')
    readiness=assignment.set_index('sample_id').feature_ready
    probabilities=['p_A','p_B','p_C','p_D']
    if (candidates[probabilities].notna().any(axis=1)&~candidates.sample_id.map(readiness)).any():
        raise ValueError('unavailable features cannot yield selection probabilities')
    eligible=assignment.loc[selected_segment&assignment.label_ready,'sample_id']
    if set(outcomes)!= {'sample_id','target','label_available_at'} or outcomes.sample_id.duplicated().any() or set(outcomes.sample_id)!=set(eligible):
        raise ValueError('exact mature selection outcomes required')
    meta=samples.set_index('sample_id')
    for row in outcomes.itertuples(index=False):
        if _instant(row.label_available_at)!=_instant(meta.loc[row.sample_id,'label_available_at']):
            raise ValueError('outcome maturity differs from manifest')
    for day in calendar:
        close=pd.to_datetime(day,format='%Y%m%d').tz_localize('Asia/Shanghai')+pd.Timedelta(hours=15)
        if not start<=close<end:raise ValueError('calendar outside selection segment')
    ledgers={};trials=[];counts=[]
    for policy in policies:
        rows,selection=apply(candidates,policy,calendar)
        daily=[r['selected'] for r in selection['daily']]
        if kind=='fixed_gate_ranking' and counts and counts[0]!=daily:raise ValueError('fixed gate changed daily coverage')
        counts.append(daily)
        assessed,metrics=evaluate(rows,samples,outcomes,target_id='P.joint.v4',
                                  evaluation_at=cutoff.isoformat(),calendar=calendar)
        pid=policy['selection']['policy_id'];ledgers[pid]=assessed
        trials.append(dict(policy_id=pid,policy_sha256=selection['policy_sha256'],
                           selection=selection,metrics=metrics))
    return ledgers,dict(registry=registry,split=split,trials=trials,
        model_fits=0,calibrator_fits=0,policy_evaluations=len(policies),
        comparison_kind=kind,same_predictions=True,
        same_predictions_and_daily_coverage=all(c==counts[0] for c in counts),
        matched_coverage_gain_proven=False,outer_outcomes_consumed=False,
        selected_policy_id=None,automatic_winner_selected=False,
        label_cutoff=cutoff.isoformat(),historical_registry_receipt_proven=False,
        shared_budget_enforced=False,formal_H05_accepted=False)
