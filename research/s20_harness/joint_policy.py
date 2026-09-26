"""Joint-probability risk gate followed by a separate penalized utility rank."""
import hashlib
import json
import math
import numpy as np
import pandas as pd
from .recommendation_policy import apply as gate


def validate(policy):
    if not isinstance(policy,dict):
        raise ValueError('exact joint policy required')
    ranking=policy.get('ranking','penalized_utility')
    if ranking=='safe_probability':
        if set(policy)!={'selection','ranking'}:raise ValueError('safe probability policy forbids weights')
    elif ranking=='penalized_utility':
        expected={'selection','weights'}|({'ranking'} if 'ranking' in policy else set())
        if set(policy)!=expected or not isinstance(policy['weights'],dict) or set(policy['weights'])!={'lambda','mu','nu'}:
            raise ValueError('exact joint policy required')
        lam,mu,nu=(policy['weights'][k] for k in ['lambda','mu','nu'])
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in [lam,mu,nu]) or not mu>lam>nu>=0:
            raise ValueError('joint weights require mu > lambda > nu >= 0')
    else:raise ValueError('unsupported joint ranking')
    selection=policy['selection']
    if not isinstance(selection,dict) or selection.get('mode')!='risk_gated' or selection.get('target_id')!='P.safe.v4' or selection.get('risk_target_id')!='P.down5.v4':
        raise ValueError('joint policy requires explicit safe-profit/down5 risk gate')
    return ranking


def apply(candidates,policy,calendar):
    ranking=validate(policy);selection=policy['selection']
    identity=['sample_id','entity_id','signal_date','prediction_at'];probabilities=['p_A','p_B','p_C','p_D']
    if candidates.columns.duplicated().any() or set(candidates.columns)!=set(identity+probabilities):
        raise ValueError('exact joint candidates without outcomes required')
    for column in probabilities:
        if not pd.api.types.is_numeric_dtype(candidates[column]) or pd.api.types.is_bool_dtype(candidates[column]):
            raise ValueError('numeric joint probabilities required')
    complete=candidates[probabilities].notna().all(axis=1)
    if (candidates[probabilities].notna().any(axis=1)&~complete).any():
        raise ValueError('partial joint probability vector')
    p=candidates.loc[complete,probabilities].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,rtol=0,atol=1e-10):
        raise ValueError('joint probabilities must form a simplex')
    reference=candidates[identity].copy()
    reference['score']=candidates.p_A
    reference['risk']=candidates.p_B+candidates.p_D
    rows,report=gate(reference,selection,calendar)
    for name in probabilities: rows[name]=candidates[name].to_numpy()
    rows['p_up']=rows.p_A+rows.p_B
    rows['utility_rank']=np.nan
    rank_column='score'
    if ranking=='penalized_utility':
        lam,mu,nu=(policy['weights'][k] for k in ['lambda','mu','nu'])
        rows['utility_rank']=rows.p_A-lam*rows.p_B-mu*rows.p_D-nu*rows.p_C
        rank_column='utility_rank'
    eligible=rows.selected|rows.reject_reason.eq('not_in_topn')
    rows.loc[eligible,'selected']=False
    rows.loc[eligible,'reject_reason']='not_in_topn'
    for date in calendar:
        chosen=rows.loc[eligible&rows.signal_date.eq(date)].sort_values([rank_column,'sample_id'],ascending=[False,True]).head(selection['n_cap'])
        rows.loc[chosen.index,'selected']=True;rows.loc[chosen.index,'reject_reason']=None
    sha=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    rows['policy_sha256']=sha
    report.update(policy=policy,policy_sha256=sha,selection_order='hard_down5_gate_then_'+ranking,
        utility_is_probability=False,probabilities_calibrated_or_validated=False,
        score_meaning='p_A_safe_profit_not_utility',risk_meaning='p_B_plus_p_D_down5')
    return rows,report
