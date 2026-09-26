"""Recompute saved joint inference and decisions without optimizer/model fits."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp, expit
from .feature_pipeline import prepare
from .joint_run import verify
from .joint_policy import apply as select
from .runtime import digest,load_plan


def replay(directory,summary_sha):
    directory=Path(directory).resolve();verify(directory,summary_sha)
    pins=load_plan(directory/'inputs.json')
    current={str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}
    if {p:h for p,h in pins.items() if Path(p).suffix=='.py'}!=current:
        raise ValueError('joint replay computational source mismatch')
    plan=load_plan(directory/'input.json');card=load_plan(directory/'joint_card.json')
    from .baseline_model import validate_seed
    seed=validate_seed(plan.get('random_seed',20))
    effect='deterministic_frequency' if plan.get('model_family')=='mature_frequency' else 'unused_by_lbfgs'
    if plan.get('model_family')=='shallow_joint':effect='feature_permutation_and_split_ties'
    if card.get('randomness')!=dict(requested_seed=seed,seed_effect=effect,independent_market_evidence=False):
        raise ValueError('joint randomness contract mismatch')
    if effect!='deterministic_frequency' and card.get('parameters',{}).get('random_state')!=seed:
        raise ValueError('joint estimator seed mismatch')
    cal=load_plan(directory/'calibration_card.json');samples=pd.DataFrame(plan['samples'])
    matrix,assignment,pre=prepare(samples,pd.DataFrame(plan['features']),plan['boundaries'],plan['feature_contract'])
    if pre!=card['preprocessing'] or card['classes']!=['A','B','C','D']:
        raise ValueError('joint preprocessing/class reconstruction mismatch')
    raw=pd.read_parquet(directory/'joint_predictions.parquet')
    pd.testing.assert_frame_equal(raw[['sample_id','segment','feature_ready']],assignment[['sample_id','segment','feature_ready']].reset_index(drop=True))
    columns=['p_A','p_B','p_C','p_D'];ready=assignment.segment.isin(['tune','calibration','selection-policy','outer-test'])&assignment.feature_ready
    x=matrix.loc[ready,pre['feature_columns']].to_numpy()
    if plan.get('model_family')=='mature_frequency':
        if card['model']!='mature_empirical_frequency':raise ValueError('frequency model identity mismatch')
        labels=pd.DataFrame(plan['fit_labels']).set_index('sample_id').loc[pre['fit_sample_ids'],'target']
        if labels.empty or not labels.isin(['A','B','C','D']).all():raise ValueError('invalid frequency fit labels')
        counts={c:int(labels.eq(c).sum()) for c in ['A','B','C','D']}
        p=[counts[c]/len(labels) for c in ['A','B','C','D']]
        if card['fit_class_counts']!=counts or card['probabilities']!=p or card['fit_sample_ids']!=pre['fit_sample_ids']:
            raise ValueError('frequency estimate reconstruction mismatch')
        probabilities=np.tile(p,(int(ready.sum()),1))
    elif plan.get('model_family')=='shallow_joint':
        from .joint_tree_inference import predict
        if card['model']!='fixed_shallow_joint_tree' or card['parameters']!=dict(max_depth=3,min_samples_leaf=5,random_state=seed):
            raise ValueError('joint tree identity mismatch')
        probabilities=predict(card['tree_state'],x)
    elif plan.get('model_family')=='conditional_three':
        from .conditional_joint_model import combine
        if card['model']!='fixed_conditional_three_logistic' or set(card['heads'])!={'up','down_if_up','down_if_not_up'}:
            raise ValueError('conditional model/head identity mismatch')
        probabilities=[]
        labels=pd.DataFrame(plan['fit_labels']).set_index('sample_id').loc[pre['fit_sample_ids'],'target']
        masks=[pd.Series(True,index=labels.index),labels.isin(['A','B']),labels.isin(['C','D'])]
        for name,mask in zip(['up','down_if_up','down_if_not_up'],masks):
            head=card['heads'][name]
            if head['fit_sample_ids']!=labels.index[mask].tolist():raise ValueError('conditional head membership mismatch')
            coef=np.asarray(head['coefficients'],dtype=float);intercept=np.asarray(head['intercepts'],dtype=float)
            if coef.shape!=(1,x.shape[1]) or intercept.shape!=(1,) or not np.isfinite(coef).all() or not np.isfinite(intercept).all():
                raise ValueError('conditional parameter shape/value mismatch')
            probabilities.append(expit((x@coef.T+intercept)[:,0]))
        probabilities=combine(*probabilities)
    else:
        weighted=plan.get('model_family')=='cost_sensitive_joint'
        if card['model']!=('fixed_cost_sensitive_multinomial' if weighted else 'fixed_multinomial_logistic'):
            raise ValueError('joint model identity mismatch')
        if weighted:
            labels=pd.DataFrame(plan['fit_labels']).set_index('sample_id').loc[pre['fit_sample_ids'],'target']
            mean=float(labels.map(plan['class_weights']).to_numpy(dtype=float).mean())
            evidence=dict(class_weights=plan['class_weights'],fit_mean_weight=mean,
                normalization='fit sample weights normalized to mean one',preserves_total_fit_weight=True,
                natural_class_probability_validated=False)
            if card['training_weight_evidence']!=evidence:raise ValueError('joint training-weight evidence mismatch')
        coefficients=np.asarray(card['coefficients'],dtype=float);intercept=np.asarray(card['intercepts'],dtype=float)
        if coefficients.shape!=(4,len(pre['feature_columns'])) or intercept.shape!=(4,) or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
            raise ValueError('joint parameter shape/value mismatch')
        z=x@coefficients.T+intercept
        probabilities=np.exp(z-logsumexp(z,axis=1,keepdims=True))
    expected=np.full((len(raw),4),np.nan);expected[ready]=probabilities
    np.testing.assert_allclose(raw[columns].to_numpy(),expected,rtol=1e-12,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(raw.p_up,expected[:,0]+expected[:,1],equal_nan=True)
    np.testing.assert_allclose(raw.p_down5,expected[:,1]+expected[:,3],equal_nan=True)
    calibrated=pd.read_parquet(directory/'calibrated_predictions.parquet')
    pd.testing.assert_frame_equal(calibrated[raw.columns],raw,check_exact=True)
    later=assignment.segment.isin(['selection-policy','outer-test'])&assignment.feature_ready
    # Use saved raw values after checking coefficient-based reproduction, so
    # temperature application itself can be compared at machine precision.
    expected_cal=np.full((len(raw),4),np.nan)
    if plan.get('calibration_method','bounded_scalar_temperature')=='identity_raw':
        from .joint_calibration import run as identity
        if plan['calibration_labels']:raise ValueError('identity calibration consumed labels')
        identity_out,identity_card=identity(samples,raw,pd.DataFrame(columns=['sample_id','target']),plan['boundaries'],card,
                                           target_id=plan['target_id'],method='identity_raw')
        if cal!=identity_card:raise ValueError('identity calibration card mismatch')
        pd.testing.assert_frame_equal(calibrated,identity_out,check_exact=True)
        expected_cal[later]=raw.loc[later,columns].to_numpy()
    else:
        if cal['method']!='bounded_scalar_temperature':raise ValueError('joint calibration method mismatch')
        temperature=cal['temperature'];epsilon=cal['epsilon']
        if not .1<=temperature<=10 or epsilon!=1e-12: raise ValueError('invalid saved temperature contract')
        logits=np.log(np.clip(raw.loc[later,columns].to_numpy(),epsilon,1))/temperature
        expected_cal[later]=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    np.testing.assert_allclose(calibrated[['cal_'+c for c in columns]],expected_cal,rtol=0,atol=1e-15,equal_nan=True)
    np.testing.assert_allclose(calibrated.cal_p_up,expected_cal[:,0]+expected_cal[:,1],equal_nan=True)
    np.testing.assert_allclose(calibrated.cal_p_down5,expected_cal[:,1]+expected_cal[:,3],equal_nan=True)
    outer=calibrated.loc[calibrated.segment.eq('outer-test')]
    candidates=samples.set_index('sample_id').loc[outer.sample_id,['entity_id','signal_date','prediction_at']].reset_index()
    for c in columns:candidates[c]=outer['cal_'+c].to_numpy()
    rows,policy=select(candidates,plan['policy'],plan['calendar'])
    pd.testing.assert_frame_equal(rows,pd.read_parquet(directory/'candidate_ledger.parquet'),check_exact=True)
    if policy!=load_plan(directory/'selection_report.json'): raise ValueError('joint policy report mismatch')
    verify(directory,summary_sha)
    if any(digest(Path(p))!=h for p,h in current.items()): raise ValueError('joint replay source changed')
    return dict(rows=len(raw),outer_candidates=len(rows),model_fits=0,calibrator_fits=0,
        saved_inference_recomputed=True,policy_recomputed=True,training_optimizer_reproduced=False,
        probability_reliability_proven=False,formal_H04_H05_accepted=False)
