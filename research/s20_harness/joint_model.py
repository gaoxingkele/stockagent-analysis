"""Fixed multinomial P-v4 reference; raw joint probabilities, not confidence."""
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from .feature_pipeline import prepare
from .baseline_model import validate_seed

CLASSES=('A','B','C','D')


def validate_weights(weights):
    if not isinstance(weights,dict) or set(weights)!=set(CLASSES) or any(
        isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in weights.values()):
        raise ValueError('exact positive finite four-class weights required')
    return dict(weights)


def run(samples,features,fit_labels,boundaries,feature_contract,*,target_id,anchor_context=None,class_weights=None,random_seed=20,model_family='multinomial'):
    validate_seed(random_seed)
    if model_family not in ['multinomial','shallow_joint']:raise ValueError('unsupported joint estimator')
    if model_family=='shallow_joint' and class_weights is not None:raise ValueError('tree weights not registered')
    if target_id!='P.joint.v4':
        raise ValueError('explicit P.joint.v4 target required')
    matrix,assignment,preprocessing=prepare(samples,features,boundaries,feature_contract,anchor_context=anchor_context)
    if fit_labels.columns.duplicated().any() or set(fit_labels.columns)!={'sample_id','target'}:
        raise ValueError('exact joint fit label schema required')
    if fit_labels.sample_id.isna().any() or fit_labels.sample_id.duplicated().any():
        raise ValueError('unique joint fit labels required')
    ids=preprocessing['fit_sample_ids']
    if set(fit_labels.sample_id)!=set(ids):
        raise ValueError('only exact eligible fit labels may be consumed')
    y=fit_labels.set_index('sample_id').loc[ids,'target']
    if y.isna().any() or not y.isin(CLASSES).all() or set(y)!=set(CLASSES):
        raise ValueError('all four resolved P classes required; no silent fallback')
    parameters=dict(C=1.,solver='lbfgs',max_iter=1000,random_state=random_seed)
    if model_family=='shallow_joint':parameters=dict(max_depth=3,min_samples_leaf=5,random_state=random_seed)
    sample_weights=None;weight_evidence=None
    if class_weights is not None:
        weights=validate_weights(class_weights)
        raw_weights=y.map(weights).to_numpy(dtype=float);normalizer=float(raw_weights.mean())
        if not np.isfinite(normalizer) or normalizer<=0:raise ValueError('invalid fit-weight normalization')
        sample_weights=raw_weights/normalizer
        weight_evidence=dict(class_weights=weights,fit_mean_weight=normalizer,
            normalization='fit sample weights normalized to mean one',
            preserves_total_fit_weight=True,natural_class_probability_validated=False)
    model=DecisionTreeClassifier(**parameters) if model_family=='shallow_joint' else LogisticRegression(**parameters)
    columns=preprocessing['feature_columns'];indexed=matrix.set_index('sample_id')
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(indexed.loc[ids,columns],y,sample_weight=sample_weights)
    if tuple(model.classes_)!=CLASSES: raise ValueError('unexpected joint class order')
    output=assignment[['sample_id','segment','feature_ready']].copy().reset_index(drop=True)
    names=['p_A','p_B','p_C','p_D','p_up','p_down5']
    for name in names: output[name]=np.nan
    output['prediction_status']='not_forward_segment'
    forward=output.segment.isin(['tune','calibration','selection-policy','outer-test'])
    output.loc[forward,'prediction_status']='feature_unavailable'
    ready=forward & output.feature_ready
    if ready.any():
        p=model.predict_proba(indexed.loc[output.loc[ready,'sample_id'],columns])
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):
            raise ValueError('invalid joint probability simplex')
        output.loc[ready,names[:4]]=p
        output.loc[ready,'p_up']=p[:,0]+p[:,1]
        output.loc[ready,'p_down5']=p[:,1]+p[:,3]
        output.loc[ready,'prediction_status']='uncalibrated_joint_reference'
    card=dict(target_id=target_id,model='fixed_multinomial_logistic',classes=list(CLASSES),
        class_meaning=dict(A='up_and_not_down5',B='up_and_down5',C='not_up_and_not_down5',D='not_up_and_down5'),
        parameters=parameters,sklearn_version=sklearn.__version__,preprocessing=preprocessing,
        randomness=dict(requested_seed=random_seed,seed_effect='unused_by_lbfgs',independent_market_evidence=False),
        fit_sample_ids=ids,fit_class_counts={k:int(y.eq(k).sum()) for k in CLASSES},
        model_level_fits=1,underlying_fits=1,calibrator_fits=0,all_candidates_retained=True,
        predicted_rows=int(ready.sum()),marginal_independence_assumed=False,
        calibration_performed=False,absolute_probability_validated=False,
        formal_H04_accepted=False,formal_training_authorized=False,production_eligible=False)
    if model_family=='shallow_joint':
        tree=model.tree_
        card['model']='fixed_shallow_joint_tree'
        card['randomness']['seed_effect']='feature_permutation_and_split_ties'
        card['tree_state']=dict(left=tree.children_left.tolist(),right=tree.children_right.tolist(),
            feature=tree.feature.tolist(),threshold=tree.threshold.tolist(),values=tree.value[:,0,:].tolist())
    else:card.update(coefficients=model.coef_.tolist(),intercepts=model.intercept_.tolist(),iterations=model.n_iter_.tolist())
    if weight_evidence is not None:
        card['model']='fixed_cost_sensitive_multinomial'
        card['training_weight_evidence']=weight_evidence
    return output,card
