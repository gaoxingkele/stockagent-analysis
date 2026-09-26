"""Empirical mature fit-label frequencies on the shared eligible universe."""
import numpy as np
import pandas as pd
from .feature_pipeline import prepare
from .joint_model import CLASSES
from .baseline_model import validate_seed


def run(samples,features,fit_labels,boundaries,feature_contract,*,target_id,random_seed=20):
    validate_seed(random_seed)
    if target_id!='P.joint.v4':raise ValueError('explicit P.joint.v4 frequency target required')
    _,assignment,pre=prepare(samples,features,boundaries,feature_contract)
    if fit_labels.columns.duplicated().any() or set(fit_labels.columns)!={'sample_id','target'} or fit_labels.sample_id.isna().any() or fit_labels.sample_id.duplicated().any():
        raise ValueError('exact unique frequency labels required')
    ids=pre['fit_sample_ids']
    if not ids or set(ids)!=set(fit_labels.sample_id):raise ValueError('exact nonempty eligible frequency labels required')
    y=fit_labels.set_index('sample_id').loc[ids,'target']
    if y.isna().any() or not y.isin(CLASSES).all():raise ValueError('resolved frequency classes required')
    counts={c:int(y.eq(c).sum()) for c in CLASSES};p=np.array([counts[c]/len(ids) for c in CLASSES])
    out=assignment[['sample_id','segment','feature_ready']].copy().reset_index(drop=True)
    for name in ['p_A','p_B','p_C','p_D','p_up','p_down5']:out[name]=np.nan
    forward=out.segment.isin(['tune','calibration','selection-policy','outer-test']);ready=forward&out.feature_ready
    out['prediction_status']='not_forward_segment';out.loc[forward,'prediction_status']='feature_unavailable'
    out.loc[ready,['p_A','p_B','p_C','p_D']]=np.tile(p,(int(ready.sum()),1))
    out.loc[ready,'p_up']=p[0]+p[1];out.loc[ready,'p_down5']=p[1]+p[3]
    out.loc[ready,'prediction_status']='uncalibrated_joint_reference'
    return out,dict(model='mature_empirical_frequency',target_id=target_id,classes=list(CLASSES),
        preprocessing=pre,fit_sample_ids=ids,fit_class_counts=counts,probabilities=p.tolist(),
        randomness=dict(requested_seed=random_seed,seed_effect='deterministic_frequency',independent_market_evidence=False),
        smoothing=0,model_level_fits=0,underlying_fits=0,calibrator_fits=0,label_frequency_estimates=1,
        predicted_rows=int(ready.sum()),all_candidates_retained=True,absolute_probability_validated=False,
        formal_H04_accepted=False,formal_training_authorized=False,production_eligible=False)
