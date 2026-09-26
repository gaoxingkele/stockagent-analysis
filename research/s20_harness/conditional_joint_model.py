"""Fixed P-joint conditional three-head reference for frozen experiment E04."""
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from .feature_pipeline import prepare
from .joint_model import CLASSES
from .baseline_model import validate_seed


def combine(up,down_if_up,down_if_not_up):
    values=[np.asarray(v,dtype=float) for v in [up,down_if_up,down_if_not_up]]
    if any(v.ndim!=1 or v.shape!=values[0].shape or not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in values):
        raise ValueError('aligned finite conditional probability vectors required')
    u,du,dn=values
    return np.column_stack([u*(1-du),u*du,(1-u)*(1-dn),(1-u)*dn])


def run(samples,features,fit_labels,boundaries,feature_contract,*,target_id,anchor_context=None,random_seed=20):
    validate_seed(random_seed)
    if target_id!='P.joint.v4':raise ValueError('explicit P.joint.v4 target required')
    matrix,assignment,pre=prepare(samples,features,boundaries,feature_contract,anchor_context=anchor_context)
    if fit_labels.columns.duplicated().any() or set(fit_labels.columns)!={'sample_id','target'} or fit_labels.sample_id.isna().any() or fit_labels.sample_id.duplicated().any():
        raise ValueError('exact unique conditional fit labels required')
    ids=pre['fit_sample_ids']
    if set(fit_labels.sample_id)!=set(ids):raise ValueError('only exact eligible conditional fit labels may be consumed')
    y=fit_labels.set_index('sample_id').loc[ids,'target']
    if y.isna().any() or not y.isin(CLASSES).all() or set(y)!=set(CLASSES):
        raise ValueError('all four resolved classes required for diverse conditional heads')
    is_up=y.isin(['A','B']);columns=pre['feature_columns'];indexed=matrix.set_index('sample_id')
    parameters=dict(C=1.,solver='lbfgs',max_iter=1000,random_state=random_seed)
    specifications=[('up',y.index,is_up),('down_if_up',y.index[is_up],y.loc[is_up].eq('B')),
                    ('down_if_not_up',y.index[~is_up],y.loc[~is_up].eq('D'))]
    models={};heads={}
    for name,members,labels in specifications:
        model=LogisticRegression(**parameters)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(indexed.loc[members,columns],labels)
        if list(model.classes_)!=[False,True]:raise ValueError('conditional class order mismatch')
        models[name]=model
        heads[name]=dict(fit_sample_ids=members.tolist(),positive_count=int(labels.sum()),
            negative_count=int((~labels).sum()),coefficients=model.coef_.tolist(),intercepts=model.intercept_.tolist(),
            iterations=model.n_iter_.tolist())
    output=assignment[['sample_id','segment','feature_ready']].copy().reset_index(drop=True)
    names=['p_A','p_B','p_C','p_D','p_up','p_down5']
    for name in names:output[name]=np.nan
    output['prediction_status']='not_forward_segment'
    forward=output.segment.isin(['tune','calibration','selection-policy','outer-test'])
    output.loc[forward,'prediction_status']='feature_unavailable';ready=forward&output.feature_ready
    if ready.any():
        x=indexed.loc[output.loc[ready,'sample_id'],columns]
        p=combine(*(models[name].predict_proba(x)[:,1] for name,_,_ in specifications))
        output.loc[ready,names[:4]]=p
        output.loc[ready,'p_up']=p[:,0]+p[:,1];output.loc[ready,'p_down5']=p[:,1]+p[:,3]
        output.loc[ready,'prediction_status']='uncalibrated_joint_reference'
    card=dict(target_id=target_id,model='fixed_conditional_three_logistic',classes=list(CLASSES),
        parameters=parameters,sklearn_version=sklearn.__version__,preprocessing=pre,fit_sample_ids=ids,
        randomness=dict(requested_seed=random_seed,seed_effect='unused_by_lbfgs',independent_market_evidence=False),
        fit_class_counts={c:int(y.eq(c).sum()) for c in CLASSES},heads=heads,
        probability_factorization='A=u*(1-du); B=u*du; C=(1-u)*(1-dn); D=(1-u)*dn',
        model_level_fits=1,underlying_fits=3,calibrator_fits=0,predicted_rows=int(ready.sum()),
        marginal_independence_assumed=False,all_candidates_retained=True,
        conditional_sample_sufficiency_proven=False,calibration_performed=False,absolute_probability_validated=False,
        formal_H04_accepted=False,formal_training_authorized=False,production_eligible=False)
    return output,card
