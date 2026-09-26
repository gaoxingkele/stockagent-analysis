"""Fixed bounded temperature calibration using only mature calibration labels."""
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from .splits import assign_segments,training_ids
from .joint_model import CLASSES


def run(samples,predictions,labels,boundaries,model_card,*,target_id,anchor_context=None,method='bounded_scalar_temperature'):
    if method not in ['bounded_scalar_temperature','identity_raw']:
        raise ValueError('unsupported joint calibration method')
    if target_id!='P.joint.v4' or model_card.get('target_id')!=target_id:
        raise ValueError('joint calibration target mismatch')
    prior=model_card.get('preprocessing',{}).get('anchor_admission')
    if (prior is None)!=(anchor_context is None): raise ValueError('joint calibration anchor mismatch')
    if anchor_context is not None:
        from .anchor_features import effective_samples
        samples,evidence=effective_samples(samples,anchor_context)
        if prior!=evidence: raise ValueError('joint calibration anchor evidence changed')
    assignment,split=assign_segments(samples,boundaries)
    if model_card['preprocessing']['split']['split_protocol_sha256']!=split['split_protocol_sha256']:
        raise ValueError('joint calibration split mismatch')
    columns=['p_A','p_B','p_C','p_D']
    required={'sample_id','segment','feature_ready','prediction_status','p_up','p_down5',*columns}
    if predictions.columns.duplicated().any() or set(predictions)!=required or predictions.sample_id.duplicated().any():
        raise ValueError('exact joint predictions required')
    if set(predictions.sample_id)!=set(samples.sample_id): raise ValueError('joint prediction universe mismatch')
    frame=predictions.set_index('sample_id').loc[samples.sample_id].reset_index()
    if frame.segment.tolist()!=assignment.segment.tolist() or frame.feature_ready.tolist()!=assignment.feature_ready.tolist():
        raise ValueError('joint prediction assignment mismatch')
    p=frame[columns].to_numpy(dtype=float)
    ready=(assignment.segment.isin(['tune','calibration','selection-policy','outer-test'])&assignment.feature_ready).to_numpy()
    if not np.isnan(p[~ready]).all() or not np.isfinite(p[ready]).all() or (p[ready]<0).any() or not np.allclose(p[ready].sum(axis=1),1,rtol=0,atol=1e-10):
        raise ValueError('joint probabilities availability/simplex mismatch')
    if not np.allclose(frame.p_up,p[:,0]+p[:,1],equal_nan=True) or not np.allclose(frame.p_down5,p[:,1]+p[:,3],equal_nan=True):
        raise ValueError('joint marginal mismatch')
    if method=='identity_raw':
        if labels.columns.duplicated().any() or set(labels)!={'sample_id','target'} or len(labels):
            raise ValueError('identity calibration requires explicit empty labels')
        output=frame.copy();later=assignment.segment.isin(['selection-policy','outer-test']).to_numpy()&ready
        for name in columns+['p_up','p_down5']:
            output['cal_'+name]=np.nan
            output.loc[later,'cal_'+name]=frame.loc[later,name].to_numpy()
        return output,dict(target_id=target_id,method=method,calibrator_fits=0,model_fits=0,
            calibration_sample_ids=[],forward_rows=int(later.sum()),all_candidates_retained=True,
            absolute_probability_validated=False,formal_H05_accepted=False,production_eligible=False)
    ids=training_ids(assignment,'calibration')
    if labels.columns.duplicated().any() or set(labels)!={'sample_id','target'} or labels.sample_id.duplicated().any() or set(labels.sample_id)!=set(ids):
        raise ValueError('exact eligible joint calibration labels required')
    y=labels.set_index('sample_id').loc[ids,'target']
    if not y.isin(CLASSES).all() or y.nunique()<2: raise ValueError('resolved diverse calibration classes required')
    indexes=frame.set_index('sample_id').loc[ids,columns].to_numpy(dtype=float)
    logits=np.log(np.clip(indexes,1e-12,1));targets=np.array([CLASSES.index(v) for v in y])
    def loss(log_temperature):
        z=logits/np.exp(log_temperature)
        return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(z)),targets]))
    fitted=minimize_scalar(loss,bounds=(np.log(.1),np.log(10.)),method='bounded',options={'xatol':1e-8,'maxiter':500})
    if not fitted.success or not np.isfinite(fitted.fun): raise ValueError('temperature fit failed')
    temperature=float(np.exp(fitted.x));output=frame.copy()
    calibrated=['cal_'+c for c in columns]
    for name in calibrated+['cal_p_up','cal_p_down5']: output[name]=np.nan
    later=assignment.segment.isin(['selection-policy','outer-test']).to_numpy()&ready
    z=np.log(np.clip(p[later],1e-12,1))/temperature
    values=np.exp(z-logsumexp(z,axis=1,keepdims=True))
    output.loc[later,calibrated]=values
    output.loc[later,'cal_p_up']=values[:,0]+values[:,1]
    output.loc[later,'cal_p_down5']=values[:,1]+values[:,3]
    return output,dict(target_id=target_id,method='bounded_scalar_temperature',temperature=temperature,
        temperature_bounds=[.1,10.],epsilon=1e-12,scipy_version=scipy.__version__,
        calibration_sample_ids=ids,calibration_class_counts={c:int(y.eq(c).sum()) for c in CLASSES},
        optimizer_evaluations=int(fitted.nfev),calibrator_fits=1,model_fits=0,
        calibrated_rows=int(later.sum()),all_candidates_retained=True,
        absolute_probability_validated=False,formal_H05_accepted=False,production_eligible=False)
