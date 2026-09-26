"""Compare probability error on frozen selections, without reselecting stocks."""
import pandas as pd
from .joint_evaluation import evaluate


def assess(samples,raw_predictions,outcomes,ledgers,comparison,calendar,*,selection_source='calibrated'):
    if selection_source not in ['raw','calibrated']:raise ValueError('explicit selection source required')
    raw=raw_predictions.set_index('sample_id')
    panels=[]
    for trial in comparison['trials']:
        rows=ledgers[trial['policy_id']]
        # Only IDs from the already validated selection segment can be read.
        subset=raw.loc[rows.sample_id]
        if not subset.segment.eq('selection-policy').all():
            raise ValueError('fixed calibration requires selection-only predictions')
        candidates=rows[['sample_id','entity_id','signal_date','prediction_at','selected']].copy()
        for col in ['p_A','p_B','p_C','p_D']:candidates[col]=subset[col].to_numpy()
        _,uncalibrated=evaluate(candidates,samples,outcomes,target_id='P.joint.v4',
            evaluation_at=comparison['label_cutoff'],calendar=calendar)
        for col in ['p_A','p_B','p_C','p_D']:candidates[col]=subset['cal_'+col].to_numpy()
        _,calibrated=evaluate(candidates,samples,outcomes,target_id='P.joint.v4',
            evaluation_at=comparison['label_cutoff'],calendar=calendar)
        deltas={}
        for scope in ['joint_all','joint_selected']:
            before=uncalibrated[scope];after=calibrated[scope]
            if before['scored_mature_rows']!=after['scored_mature_rows']:
                raise ValueError('calibration comparison scored membership differs')
            deltas[scope]={key:(after[key]-before[key] if after[key] is not None and before[key] is not None else None)
                for key in ['multiclass_brier_sum_known_only','log_loss_known_only']}
        panels.append(dict(policy_id=trial['policy_id'],policy_sha256=trial['policy_sha256'],
            selected_sample_ids=rows.loc[rows.selected,'sample_id'].tolist(),raw_metrics=uncalibrated,calibrated_metrics=calibrated,
            calibrated_minus_raw_known_only=deltas))
    return dict(panels=panels,selection_reference='saved_'+selection_source+'_policy_selection',
        identical_selected_membership=True,model_fits=0,calibrator_fits=0,policy_evaluations=0,
        reselected_raw_policy_evaluated=False,lower_error_is_better=True,
        known_only_diagnostic=True,calibration_improvement_proven=False,formal_H05_accepted=False)
