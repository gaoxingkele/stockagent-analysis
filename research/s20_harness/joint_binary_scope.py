"""Explicit P.A label projection for controlled binary/joint diagnostics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from .runtime import load_plan


def project(joint, *, model_family='logistic'):
    required={'schema_version','evidence_mode','target_id','samples','features','fit_labels',
              'calibration_labels','boundaries','feature_contract','policy','calendar'}
    if set(joint)!=required or joint['schema_version']!='joint-1' or joint['target_id']!='P.joint.v4':
        raise ValueError('exact joint reference plan required')
    if model_family not in ['logistic','shallow_tree']:
        raise ValueError('supported binary control family required')
    binary=deepcopy(joint)
    for field in ['fit_labels','calibration_labels']:
        records=joint[field]
        if not isinstance(records,list) or any(not isinstance(r,dict) or set(r)!={'sample_id','target'}
                                              or r['target'] not in ['A','B','C','D'] for r in records):
            raise ValueError('resolved joint training classes required')
        if len({r['sample_id'] for r in records})!=len(records):
            raise ValueError('unique joint training label identities required')
        binary[field]=[dict(sample_id=r['sample_id'],target=r['target']=='A') for r in records]
    policy=joint['policy']['selection']
    if policy['mode']!='risk_gated' or policy['target_id']!='P.safe.v4' or policy['risk_target_id']!='P.down5.v4':
        raise ValueError('explicit joint safe-profit/down5 policy required')
    binary.update(schema_version='1',target_id='P.safe.v4',model_family=model_family,
        policy=dict(policy_id='joint-A-binary-control',target_id='P.safe.v4',risk_target_id=None,
                    mode='score_only_control',frozen_at=policy['frozen_at'],n_cap=policy['n_cap'],
                    min_score=policy['min_score'],max_risk=None))
    return binary


def validate(joint,binary):
    expected=project(joint,model_family=binary.get('model_family','logistic'))
    def canonical(value):
        return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    if canonical(expected)!=canonical(binary):
        raise ValueError('binary/joint shared scope or A-label projection mismatch')
    return dict(shared_scope_sha256=hashlib.sha256(canonical(expected).encode()).hexdigest(),
        binary_target='P.safe.v4',joint_target='P.joint.v4',positive_joint_classes=['A'],
        same_samples_features_splits=True,fit_and_calibration_projection_verified=True,
        policy_identical=False,policy_difference='binary score-only versus joint risk-gated utility',
        pure_model_effect_identified=False,formal_promotion_authorized=False)


def bind(joint_directory,joint_sha,binary_directory,binary_sha):
    from .joint_run import verify as verify_joint
    from .baseline_run import verify as verify_binary
    joint_directory=Path(joint_directory).resolve();binary_directory=Path(binary_directory).resolve()
    verify_joint(joint_directory,joint_sha);verify_binary(binary_directory,binary_sha)
    result=validate(load_plan(joint_directory/'input.json'),load_plan(binary_directory/'input.json'))
    # Both trained pipelines must have consumed the exact same eligible IDs.
    joint_card=load_plan(joint_directory/'joint_card.json');binary_card=load_plan(binary_directory/'baseline_card.json')
    if joint_card['fit_sample_ids']!=binary_card['fit_sample_ids']:
        raise ValueError('binary/joint consumed fit membership mismatch')
    jc=load_plan(joint_directory/'calibration_card.json');bc=load_plan(binary_directory/'calibration_card.json')
    if jc['calibration_sample_ids']!=bc['calibration_sample_ids']:
        raise ValueError('binary/joint consumed calibration membership mismatch')
    verify_joint(joint_directory,joint_sha);verify_binary(binary_directory,binary_sha)
    return dict(result,joint_directory=str(joint_directory),joint_summary_sha256=joint_sha,
        binary_directory=str(binary_directory),binary_summary_sha256=binary_sha,
        consumed_membership_verified=True,models_refit=0)


def compare_ranking(joint_directory,joint_sha,binary_directory,binary_sha,outcomes,*,evaluation_at,k):
    """Fixed same-date top-k diagnostic, not either saved deployable policy."""
    import pandas as pd
    from .baseline_evaluation import evaluate
    if type(k) is not int or k not in [1,3,5,10,20]:
        raise ValueError('registered diagnostic top-k cap required')
    binding=bind(joint_directory,joint_sha,binary_directory,binary_sha)
    joint_directory=Path(joint_directory);binary_directory=Path(binary_directory)
    plan=load_plan(joint_directory/'input.json')
    if outcomes.columns.duplicated().any() or set(outcomes.columns)!={'sample_id','target','label_available_at'}:
        raise ValueError('exact joint outcome schema required')
    if not outcomes.target.dropna().isin(['A','B','C','D']).all():
        raise ValueError('resolved joint outcome classes or unknown required')
    labels=outcomes.copy()
    labels['target']=pd.Series([None if pd.isna(v) else v=='A' for v in outcomes.target],index=outcomes.index,dtype=object)
    frames=[pd.read_parquet(d/'candidate_ledger.parquet') for d in [joint_directory,binary_directory]]
    identity=['sample_id','entity_id','signal_date','prediction_at']
    pd.testing.assert_frame_equal(frames[0][identity],frames[1][identity],check_exact=True)
    masks=[f.score.notna() for f in frames]
    if masks[0].tolist()!=masks[1].tolist():
        raise ValueError('ranking score availability mismatch; do not silently intersect candidates')
    assessed=[];metrics={}
    for name,frame in zip(['joint_pA','binary_safe'],frames):
        # Do not carry the saved policy's selected/rejection/utility fields into
        # a diagnostic policy that deliberately compares only the two scores.
        table=frame[identity+['score']].copy();table['selected']=False
        for day in plan['calendar']:
            eligible=table.loc[table.signal_date.eq(day)&table.score.notna()]
            ids=eligible.sort_values(['score','sample_id'],ascending=[False,True]).head(k).index
            table.loc[ids,'selected']=True
        rows,metrics[name]=evaluate(table,pd.DataFrame(plan['samples']),labels,
            evaluation_at=evaluation_at,calendar=plan['calendar'])
        rows['ranking_method']=name;assessed.append(rows)
    counts=[f.groupby('signal_date').selected.sum().to_dict() for f in assessed]
    if counts[0]!=counts[1]:raise ValueError('diagnostic ranking counts differ')
    bind(joint_directory,joint_sha,binary_directory,binary_sha)
    return pd.concat(assessed,ignore_index=True),dict(binding=binding,k_cap=k,metrics=metrics,
        daily_selected_counts=counts[0],calendar=plan['calendar'],same_day_same_count=True,
        scored_shortfall_days=[day for day in plan['calendar'] if counts[0].get(day,0)<k],
        all_candidate_rows_retained=True,original_policies_evaluated=False,
        learned_risk_gate_evaluated=False,diagnostic_only=True,models_refit=0,
        confidence_intervals_computed=False,formal_promotion_authorized=False)
