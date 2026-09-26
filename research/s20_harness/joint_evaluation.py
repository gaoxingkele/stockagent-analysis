"""Mature P-joint outcome diagnostics, isolated from training and selection."""
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from .baseline_evaluation import evaluate as binary_evaluate
from .joint_replay import replay
from .joint_run import verify
from .runtime import atomic_json, digest, load_plan, now

EVENTS = {'safe_profit': ('A',), 'up': ('A','B'), 'down5': ('B','D')}


def evaluate(candidates, samples, outcomes, *, target_id, evaluation_at, calendar):
    if target_id != 'P.joint.v4':
        raise ValueError('explicit P.joint.v4 evaluation required')
    if outcomes.columns.duplicated().any() or set(outcomes.columns) != {'sample_id','target','label_available_at'}:
        raise ValueError('exact joint outcome schema required')
    if not outcomes.target.dropna().isin(['A','B','C','D']).all():
        raise ValueError('resolved joint class or unknown required')
    names=['p_A','p_B','p_C','p_D']
    if candidates.columns.duplicated().any():
        raise ValueError('duplicate candidate columns')
    p=candidates[names]
    if any(not pd.api.types.is_numeric_dtype(p[c]) or pd.api.types.is_bool_dtype(p[c]) for c in names):
        raise ValueError('numeric joint probabilities required')
    known=p.notna().all(axis=1)
    if (p.notna().any(axis=1)&~known).any() or not np.isfinite(p.loc[known]).all().all() or (p.loc[known]<0).any().any() or (p.loc[known]>1).any().any() or not np.allclose(p.loc[known].sum(axis=1),1,rtol=0,atol=1e-10):
        raise ValueError('complete joint simplex or all unknown required')
    if (candidates.selected & ~known).any():
        raise ValueError('selected candidate requires joint prediction')
    rows=candidates.copy().reset_index(drop=True);panels={}
    scores={'safe_profit':p.p_A,'up':p.p_A+p.p_B,'down5':p.p_B+p.p_D}
    for event,classes in EVENTS.items():
        labels=outcomes.copy()
        labels['target']=pd.Series([None if pd.isna(v) else v in classes for v in outcomes.target],index=outcomes.index,dtype=object)
        frame=candidates.copy();frame['score']=scores[event]
        assessed,panels[event]=binary_evaluate(frame,samples,labels,evaluation_at=evaluation_at,calendar=calendar)
        # Explicit nullable logical dtype survives parquet for all-known as
        # well as partially mature snapshots; don't rely on object inference.
        rows[event+'_target']=assessed.evaluation_target.astype('boolean')
        rows['evaluation_status']=assessed.evaluation_status
    labels=outcomes.set_index('sample_id').target
    rows['evaluation_class']=[labels.loc[sid] if status=='mature' else None
                              for sid,status in zip(rows.sample_id,rows.evaluation_status)]

    def joint_metrics(frame):
        valid=frame.evaluation_class.notna() & frame[names].notna().all(axis=1)
        scored=frame.loc[valid];probs=scored[names].to_numpy(dtype=float)
        truth=np.array([['A','B','C','D'].index(v) for v in scored.evaluation_class],dtype=int)
        onehot=np.eye(4)[truth]
        return dict(scored_mature_rows=len(scored),
            class_counts=frame.evaluation_class.value_counts().to_dict(),
            unknown_outcomes=int(frame.evaluation_class.isna().sum()),
            multiclass_brier_sum_known_only=float(np.mean(np.sum((probs-onehot)**2,axis=1))) if len(scored) else None,
            log_loss_known_only=float(-np.log(np.clip(probs[np.arange(len(scored)),truth],1e-12,1)).mean()) if len(scored) else None,
            log_loss_clip=1e-12,known_only_diagnostic=True)
    return rows,dict(target_id=target_id,evaluation_at=evaluation_at,events=panels,
        joint_all=joint_metrics(rows),joint_selected=joint_metrics(rows.loc[rows.selected]),
        all_candidates_retained=True,probability_reliability_proven=False,
        risk10_evaluated=False,opportunity_touch_evaluated=False,executed_returns_evaluated=False,
        confidence_intervals_computed=False,formal_promotion_authorized=False)


def build(root, run_directory, run_sha, outcome_path, outcome_sha):
    directory=Path(run_directory).resolve();source=Path(outcome_path).resolve()
    validation=replay(directory,run_sha)
    if digest(source)!=outcome_sha: raise ValueError('joint outcome pin mismatch')
    pins={str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}
    payload=load_plan(source);plan=load_plan(directory/'input.json')
    if set(payload)!={'target_id','evaluation_at','outcomes'} or not isinstance(payload['outcomes'],list) or any(
        not isinstance(r,dict) or set(r)!={'sample_id','target','label_available_at'} for r in payload['outcomes']):
        raise ValueError('exact joint evaluation payload required')
    if payload['target_id']!=plan['target_id']: raise ValueError('joint evaluation target mismatch')
    rows,metrics=evaluate(pd.read_parquet(directory/'candidate_ledger.parquet'),pd.DataFrame(plan['samples']),
        pd.DataFrame(payload['outcomes'],columns=['sample_id','target','label_available_at']),
        target_id=payload['target_id'],evaluation_at=payload['evaluation_at'],calendar=plan['calendar'])
    verify(directory,run_sha)
    if digest(source)!=outcome_sha or any(digest(Path(p))!=h for p,h in pins.items()):
        raise ValueError('joint evaluation dependency changed')
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('joint-evaluation-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    rows.to_parquet(out/'evaluated_candidates.parquet',index=False)
    atomic_json(out/'metrics.json',metrics);atomic_json(out/'code_pins.json',pins)
    report=dict(directory=str(out),at=now(),run_directory=str(directory),run_summary_sha256=run_sha,
        outcome_path=str(source),outcome_sha256=outcome_sha,source_validation=validation,
        evidence_mode=plan['evidence_mode'],rows=len(rows),selected=int(rows.selected.sum()),
        target_id=payload['target_id'],model_fits=0,calibrator_fits=0,formal_promotion_authorized=False,
        artifacts={n:digest(out/n) for n in ['evaluated_candidates.parquet','metrics.json','code_pins.json']})
    atomic_json(out/'summary.json',report)
    return report
