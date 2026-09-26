"""Independent B10 outcomes for a fixed joint policy; no invented B10 score."""
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from .joint_evaluation import evaluate as joint_evaluate
from .baseline_evaluation import evaluate as binary_evaluate
from .joint_replay import replay
from .joint_run import verify
from .runtime import atomic_json,digest,load_plan,now


def evaluate(candidates,samples,joint_outcomes,risk10_outcomes,*,joint_target_id,risk_target_id,evaluation_at,calendar):
    if risk_target_id!='P.B10.v4':raise ValueError('independent P.B10.v4 outcome endpoint required')
    rows,joint_metrics=joint_evaluate(candidates,samples,joint_outcomes,target_id=joint_target_id,
        evaluation_at=evaluation_at,calendar=calendar)
    risk_candidates=candidates.copy()
    risk_candidates['score']=np.nan
    risk_rows,risk_metrics=binary_evaluate(risk_candidates,samples,risk10_outcomes,
        evaluation_at=evaluation_at,calendar=calendar)
    rows['risk10_target']=risk_rows.evaluation_target.astype('boolean')
    rows['risk10_status']=risk_rows.evaluation_status
    # Check only independently mature known labels; don't resolve unknown
    # joint paths by inference from a different endpoint or maturity clock.
    contradictions=rows.risk10_target.eq(True).fillna(False)&rows.down5_target.eq(False)
    if contradictions.any():raise ValueError('mature risk10 contradicts full-window down5 label')
    for column in ['safe_profit_target','up_target','down5_target']:
        rows[column]=rows[column].astype('boolean')
    return rows,dict(joint=joint_metrics,risk10=risk_metrics,
        risk10_target_id=risk_target_id,risk10_probability_available=False,
        all_candidate_rows_retained=True,selection_recomputed=False,
        endpoint_consistency_checked=True,market_label_semantics_proven=False,
        executed_returns_evaluated=False,confidence_intervals_computed=False,formal_G3_passed=False)


def build(root,run_directory,run_sha,input_path,input_sha):
    directory=Path(run_directory).resolve();source=Path(input_path).resolve()
    validation=replay(directory,run_sha)
    if digest(source)!=input_sha:raise ValueError('endpoint input pin mismatch')
    payload=load_plan(source)
    if set(payload)!={'evaluation_at','joint','risk10'}:raise ValueError('exact paired endpoint input required')
    for name in ['joint','risk10']:
        value=payload[name]
        if not isinstance(value,dict) or set(value)!={'target_id','outcomes'} or not isinstance(value['outcomes'],list) or any(
            not isinstance(r,dict) or set(r)!={'sample_id','target','label_available_at'} for r in value['outcomes']):
            raise ValueError('exact independent endpoint records required')
    code={str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}
    plan=load_plan(directory/'input.json')
    frame=lambda name:pd.DataFrame(payload[name]['outcomes'],columns=['sample_id','target','label_available_at'])
    rows,metrics=evaluate(pd.read_parquet(directory/'candidate_ledger.parquet'),pd.DataFrame(plan['samples']),
        frame('joint'),frame('risk10'),joint_target_id=payload['joint']['target_id'],risk_target_id=payload['risk10']['target_id'],
        evaluation_at=payload['evaluation_at'],calendar=plan['calendar'])
    verify(directory,run_sha)
    if digest(source)!=input_sha or any(digest(Path(p))!=h for p,h in code.items()):raise ValueError('endpoint source changed')
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('joint-endpoints-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'input.json',payload);atomic_json(out/'metrics.json',metrics)
    rows.to_parquet(out/'evaluated_endpoints.parquet',index=False)
    atomic_json(out/'bindings.json',dict(run_directory=str(directory),run_summary_sha256=run_sha,
        input_path=str(source),input_sha256=input_sha,code=code))
    report=dict(directory=str(out),at=now(),rows=len(rows),source_validation=validation,
        evidence_mode=plan['evidence_mode'],models_refit=0,policy_reselected=False,formal_G3_passed=False,
        artifacts={n:digest(out/n) for n in ['input.json','metrics.json','evaluated_endpoints.parquet','bindings.json']})
    atomic_json(out/'summary.json',report)
    return report


def verify_evaluation(root,directory,summary_sha):
    directory=Path(directory).resolve()
    scope=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'
    if not directory.is_relative_to(scope) or digest(directory/'summary.json')!=summary_sha:
        raise ValueError('endpoint summary pin/scope mismatch')
    report=load_plan(directory/'summary.json')
    names={'input.json','metrics.json','evaluated_endpoints.parquet','bindings.json'}
    if report['directory']!=str(directory) or set(report['artifacts'])!=names:
        raise ValueError('endpoint artifact contract mismatch')
    pins={str(directory/'summary.json'):summary_sha,**{str(directory/n):h for n,h in report['artifacts'].items()}}
    def check():
        if any(digest(Path(p))!=h for p,h in pins.items()):raise ValueError('endpoint artifact/source pin mismatch')
    check()
    binding=load_plan(directory/'bindings.json')
    source=Path(binding['input_path']).resolve();parent=Path(binding['run_directory']).resolve()
    if not parent.is_relative_to(scope):raise ValueError('endpoint parent outside research scope')
    current={str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}
    if binding['code']!=current:raise ValueError('endpoint computational inventory mismatch')
    for p,h in {**current,str(source):binding['input_sha256']}.items():
        if p in pins and pins[p]!=h:raise ValueError('conflicting endpoint source pins')
        pins[p]=h
    check()
    payload=load_plan(directory/'input.json')
    if payload!=load_plan(source):raise ValueError('saved/original endpoint input mismatch')
    if set(payload)!={'evaluation_at','joint','risk10'}:raise ValueError('exact endpoint schema required')
    for name in ['joint','risk10']:
        value=payload[name]
        if not isinstance(value,dict) or set(value)!={'target_id','outcomes'} or not isinstance(value['outcomes'],list) or any(
            not isinstance(r,dict) or set(r)!={'sample_id','target','label_available_at'} for r in value['outcomes']):
            raise ValueError('exact endpoint records required')
    validation=replay(parent,binding['run_summary_sha256'])
    plan=load_plan(parent/'input.json')
    frame=lambda name:pd.DataFrame(payload[name]['outcomes'],columns=['sample_id','target','label_available_at'])
    rows,metrics=evaluate(pd.read_parquet(parent/'candidate_ledger.parquet'),pd.DataFrame(plan['samples']),
        frame('joint'),frame('risk10'),joint_target_id=payload['joint']['target_id'],risk_target_id=payload['risk10']['target_id'],
        evaluation_at=payload['evaluation_at'],calendar=plan['calendar'])
    pd.testing.assert_frame_equal(pd.read_parquet(directory/'evaluated_endpoints.parquet'),rows,check_exact=True)
    if load_plan(directory/'metrics.json')!=metrics:raise ValueError('endpoint metrics reconstruction mismatch')
    expected=dict(rows=len(rows),source_validation=validation,evidence_mode=plan['evidence_mode'],
                  models_refit=0,policy_reselected=False,formal_G3_passed=False)
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('endpoint summary claims mismatch')
    verify(parent,binding['run_summary_sha256']);check()
    return dict(rows=len(rows),endpoints_and_metrics_recomputed=True,models_refit=0,
                formal_G3_passed=False)
