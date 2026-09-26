"""Persist and reconstruct a pinned, fixed-count ranking comparison."""
from pathlib import Path
import uuid
import pandas as pd
from .joint_binary_scope import compare_ranking
from .runtime import atomic_json, digest, load_plan, now
from .comparison_parent_binding import audit as audit_parents, preflight

ARTIFACTS={'input.json','inputs.json','evaluated_rankings.parquet','metrics.json'}


def _compute(plan):
    required={'schema_version','joint','binary','target_id','evaluation_at','k','outcomes'}
    if plan.get('schema_version')=='2':required.add('parent_budget_references')
    if set(plan)!=required or plan['schema_version'] not in ['1','2'] or plan['target_id']!='P.joint.v4':
        raise ValueError('exact ranking comparison input required')
    for name in ['joint','binary']:
        if not isinstance(plan[name],dict) or set(plan[name])!={'directory','summary_sha256'}:
            raise ValueError('exact pinned model reference required')
    if not isinstance(plan['outcomes'],list) or any(not isinstance(r,dict) or set(r)!={'sample_id','target','label_available_at'} for r in plan['outcomes']):
        raise ValueError('exact joint outcome records required')
    rows,metrics=compare_ranking(plan['joint']['directory'],plan['joint']['summary_sha256'],
        plan['binary']['directory'],plan['binary']['summary_sha256'],
        pd.DataFrame(plan['outcomes'],columns=['sample_id','target','label_available_at']),
        evaluation_at=plan['evaluation_at'],k=plan['k'])
    # Preserve an explicit nullable logical type through parquet even when a
    # particular snapshot happens to contain only resolved outcomes.
    rows['evaluation_target']=rows.evaluation_target.astype('boolean')
    return rows,metrics


def _code():
    return {str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}


def build(root,input_path,input_sha):
    source=Path(input_path).resolve()
    if digest(source)!=input_sha:raise ValueError('comparison input pin mismatch')
    pins={str(source):input_sha,**_code()}
    plan=load_plan(source)
    parents=audit_parents(root,plan)
    rows,metrics=_compute(plan)
    if audit_parents(root,plan)!=parents:raise ValueError('comparison parent binding changed')
    if any(digest(Path(p))!=h for p,h in pins.items()):raise ValueError('comparison source changed')
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('joint-binary-comparison-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'input.json',plan);atomic_json(out/'inputs.json',pins)
    rows.to_parquet(out/'evaluated_rankings.parquet',index=False);atomic_json(out/'metrics.json',metrics)
    report=dict(directory=str(out),at=now(),input_path=str(source),input_sha256=input_sha,
        rows=len(rows),models_refit=0,calibrators_refit=0,
        parent_budget_binding=parents,
        comparison_kind='fixed_count_ranking_diagnostic',formal_promotion_authorized=False,
        prediction_algorithm_independently_replayed=False,search_budget_registered=False,
        artifacts={n:digest(out/n) for n in sorted(ARTIFACTS)})
    if any(digest(Path(p))!=h for p,h in pins.items()):raise ValueError('comparison source changed')
    atomic_json(out/'summary.json',report)
    return report


def verify(root,directory,summary_sha):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(Path(root).resolve()/'output/experiments/s20_safe_v4/sources'):
        raise ValueError('comparison outside research scope')
    if digest(directory/'summary.json')!=summary_sha:raise ValueError('comparison summary pin mismatch')
    report=load_plan(directory/'summary.json')
    if report['directory']!=str(directory) or set(report['artifacts'])!=ARTIFACTS:
        raise ValueError('comparison artifact contract mismatch')
    pins={str(directory/'summary.json'):summary_sha,**{str(directory/n):h for n,h in report['artifacts'].items()}}
    def check():
        if any(digest(Path(p))!=h for p,h in pins.items()):raise ValueError('comparison artifact/source pin mismatch')
    check()
    sources=load_plan(directory/'inputs.json')
    source=Path(report['input_path']).resolve()
    if sources.get(str(source))!=report['input_sha256']:
        raise ValueError('comparison original input binding mismatch')
    if {p:h for p,h in sources.items() if Path(p).suffix=='.py'}!=_code():
        raise ValueError('comparison code inventory mismatch')
    for p,h in sources.items():
        if p in pins and pins[p]!=h:raise ValueError('conflicting comparison pins')
        pins[p]=h
    check()
    if load_plan(source)!=load_plan(directory/'input.json'):
        raise ValueError('comparison saved/original input mismatch')
    rows,metrics=_compute(load_plan(directory/'input.json'))
    parents=audit_parents(root,load_plan(directory/'input.json'))
    if report.get('parent_budget_binding')!=parents:raise ValueError('comparison parent budget evidence mismatch')
    pd.testing.assert_frame_equal(pd.read_parquet(directory/'evaluated_rankings.parquet'),rows,check_exact=True)
    if load_plan(directory/'metrics.json')!=metrics:raise ValueError('comparison metrics reconstruction mismatch')
    expected=dict(rows=len(rows),models_refit=0,calibrators_refit=0,
        comparison_kind='fixed_count_ranking_diagnostic',formal_promotion_authorized=False,
        prediction_algorithm_independently_replayed=False,search_budget_registered=False)
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('comparison summary claims mismatch')
    check()
    return dict(rows=len(rows),rankings_and_metrics_recomputed=True,models_refit=0,
                formal_promotion_authorized=False)


def run_registered(root,input_path,input_sha,budget,trial_id,attempt_id):
    """Reserve two fixed ranking evaluations; parent fits are not recharged.

    Version 2 inputs additionally bind both parent training reservations.
    Neither version proves global accounting or that k was chosen before
    researchers saw historical outcomes.
    """
    expected=dict(model_fits=0,underlying_fits=0,calibrator_fits=0,policy_evaluations=2)
    trial=next((t for t in budget.contract['trials'] if t['trial_id']==trial_id),None)
    if trial is None or trial['costs']!=expected or digest(Path(input_path))!=input_sha:
        raise ValueError('registered comparison cost/input mismatch')
    inventory=preflight(root,load_plan(input_path),budget)
    ticket=budget.reserve(trial_id,attempt_id,input_sha)
    if not ticket['newly_reserved']:
        status=budget.status()
        if ticket['state']!='SUCCEEDED_DIAGNOSTIC':
            return dict(executed=False,reusable=False,budget=status)
        attempt=next(a for a in status['attempts'] if (a['trial_id'],a['attempt_id'])==(trial_id,attempt_id))
        artifact=attempt['result']
        verify(root,artifact['directory'],artifact['summary_sha256'])
        report=load_plan(Path(artifact['directory'])/'summary.json')
        if report['input_sha256']!=input_sha or Path(report['input_path']).resolve()!=Path(input_path).resolve():
            raise ValueError('registered comparison input binding mismatch')
        return dict(executed=False,reusable=True,artifact=artifact,budget=status,
                    parent_budget_inventory=inventory,
                    local_comparison_reserved=True,parent_training_budget_verified=report['parent_budget_binding']['parent_training_budget_verified'])
    try:
        report=build(root,input_path,input_sha)
        artifact=dict(directory=report['directory'],summary_sha256=digest(Path(report['directory'])/'summary.json'))
        verify(root,artifact['directory'],artifact['summary_sha256'])
        budget.finish(trial_id,attempt_id,'SUCCEEDED_DIAGNOSTIC',artifact)
    except Exception as exc:
        budget.finish(trial_id,attempt_id,'FAILED',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(executed=True,reusable=False,artifact=artifact,budget=budget.status(),
                parent_budget_inventory=inventory,
                local_comparison_reserved=True,parent_training_budget_verified=report['parent_budget_binding']['parent_training_budget_verified'])
