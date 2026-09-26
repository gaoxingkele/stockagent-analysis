"""Pre-reserved policy-only diagnostic execution on an existing shared budget."""
import hashlib
import json
from pathlib import Path
import sqlite3
from . import joint_policy_run
from .runtime import digest, load_plan
from .trial_budget import canonical
from .attempt_inventory import inspect


def run_existing(root,run_directory,run_sha,input_path,input_sha,path,contract_sha,trial_id,attempt_id):
    """Never invent a budget from CLI flags; validate the existing contract first."""
    from .trial_budget import Budget
    path=Path(path).resolve()
    inspect(root,path,contract_sha)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        raw=db.execute('SELECT payload FROM budget_contract WHERE id=1').fetchone()[0]
    if hashlib.sha256(raw.encode()).hexdigest()!=contract_sha:
        raise ValueError('policy budget changed before opening')
    return run(root,run_directory,run_sha,input_path,input_sha,Budget(path,json.loads(raw)),trial_id,attempt_id)


def binding(run_directory,run_sha,input_path,input_sha):
    directory=Path(run_directory).resolve();source=Path(input_path).resolve()
    if digest(directory/'summary.json')!=run_sha or digest(source)!=input_sha:
        raise ValueError('policy source pin mismatch')
    payload=load_plan(source)
    policies=payload['registry']['policies']
    if not isinstance(policies,list) or not 2<=len(policies)<=6:
        raise ValueError('two to six registered policy evaluations required')
    identity=dict(run_directory=str(directory),run_sha256=run_sha,input_path=str(source),input_sha256=input_sha,
                  code_pins={str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')})
    return identity,hashlib.sha256(canonical(identity).encode()).hexdigest(),dict(
        model_fits=0,underlying_fits=0,calibrator_fits=0,policy_evaluations=len(policies))


def run(root,run_directory,run_sha,input_path,input_sha,budget,trial_id,attempt_id):
    if not budget.path.is_relative_to(Path(root).resolve()):raise ValueError('workspace budget required')
    identity,sha,costs=binding(run_directory,run_sha,input_path,input_sha)
    trial=next((t for t in budget.contract['trials'] if t['trial_id']==trial_id),None)
    if trial is None or canonical(trial['costs'])!=canonical(costs) or trial['input_sha256']!=sha:
        raise ValueError('registered full binding and exact policy costs required')
    ticket=budget.reserve(trial_id,attempt_id,sha)
    if not ticket['newly_reserved']:
        if ticket['state']=='SUCCEEDED_DIAGNOSTIC':
            checked=verify(root,budget.path,budget.sha,trial_id,attempt_id)
            return dict(executed=False,verification=checked,budget=budget.status())
        raise ValueError('existing failed/reserved policy attempt requires reconciliation')
    try:
        report=joint_policy_run.build(root,run_directory,run_sha,input_path,input_sha)
        current,current_sha,_=binding(run_directory,run_sha,input_path,input_sha)
        if current_sha!=sha:raise ValueError('policy binding changed during execution')
        result=dict(binding=identity,binding_sha256=sha,directory=report['directory'],
                    summary_sha256=digest(Path(report['directory'])/'summary.json'))
        budget.finish(trial_id,attempt_id,'SUCCEEDED_DIAGNOSTIC',result)
    except Exception as exc:
        budget.finish(trial_id,attempt_id,'FAILED',dict(type=type(exc).__name__,message=str(exc)))
        raise
    return dict(executed=True,verification=verify(root,budget.path,budget.sha,trial_id,attempt_id),budget=budget.status())


def verify(root,path,contract_sha,trial_id,attempt_id):
    path=Path(path).resolve();inspect(root,path,contract_sha)
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        contract=json.loads(db.execute('SELECT payload FROM budget_contract WHERE id=1').fetchone()[0])
        row=db.execute('SELECT costs,state,result FROM attempts WHERE trial_id=? AND attempt_id=?',(trial_id,attempt_id)).fetchone()
    if row is None or row[1]!='SUCCEEDED_DIAGNOSTIC':raise ValueError('completed policy budget attempt required')
    result=json.loads(row[2])
    if set(result)!={'binding','binding_sha256','directory','summary_sha256'}:raise ValueError('exact policy budget result required')
    source=result['binding']
    identity,sha,costs=binding(source['run_directory'],source['run_sha256'],source['input_path'],source['input_sha256'])
    trial=next(t for t in contract['trials'] if t['trial_id']==trial_id)
    if canonical(identity)!=canonical(source) or sha!=result['binding_sha256'] or trial['input_sha256']!=sha or canonical(json.loads(row[0]))!=canonical(costs):
        raise ValueError('policy budget binding/charge mismatch')
    checked=joint_policy_run.verify(root,result['directory'],result['summary_sha256'])
    report=load_plan(Path(result['directory'])/'summary.json')
    if any(report[k]!=source[k] for k in ['run_directory','run_sha256','input_path','input_sha256']) or report['policy_evaluations']!=costs['policy_evaluations']:
        raise ValueError('policy budget output binding mismatch')
    inspect(root,path,contract_sha)
    return dict(policy_budget_and_result_verified=True,charged_costs=costs,
                directory=result['directory'],summary_sha256=result['summary_sha256'],
                saved_comparison=checked,global_search_budget_coverage_proven=False,
                historical_preregistration_proven=False,formal_H05_accepted=False)
