"""Sequential synthetic search jobs using the single registered global budget."""
from pathlib import Path
import sqlite3
import uuid
import os

from .search_registration import verify, specification
from .trial_budget import Budget, verify_attempt
from .bounded_joint import run as run_joint
from .process_runner import Limits
from .runtime import atomic_json,load_plan,now,digest
from .label_availability import _instant


def writable_path(path):
    value=str(Path(path).resolve())
    return Path('\\\\?\\'+value) if os.name=='nt' and not value.startswith('\\\\?\\') else Path(value)


def run(root,directory,plan_sha,*,max_new_jobs=1):
    if type(max_new_jobs)is not int or not 1<=max_new_jobs<=72:raise ValueError('bounded new-job limit required')
    root=Path(root).resolve();directory=Path(directory).resolve()
    registration=verify(root,directory,plan_sha)
    spec=specification(root,load_plan_path(directory),plan_sha)
    plan=load_plan(spec['plan_path'])
    manifests={c['candidate_id']:load_plan(c['manifest_path']) for c in plan['candidates']}
    if any(m['evidence_mode']!='synthetic' for m in manifests.values()):
        raise ValueError('shared runner is synthetic-only until formal upstream acceptance')
    lock=sqlite3.connect(directory/'execution_lock.sqlite',timeout=0)
    try:
        try:lock.execute('BEGIN IMMEDIATE')
        except sqlite3.OperationalError as exc:raise ValueError('registered search execution already owned') from exc
        budget=Budget(registration['shared_budget_path'],spec['budget_contract'])
        completed=[];executed=0
        def checkpoint(state,error=None):
            atomic_json(writable_path(directory/'execution_checkpoint.json'),dict(at=now(),status=state,
                verified_jobs=completed,executed_this_call=executed,budget=budget.status(),error=error,
                plan_sha256=plan_sha,formal_training_authorized=False))
        checkpoint('RUNNING_SYNTHETIC')
        try:
            for job in registration['mapping']:
                verify(root,directory,plan_sha)
                limits=Limits(**manifests[job['candidate_id']]['limits'])
                prior=next((a for a in budget.status()['attempts'] if
                    (a['trial_id'],a['attempt_id'])==(job['shared_trial_id'],job['attempt_id'])),None)
                if prior is None and executed>=max_new_jobs:break
                result=run_joint(root,job['input_path'],job['input_sha256'],budget,
                    job['shared_trial_id'],job['attempt_id'],limits)
                if not result.get('executed') and not result.get('reusable'):
                    raise ValueError('failed or reserved attempt requires reconciliation; no automatic retry')
                executed+=int(result['executed']);artifact=result['artifact']
                evidence=verify_attempt(root,budget.path,budget.sha,job['shared_trial_id'],job['attempt_id'],
                    job['input_path'],job['input_sha256'],artifact,limits)
                # This only proves local ordering for these owned launches.
                if _instant(evidence['reserved_at'])<_instant(registration['registered_at']):
                    raise ValueError('attempt predates local search registration')
                completed.append(dict(job,artifact=artifact,executed_this_call=result['executed']))
                checkpoint('RUNNING_SYNTHETIC')
            verify(root,directory,plan_sha)
        except Exception as exc:
            checkpoint('FAILED',dict(type=type(exc).__name__,message=str(exc)));raise
        status='COMPLETED_SYNTHETIC' if len(completed)==len(registration['mapping']) else 'PARTIAL_SYNTHETIC'
        checkpoint(status)
        report=dict(at=now(),directory=str(directory),plan_sha256=plan_sha,status=status,
            jobs=completed,registered_jobs=len(registration['mapping']),executed_this_call=executed,
            budget=budget.status(),shared_budget_used=True,formal_H04_accepted=False,
            formal_training_authorized=False,finalists_selected=False)
        receipt=writable_path(directory/('execution-receipt-'+uuid.uuid4().hex+'.json'));atomic_json(receipt,report)
        return dict(report,receipt_path=str(receipt))
    finally:
        lock.rollback();lock.close()


def load_plan_path(directory):
    import json
    with sqlite3.connect((Path(directory)/'registration.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        return json.loads(db.execute('SELECT payload FROM registration WHERE id=1').fetchone()[0])['plan_path']


def verify_receipt(root,path,sha):
    """Verify a historical partial/completed receipt without mutating or fitting."""
    path=Path(path).resolve()
    if digest(path)!=sha:raise ValueError('search execution receipt pin mismatch')
    report=load_plan(path)
    expected={'at','directory','plan_sha256','status','jobs','registered_jobs','executed_this_call',
              'budget','shared_budget_used','formal_H04_accepted','formal_training_authorized','finalists_selected'}
    if set(report)!=expected:raise ValueError('exact search execution receipt required')
    directory=Path(report['directory']).resolve()
    # Windows extended-path and ordinary spellings name the same parent.
    if not path.parent.samefile(directory):raise ValueError('search execution receipt scope mismatch')
    registration=verify(root,directory,report['plan_sha256'])
    mapping=registration['mapping'];jobs=report['jobs']
    if not isinstance(jobs,list) or not 0<len(jobs)<=len(mapping):raise ValueError('bounded nonempty receipt jobs required')
    if type(report['registered_jobs'])is not int or report['registered_jobs']!=len(mapping):
        raise ValueError('registered job denominator mismatch')
    status='COMPLETED_SYNTHETIC' if len(jobs)==len(mapping) else 'PARTIAL_SYNTHETIC'
    if report['status']!=status:raise ValueError('search completion status mismatch')
    if report['shared_budget_used'] is not True or any(report[k] is not False for k in ['formal_H04_accepted','formal_training_authorized','finalists_selected']):
        raise ValueError('unsupported search execution claim')
    snapshot=report['budget'];current=registration['budget']
    if set(snapshot)!={'budget_sha256','reserved_counts','attempts','failed_attempts_refunded','formal_training_authorized'}:
        raise ValueError('exact receipt budget snapshot required')
    if snapshot['budget_sha256']!=registration['shared_budget_contract_sha256'] or snapshot['failed_attempts_refunded'] is not False or snapshot['formal_training_authorized'] is not False:
        raise ValueError('receipt budget identity mismatch')
    from .trial_budget import COUNTERS,canonical
    actual={ (t['trial_id'],a['attempt_id']):a for t in current['trials'] for a in t['attempts'] }
    seen=set();sums={k:0 for k in COUNTERS}
    for attempt in snapshot['attempts']:
        if set(attempt)!={'trial_id','attempt_id','costs','state','result'}:raise ValueError('exact receipt attempt required')
        key=(attempt['trial_id'],attempt['attempt_id'])
        if key in seen or key not in actual:raise ValueError('receipt attempt identity mismatch')
        seen.add(key);observed=actual[key]
        if attempt['state']!='SUCCEEDED_DIAGNOSTIC' or observed['recorded_state']!=attempt['state'] or canonical(observed['recorded_result'])!=canonical(attempt['result']) or canonical(observed['costs'])!=canonical(attempt['costs']):
            raise ValueError('receipt attempt differs from shared budget')
        for k in COUNTERS:sums[k]+=attempt['costs'][k]
    if canonical(sums)!=canonical(snapshot['reserved_counts']):raise ValueError('receipt budget count mismatch')
    plan=load_plan(load_plan_path(directory))
    manifests={c['candidate_id']:load_plan(c['manifest_path']) for c in plan['candidates']}
    if any(m['evidence_mode']!='synthetic' for m in manifests.values()):
        raise ValueError('synthetic execution receipt requires synthetic registered inputs')
    executed=0;keys=set()
    for job,registered in zip(jobs,mapping):
        if set(job)!=set(registered)|{'artifact','executed_this_call'} or any(job[k]!=v for k,v in registered.items()):
            raise ValueError('receipt registered order/identity mismatch')
        if type(job['executed_this_call'])is not bool:raise ValueError('boolean execution marker required')
        executed+=job['executed_this_call'];keys.add((job['shared_trial_id'],job['attempt_id']))
        limits=Limits(**manifests[job['candidate_id']]['limits'])
        evidence=verify_attempt(root,registration['shared_budget_path'],registration['shared_budget_contract_sha256'],
            job['shared_trial_id'],job['attempt_id'],job['input_path'],job['input_sha256'],job['artifact'],limits)
        if not _instant(registration['registered_at'])<=_instant(evidence['reserved_at'])<=_instant(report['at']):
            raise ValueError('receipt registration/reservation chronology mismatch')
    if keys!=seen:raise ValueError('receipt job/attempt coverage mismatch')
    if type(report['executed_this_call'])is not int or report['executed_this_call']!=executed:
        raise ValueError('receipt execution count mismatch')
    verify(root,directory,report['plan_sha256'])
    if digest(path)!=sha:raise ValueError('search receipt changed during verification')
    return dict(search_execution_receipt_verified=True,status=status,jobs=len(jobs),models_refit=0,
        executed_this_call_independently_proven=False,formal_H04_accepted=False)
