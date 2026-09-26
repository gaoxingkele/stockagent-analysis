"""Read-only transaction snapshot of all registered trials, including failures."""
from pathlib import Path
import hashlib
import json
import re
import sqlite3

from .trial_budget import COUNTERS, canonical


def inspect(root,path,contract_sha):
    path=Path(path).resolve();root=Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('existing workspace budget required')
    if not re.fullmatch('[0-9a-f]{64}',contract_sha or ''):
        raise ValueError('budget contract pin required')
    # Never construct Budget here: its constructor creates tables/triggers.
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=30) as db:
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        contracts=db.execute('SELECT id,payload FROM budget_contract').fetchall()
        records=db.execute('SELECT trial_id,attempt_id,costs,state,reserved_at,result FROM attempts ORDER BY trial_id,attempt_id').fetchall()
    if len(contracts)!=1 or contracts[0][0]!=1:
        raise ValueError('exact stored budget contract required')
    payload=contracts[0][1]
    if hashlib.sha256(payload.encode()).hexdigest()!=contract_sha:
        raise ValueError('budget contract pin mismatch')
    contract=json.loads(payload)
    if canonical(contract)!=payload or set(contract)!={'budget_id','limits','trials'}:
        raise ValueError('canonical exact budget contract required')
    def counts(value):
        if not isinstance(value,dict) or set(value)!=set(COUNTERS) or any(type(v)is not int or v<0 for v in value.values()):
            raise ValueError('nonnegative integer budget counters required')
    counts(contract['limits'])
    if contract['limits']['model_fits']>72:raise ValueError('model fit budget exceeds frozen cap')
    if not isinstance(contract['budget_id'],str) or not contract['budget_id'] or not isinstance(contract['trials'],list) or not 1<=len(contract['trials'])<=1000:
        raise ValueError('bounded named budget required')
    trials={}
    for trial in contract['trials']:
        if set(trial)!={'trial_id','input_sha256','costs','max_attempts'}:
            raise ValueError('exact registered trial required')
        tid=trial['trial_id']
        if not isinstance(tid,str) or not tid or tid in trials:raise ValueError('unique trial identity required')
        if not re.fullmatch('[0-9a-f]{64}',trial['input_sha256'] or ''):raise ValueError('trial input pin required')
        counts(trial['costs'])
        if type(trial['max_attempts'])is not int or not 1<=trial['max_attempts']<=3:raise ValueError('bounded attempts required')
        trials[tid]=dict(trial,attempts=[])
    used={c:0 for c in COUNTERS};seen=set()
    states={'RESERVED':'RESERVED_LIVENESS_UNKNOWN','FAILED':'RECORDED_FAILED',
            'SUCCEEDED_DIAGNOSTIC':'RECORDED_SUCCESS_UNVERIFIED'}
    for tid,aid,cost,state,reserved,result in records:
        if tid not in trials or not isinstance(aid,str) or not aid or (tid,aid) in seen:
            raise ValueError('unregistered or duplicate attempt')
        seen.add((tid,aid));charge=json.loads(cost);counts(charge)
        if charge!=trials[tid]['costs']:raise ValueError('attempt charge mismatch')
        if state not in states:raise ValueError('unknown attempt state')
        if not isinstance(reserved,str) or not reserved:raise ValueError('reservation time required')
        if (state=='RESERVED')!=(result is None):raise ValueError('attempt state/result mismatch')
        evidence=json.loads(result) if result is not None else None
        if evidence is not None and not isinstance(evidence,dict):raise ValueError('object attempt result required')
        for c in COUNTERS:used[c]+=charge[c]
        trials[tid]['attempts'].append(dict(attempt_id=aid,recorded_state=state,observation=states[state],
            costs=charge,reserved_at=reserved,recorded_result=evidence,
            result_artifacts_verified=False,process_liveness_checked=False))
    for trial in trials.values():
        if len(trial['attempts'])>trial['max_attempts']:raise ValueError('attempt cap exceeded')
        trial['observation']='NOT_RESERVED' if not trial['attempts'] else 'ATTEMPTS_RECORDED'
    if any(used[c]>contract['limits'][c] for c in COUNTERS):raise ValueError('budget overspent')
    snapshot=dict(budget_id=contract['budget_id'],contract_sha256=contract_sha,
        limits=contract['limits'],reserved_counts=used,trials=list(trials.values()))
    return dict(snapshot,snapshot_sha256=hashlib.sha256(canonical(snapshot).encode()).hexdigest(),
        budget_path=str(path),snapshot_is_current_process_status=False,
        success_artifacts_verified=False,formal_training_authorized=False,
        failed_attempts_refunded=False,launch_authorized=False,
        database_modified=False,models_refit=0)
