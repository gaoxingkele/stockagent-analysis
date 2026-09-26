"""Bind both comparison models to recorded prelaunch training reservations."""
from pathlib import Path
from .process_runner import Limits
from .trial_budget import verify_attempt
from .trial_budget import COUNTERS


def audit(root,plan):
    if plan.get('schema_version')!='2':
        return dict(parent_training_budget_verified=False,parents={})
    refs=plan.get('parent_budget_references')
    fields={'path','contract_sha256','trial_id','attempt_id','input_path','input_sha256','artifact','limits'}
    if not isinstance(refs,dict) or set(refs)!={'joint','binary'}:
        raise ValueError('both comparison parent budgets required')
    results={};seen=set()
    for name,ref in refs.items():
        if not isinstance(ref,dict) or set(ref)!=fields:
            raise ValueError('exact comparison parent budget reference required')
        model=plan[name];artifact=ref['artifact']
        if Path(artifact['directory']).resolve()!=Path(model['directory']).resolve() or artifact['summary_sha256']!=model['summary_sha256']:
            raise ValueError('comparison parent budget points to another model')
        if artifact.get('pipeline_kind','baseline')!=('joint' if name=='joint' else 'baseline'):
            raise ValueError('comparison parent pipeline mismatch')
        identity=(str(Path(ref['path']).resolve()),ref['trial_id'],ref['attempt_id'])
        if identity in seen:raise ValueError('comparison parents share one training attempt')
        seen.add(identity)
        checked=verify_attempt(root,ref['path'],ref['contract_sha256'],ref['trial_id'],ref['attempt_id'],
            ref['input_path'],ref['input_sha256'],artifact,Limits(**ref['limits']))
        # Freeze only this attempt; later reservations in the ledger are valid
        # and must not invalidate a previously checked model binding.
        checked.pop('reserved_counts_at_review')
        results[name]=checked
    return dict(parent_training_budget_verified=True,parents=results,
                global_pipeline_budget_proven=False,historical_preregistration_proven=False)


def preflight(root,plan,child_budget):
    binding=audit(root,plan)
    ledgers={}
    for ref in plan.get('parent_budget_references',{}).values():
        checked=verify_attempt(root,ref['path'],ref['contract_sha256'],ref['trial_id'],ref['attempt_id'],
            ref['input_path'],ref['input_sha256'],ref['artifact'],Limits(**ref['limits']))
        path=str(Path(ref['path']).resolve())
        item=dict(contract_sha256=ref['contract_sha256'],reserved_counts=checked['reserved_counts_at_review'])
        if path in ledgers and ledgers[path]!=item:
            raise ValueError('parent ledger changed or conflicting contract during preflight')
        ledgers[path]=item
    child_path=str(child_budget.path.resolve())
    if child_path in ledgers and ledgers[child_path]['contract_sha256']!=child_budget.sha:
        raise ValueError('child/parent budget contract mismatch')
    external={p:v for p,v in ledgers.items() if p!=child_path}
    used={k:sum(v['reserved_counts'][k] for v in external.values()) for k in COUNTERS}
    total={k:used[k]+child_budget.contract['limits'][k] for k in COUNTERS}
    if total['model_fits']>72:
        raise ValueError('comparison parent and child capacity exceeds 72 model fits')
    return dict(referenced_ledgers=ledgers,external_reserved_counts=used,
        child_registered_capacity=child_budget.contract['limits'],combined_capacity=total,
        parent_training_budget_verified=binding['parent_training_budget_verified'],
        accounting_scope='referenced ledger snapshots plus full child capacity',
        atomic_cross_ledger_reservation=False,unrelated_ledgers_inventoried=False,
        global_pipeline_budget_proven=False)
