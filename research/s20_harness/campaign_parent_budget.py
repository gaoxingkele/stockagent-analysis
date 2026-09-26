"""Account referenced parent ledgers once before releasing child jobs."""
from pathlib import Path
from .anchor_parent_receipts import verify_budget_binding
from .trial_budget import verify_attempt,COUNTERS
from .process_runner import Limits


def inspect(root,plans,child_limits,child_budget_path):
    ledgers={};attempts=set();unbound=0
    for plan in plans:
        if 'anchor_context' not in plan: continue
        bindings=plan.get('anchor_parent_bindings',[])
        bound={b['model_id'] for b in bindings if 'budget_reference' in b}
        unbound+=len(set(plan['anchor_context']['models'])-bound)
        for binding in bindings:
            if 'budget_reference' not in binding: continue
            verify_budget_binding(root,binding)
            ref=binding['budget_reference'];path=Path(ref['path']).resolve()
            result=verify_attempt(root,path,ref['contract_sha256'],ref['trial_id'],ref['attempt_id'],
                ref['input_path'],ref['input_sha256'],ref['artifact'],Limits(**ref['limits']))
            key=str(path)
            if key in ledgers and ledgers[key]['contract_sha256']!=ref['contract_sha256']:
                raise ValueError('conflicting referenced parent budget contracts')
            ledgers[key]=dict(contract_sha256=ref['contract_sha256'],reserved_counts=result['reserved_counts_at_review'])
            attempts.add((key,ref['trial_id'],ref['attempt_id']))
    external={k:v for k,v in ledgers.items() if Path(k)!=Path(child_budget_path).resolve()}
    reserved={c:sum(v['reserved_counts'][c] for v in external.values()) for c in COUNTERS}
    combined=child_limits['model_fits']+reserved['model_fits']
    if combined>72: raise ValueError('referenced parent and child campaign exceeds 72 model fits')
    return dict(referenced_ledgers=ledgers,unique_parent_attempts=len(attempts),
        external_reserved_counts=reserved,child_model_fit_capacity=child_limits['model_fits'],
        combined_model_fit_capacity=combined,unbound_anchor_models=unbound,
        all_referenced_anchors_budget_bound=unbound==0,
        unrelated_research_ledgers_inventoried=False,formal_training_authorized=False)
