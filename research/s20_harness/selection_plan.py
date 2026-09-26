"""Pinned candidate-universe preflight; validation is not proof of preregistration."""
from pathlib import Path

from .runtime import digest, load_plan
from .multifold_run import comparison_variation,check_seed_grid
from .trial_budget import COUNTERS
from .label_availability import _instant
from .process_runner import Limits


def inspect(root,path,sha):
    root=Path(root).resolve();path=Path(path).resolve()
    pins={path:sha}
    if digest(path)!=sha:raise ValueError('selection plan pin mismatch')
    plan=load_plan(path)
    required={'schema_version','baseline_id','candidates','limits','finalist_cap','selection_rule'}|({'feature_ablation'} if 'feature_ablation' in plan else set())
    if set(plan)!=required or plan['schema_version']!='selection-1':
        raise ValueError('exact selection plan required')
    if plan['selection_rule'] not in ['matched_cell_robust_dominance_no_automatic_tiebreak',
                                      'all_registered_nondominated_no_automatic_tiebreak']:
        raise ValueError('explicit supported selection rule required')
    if type(plan['finalist_cap'])is not int or not 1<=plan['finalist_cap']<=3:
        raise ValueError('finalist cap must be between one and three')
    candidates=plan['candidates']
    if not isinstance(candidates,list) or not 2<=len(candidates)<=12:
        raise ValueError('two to twelve registered candidates required')
    limits=plan['limits']
    if set(limits)!=set(COUNTERS) or any(type(v)is not int or v<0 for v in limits.values()) or limits['model_fits']>72:
        raise ValueError('bounded total pipeline budget required')
    ids=set();campaigns=set();input_hashes=set();totals={c:0 for c in COUNTERS};scope_by_fold={};grid=None
    allocations={c:0 for c in COUNTERS};evidence_modes=set();ablation_cells={}
    for candidate in candidates:
        if set(candidate)!={'candidate_id','manifest_path','manifest_sha256'}:raise ValueError('exact candidate reference required')
        cid=candidate['candidate_id']
        if not isinstance(cid,str) or not cid.strip() or cid in ids:raise ValueError('unique candidate ID required')
        ids.add(cid);manifest_path=Path(candidate['manifest_path']).resolve();pin=candidate['manifest_sha256']
        if digest(manifest_path)!=pin or pin in campaigns:raise ValueError('candidate manifest pin/duplicate mismatch')
        campaigns.add(pin);pins[manifest_path]=pin;manifest=load_plan(manifest_path)
        comparison_variation(manifest)
        if manifest['schema_version']!='7':raise ValueError('explicit seed campaign required')
        if manifest['evidence_mode'] not in ['synthetic','supplied_reference']:raise ValueError('diagnostic evidence mode required')
        evidence_modes.add(manifest['evidence_mode'])
        Limits(**manifest['limits'])
        budget=manifest['budget']
        if set(budget)!={'budget_id','limits','trials'} or not isinstance(budget['budget_id'],str) or not budget['budget_id']:
            raise ValueError('exact named candidate budget required')
        allocation=budget['limits']
        if set(allocation)!=set(COUNTERS) or any(type(v)is not int or v<0 for v in allocation.values()):
            raise ValueError('integer candidate budget allocation required')
        for counter in COUNTERS:allocations[counter]+=allocation[counter]
        current=(manifest['seed_grid'],manifest['fold_grid'])
        if len(current[0])!=2 or len(current[1])!=3:raise ValueError('initial search requires two seeds and three folds')
        if grid is not None and grid!=current:raise ValueError('shared search grid required')
        grid=current;inputs=[];job_ids=set();trial_ids=set();intervals={};candidate_costs={c:0 for c in COUNTERS}
        for job in manifest['jobs']:
            if set(job)!={'job_id','fold_id','input_path','input_sha256','trial_id','attempt_id','pipeline_kind'}:
                raise ValueError('exact candidate job required')
            if any(not isinstance(job[k],str) or not job[k].strip() for k in ['job_id','fold_id','trial_id','attempt_id']):
                raise ValueError('named candidate job identities required')
            if job['job_id'] in job_ids or job['trial_id'] in trial_ids:raise ValueError('unique candidate jobs/trials required')
            job_ids.add(job['job_id']);trial_ids.add(job['trial_id'])
            input_path=Path(job['input_path']).resolve();input_sha=job['input_sha256']
            if digest(input_path)!=input_sha:raise ValueError('selection candidate input pin mismatch')
            if input_sha in input_hashes:raise ValueError('duplicate selectable pipeline input')
            input_hashes.add(input_sha);pins[input_path]=input_sha;payload=load_plan(input_path);inputs.append(payload)
            if payload.get('evidence_mode')!=manifest['evidence_mode']:raise ValueError('mixed candidate evidence')
            bounds=payload['boundaries'][4]
            interval=(_instant(bounds['start_at']),_instant(bounds['end_at']))
            if interval[0]>=interval[1]:raise ValueError('reversed candidate outer fold')
            if job['fold_id'] in intervals and intervals[job['fold_id']]!=interval:
                raise ValueError('inconsistent candidate outer fold')
            intervals[job['fold_id']]=interval
            if job.get('pipeline_kind')!='joint' or payload.get('target_id')!='P.joint.v4':
                raise ValueError('joint comparison endpoints required')
            from .joint_run import pipeline_costs
            costs=pipeline_costs(payload)
            matches=[t for t in manifest['budget']['trials'] if t['trial_id']==job['trial_id']]
            if len(matches)!=1 or matches[0]['input_sha256']!=input_sha or matches[0]['costs']!=costs:
                raise ValueError('selection trial cost/input binding mismatch')
            trial=matches[0]
            if set(trial)!={'trial_id','input_sha256','costs','max_attempts'} or type(trial['max_attempts'])is not int or not 1<=trial['max_attempts']<=3:
                raise ValueError('bounded exact candidate trial required')
            if any(type(v)is not int for v in trial['costs'].values()):raise ValueError('integer trial costs required')
            for counter in COUNTERS:
                totals[counter]+=costs[counter];candidate_costs[counter]+=costs[counter]
            # Include the same-seed input here; only family/loss weights vary.
            from .trial_budget import canonical
            import hashlib
            excluded=['features','feature_contract'] if 'feature_ablation' in plan else ['model_family','class_weights']
            scope=hashlib.sha256(canonical({k:v for k,v in payload.items() if k not in excluded}).encode()).hexdigest()
            key=(job['fold_id'],payload['random_seed'])
            ablation_cells.setdefault(key,{})[cid]=payload
            if key in scope_by_fold and scope_by_fold[key]!=scope:raise ValueError('selection comparison input scope mismatch')
            scope_by_fold[key]=scope
        if len(trial_ids)!=len(manifest['budget']['trials']):raise ValueError('unaccounted campaign trials')
        check_seed_grid(manifest,inputs)
        ordered=sorted(intervals.values())
        if any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):raise ValueError('candidate outer folds overlap')
        if any(candidate_costs[c]>allocation[c] for c in COUNTERS):raise ValueError('candidate allocation below initial costs')
    if plan['baseline_id'] not in ids:raise ValueError('registered baseline required')
    if 'feature_ablation' in plan:
        from .feature_group_ablation import validate_registered_cells
        validate_registered_cells(plan['feature_ablation'],ablation_cells,plan['baseline_id'])
    if any(totals[c]>limits[c] for c in COUNTERS):raise ValueError('total search budget exceeded')
    if len(evidence_modes)!=1:raise ValueError('mixed search evidence modes')
    if any(allocations[c]>limits[c] for c in COUNTERS):raise ValueError('candidate allocations exceed shared search budget')
    if any(digest(p)!=h for p,h in pins.items()):raise ValueError('selection inputs changed during preflight')
    return dict(plan_sha256=sha,candidate_ids=[c['candidate_id'] for c in candidates],baseline_id=plan['baseline_id'],
        planned_initial_costs=totals,allocated_campaign_limits=allocations,finalist_cap=plan['finalist_cap'],input_contracts_validated=True,
        preregistration_before_training_proven=False,global_budget_enforcement_connected=False,
        formal_H04_accepted=False,training_authorized=False,finalists_selected=False,
        parent_anchor_costs_verified=False,models_refit=0)
