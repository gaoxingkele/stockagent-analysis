"""Registered candidate aggregation without finalist promotion."""
from pathlib import Path
import hashlib
import pandas as pd
from .search_evaluation import verify
from .registered_search_run import load_plan_path
from .runtime import load_plan
from .metrics import binary_bounds
from .seed_summary import aggregate
from .configuration_comparison import compare_cells
from .trial_budget import canonical
from .baseline_coverage import inventory


def inspect(root,directory,sha):
    directory=Path(directory).resolve();verify(root,directory,sha)
    evaluation=load_plan(directory/'summary.json');receipt=load_plan(evaluation['receipt_path'])
    selection=load_plan(load_plan_path(Path(receipt['directory'])))
    refs={r['fold_id']:r['sha256'] for r in evaluation['outcomes']}
    predictions=pd.read_parquet(directory/'evaluated_predictions.parquet');cells=[];baseline_cards=[];strata=[]
    for job in receipt['jobs']:
        plan=load_plan(Path(job['artifact']['directory'])/'input.json')
        baseline_cards.append(dict(job_id=job['shared_trial_id'],fold_id=job['fold_id'],target_id=plan['target_id'],
            model_family=plan.get('model_family','multinomial'),evidence_mode=plan['evidence_mode'],
            policy_mode=plan['policy'].get('mode'),
            policy_sha256=hashlib.sha256(canonical(plan['policy']).encode()).hexdigest()))
        table=predictions.loc[predictions.job_id.eq(job['shared_trial_id'])];picked=table.loc[table.selected]
        controls=selection.get('feature_ablation',{}).get('control_strata')
        if controls is not None:
            from .feature_strata import summarize
            strata.append(dict(candidate_id=job['candidate_id'],job_id=job['shared_trial_id'],fold_id=job['fold_id'],
                random_seed=plan['random_seed'],**summarize(plan,table,controls)))
        excluded=['features','feature_contract','random_seed'] if 'feature_ablation' in selection else ['model_family','class_weights','random_seed']
        scope={k:v for k,v in plan.items() if k not in excluded}
        cells.append(dict(candidate_id=job['candidate_id'],job_id=job['shared_trial_id'],fold_id=job['fold_id'],
            random_seed=plan['random_seed'],input_sha256=job['input_sha256'],candidates=len(table),selected=len(picked),
            comparison_scope=hashlib.sha256(canonical(scope).encode()).hexdigest(),outcome_sha256=refs[job['fold_id']],
            daily_selected=[dict(signal_date=d,count=int(picked.signal_date.eq(d).sum())) for d in plan['calendar']],
            safe_profit=binary_bounds([None if pd.isna(v) else bool(v) for v in picked.safe_profit_target]),
            down5=binary_bounds([None if pd.isna(v) else bool(v) for v in picked.down5_target])))
    if sum(c['candidates'] for c in cells)!=len(predictions):raise ValueError('candidate denominator mismatch')
    grouped={c['candidate_id']:[r for r in cells if r['candidate_id']==c['candidate_id']] for c in selection['candidates']}
    summaries=[];comparisons=[]
    for candidate in selection['candidates']:
        cid=candidate['candidate_id'];manifest=load_plan(candidate['manifest_path'])
        summaries.append(dict(candidate_id=cid,per_seed=aggregate(grouped[cid],manifest['seed_grid'],manifest['fold_grid'])))
        if cid!=selection['baseline_id']:
            comparisons.append(dict(candidate_id=cid,baseline_id=selection['baseline_id'],
                **compare_cells(grouped[cid],grouped[selection['baseline_id']])))
    verify(root,directory,sha)
    ablation=None
    if 'feature_ablation' in selection:
        from .feature_group_ablation import contrasts
        ablation=contrasts(selection['feature_ablation'],grouped)
    return dict(evaluation_directory=str(directory),evaluation_summary_sha256=sha,
        registered_candidate_ids=list(grouped),baseline_id=selection['baseline_id'],cells=cells,
        candidates=summaries,baseline_comparisons=comparisons,registered_rule=selection['selection_rule'],
        baseline_coverage=inventory(baseline_cards),
        feature_ablation_contract=selection.get('feature_ablation'),conditional_feature_increment_proven=False,
        feature_ablation_comparisons=ablation,
        feature_control_strata=strata,
        registered_finalist_cap=selection['finalist_cap'],models_refit=0,formal_H04_accepted=False,
        finalists_selected=False,bounds_are_confidence_intervals=False,risk10_evaluated=False)
