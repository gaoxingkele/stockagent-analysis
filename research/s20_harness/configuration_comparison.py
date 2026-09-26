"""Read-only matched-cell comparison of verified seed summaries, not selection."""
from pathlib import Path
import hashlib

from .seed_summary import verify as verify_seed
from .runtime import load_plan
from .trial_budget import canonical


def compare_cells(left,right):
    """Robust dominance requires no worse bounds in every matched fold/seed."""
    def indexed(rows):
        result={(r['fold_id'],r['random_seed']):r for r in rows}
        if len(result)!=len(rows) or not result:raise ValueError('unique nonempty configuration cells required')
        return result
    a,b=indexed(left),indexed(right)
    if set(a)!=set(b):raise ValueError('configuration fold/seed grid mismatch')
    # Validate every cell before reporting coverage; an earlier coverage mismatch
    # must not conceal a later incompatible outcome snapshot.
    for key in a:
        if a[key]['comparison_scope']!=b[key]['comparison_scope'] or a[key]['outcome_sha256']!=b[key]['outcome_sha256']:
            raise ValueError('configuration input/outcome scope mismatch')
    cells=[];left_ok=right_ok=True;left_strict=right_strict=False
    for key in sorted(a):
        x,y=a[key],b[key]
        if x['comparison_scope']!=y['comparison_scope'] or x['outcome_sha256']!=y['outcome_sha256']:
            raise ValueError('configuration input/outcome scope mismatch')
        if x['daily_selected']!=y['daily_selected']:
            return dict(status='INCOMPARABLE_COVERAGE',cells=[],dominant=None)
        if not x['selected'] or not y['selected']:
            cells.append(dict(fold_id=key[0],random_seed=key[1],status='NO_SELECTION'))
            left_ok=right_ok=False
            continue
        gain_lower=x['safe_profit']['rate_lower']-y['safe_profit']['rate_upper']
        gain_upper=x['safe_profit']['rate_upper']-y['safe_profit']['rate_lower']
        risk_lower=x['down5']['rate_lower']-y['down5']['rate_upper']
        risk_upper=x['down5']['rate_upper']-y['down5']['rate_lower']
        left_ok &= gain_lower>=0 and risk_upper<=0
        right_ok &= gain_upper<=0 and risk_lower>=0
        left_strict |= gain_lower>0 or risk_upper<0
        right_strict |= gain_upper<0 or risk_lower>0
        cells.append(dict(fold_id=key[0],random_seed=key[1],status='MATCHED',
            safe_profit_difference_bounds=[gain_lower,gain_upper],down5_difference_bounds=[risk_lower,risk_upper]))
    dominant='left' if left_ok and left_strict else 'right' if right_ok and right_strict else None
    return dict(status='ROBUST_DESCRIPTIVE_DOMINANCE' if dominant else 'NO_ROBUST_DOMINANCE',cells=cells,dominant=dominant)


def inspect(root,left_directory,left_sha,right_directory,right_sha):
    refs=[(Path(left_directory).resolve(),left_sha),(Path(right_directory).resolve(),right_sha)]
    if refs[0][0]==refs[1][0]:raise ValueError('distinct configuration summary directories required')
    configs=[]
    for directory,sha in refs:
        verify_seed(root,directory,sha)
        summary=load_plan(directory/'summary.json');data=load_plan(directory/'seed_summary.json')
        evaluation=Path(summary['evaluation_directory']);inputs=load_plan(evaluation/'inputs.json')
        campaign=load_plan(inputs['receipt_path']);jobs={j['job_id']:j for j in campaign['jobs']}
        outcomes={r['fold_id']:r['sha256'] for r in inputs['outcomes']}
        cells=[]
        for row in data['jobs']:
            plan=load_plan(Path(jobs[row['job_id']]['directory'])/'input.json')
            # Only family, loss weights and seed may differ. All data, times,
            # features, labels, policies and calendar remain exact matches.
            scope={k:v for k,v in plan.items() if k not in ['model_family','class_weights','random_seed']}
            cells.append(dict(row,comparison_scope=hashlib.sha256(canonical(scope).encode()).hexdigest(),
                outcome_sha256=outcomes[row['fold_id']]))
        configs.append(cells)
    result=compare_cells(*configs)
    for directory,sha in refs:verify_seed(root,directory,sha)
    return dict(result,left_directory=str(refs[0][0]),left_summary_sha256=left_sha,
        right_directory=str(refs[1][0]),right_summary_sha256=right_sha,
        models_refit=0,formal_H04_accepted=False,finalists_selected=False,
        bounds_are_confidence_intervals=False,risk10_evaluated=False,
        search_preregistration_verified=False,independent_market_evidence=False)
