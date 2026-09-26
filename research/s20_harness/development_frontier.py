"""Same-coverage diagnostic joint-event frontier, not statistical promotion."""
from pathlib import Path
import json
import uuid
import io

import pandas as pd

from .campaign_evaluation import verify as verify_evaluation
from .metrics import binary_bounds
from .runtime import atomic_json,digest,load_plan

COMPARISON='same fold, input scope and exact daily selected counts; unknown-outcome robust domination'
FALSE_CLAIMS=('bounds_are_confidence_intervals','risk10_evaluated','formal_H04_accepted',
              'finalists_selected','configuration_seed_aggregation_performed','non_dominated_is_efficacy_proof')

def compare(rows):
    """Domination must hold even under the unknown-outcome bounds.

    Only identical per-day selection counts within the same fold/input scope
    are compared. This is descriptive, not a paired confidence interval.
    """
    result=[]
    for row in rows:
        value=dict(row,dominated_by=[],comparable_jobs=[])
        if row['selected']==0:
            value['frontier_status']='NO_SELECTION'
        else:
            for other in rows:
                if (other['job_id']==row['job_id'] or other['fold_id']!=row['fold_id']
                        or other['scope_sha256']!=row['scope_sha256']
                        or other['daily_selected']!=row['daily_selected'] or other['selected']==0):
                    continue
                value['comparable_jobs'].append(other['job_id'])
                gain=other['safe_profit']['rate_lower']>=row['safe_profit']['rate_upper']
                risk=other['down5']['rate_upper']<=row['down5']['rate_lower']
                strict=(other['safe_profit']['rate_lower']>row['safe_profit']['rate_upper']
                        or other['down5']['rate_upper']<row['down5']['rate_lower'])
                if gain and risk and strict:value['dominated_by'].append(other['job_id'])
            value['frontier_status']='DOMINATED_UNKNOWN_BOUNDS' if value['dominated_by'] else 'NON_DOMINATED_DIAGNOSTIC'
        result.append(value)
    return result


def reconstruct(root,directory,summary_sha):
    directory=Path(directory).resolve()
    verify_evaluation(root,directory,summary_sha)
    inputs=load_plan(directory/'inputs.json');campaign=load_plan(inputs['receipt_path'])
    manifest=load_plan(Path(campaign['directory'])/'manifest.json')
    registered={j['job_id']:j for j in manifest['jobs']}
    predictions=pd.read_parquet(directory/'evaluated_predictions.parquet')
    rows=[]
    for job in campaign['jobs']:
        registration=registered[job['job_id']]
        if registration.get('pipeline_kind')!='joint':
            raise ValueError('joint safe-profit/down5 endpoints required; binary complement is not downside')
        plan=load_plan(Path(job['directory'])/'input.json')
        if plan['target_id']!='P.joint.v4':raise ValueError('explicit joint target required')
        table=predictions.loc[predictions.job_id.eq(job['job_id'])]
        picked=table.loc[table.selected]
        rows.append(dict(job_id=job['job_id'],fold_id=job['fold_id'],trial_id=registration['trial_id'],
            attempt_id=registration['attempt_id'],input_sha256=registration['input_sha256'],
            source_summary_sha256=job['summary_sha256'],model_family=plan.get('model_family','multinomial'),
            target_id=plan['target_id'],scope_sha256=campaign['same_fold_comparison_scope_sha256'][job['fold_id']],
            candidates=len(table),selected=len(picked),
            daily_selected=[dict(signal_date=d,count=int(picked.signal_date.eq(d).sum())) for d in plan['calendar']],
            safe_profit=binary_bounds([None if pd.isna(v) else bool(v) for v in picked.safe_profit_target]),
            down5=binary_bounds([None if pd.isna(v) else bool(v) for v in picked.down5_target])))
    if len(predictions)!=sum(r['candidates'] for r in rows):raise ValueError('frontier candidate denominator mismatch')
    result=compare(rows)
    verify_evaluation(root,directory,summary_sha)
    return result


def flatten(rows):
    flat=[{k:v for k,v in r.items() if k not in ['safe_profit','down5','daily_selected','dominated_by','comparable_jobs']}
          | {'safe_profit_lower':r['safe_profit']['rate_lower'],'safe_profit_upper':r['safe_profit']['rate_upper'],
             'down5_lower':r['down5']['rate_lower'],'down5_upper':r['down5']['rate_upper'],
             'dominated_by':json.dumps(r['dominated_by'])} for r in rows]
    return pd.DataFrame(flat)


def build(root,directory,summary_sha):
    root=Path(root).resolve();directory=Path(directory).resolve()
    rows=reconstruct(root,directory,summary_sha)
    out=root/'output/experiments/s20_safe_v4/sources'/('development-frontier-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'frontier.json',rows)
    flatten(rows).to_csv(out/'development_frontier.csv',index=False)
    report=dict(directory=str(out),evaluation_directory=str(directory),evaluation_summary_sha256=summary_sha,
        code_sha256=digest(Path(__file__)),jobs=len(rows),models_refit=0,
        comparison=COMPARISON,
        bounds_are_confidence_intervals=False,risk10_evaluated=False,formal_H04_accepted=False,
        finalists_selected=False,configuration_seed_aggregation_performed=False,
        non_dominated_is_efficacy_proof=False,
        artifacts={n:digest(out/n) for n in ['frontier.json','development_frontier.csv']})
    verify_evaluation(root,directory,summary_sha)
    atomic_json(out/'summary.json',report)
    return report


def verify(root,directory,summary_sha):
    root=Path(root).resolve();directory=Path(directory).resolve()
    if not directory.is_relative_to(root/'output/experiments/s20_safe_v4/sources'):
        raise ValueError('frontier outside research scope')
    if digest(directory/'summary.json')!=summary_sha:raise ValueError('frontier summary mismatch')
    report=load_plan(directory/'summary.json')
    expected={'directory','evaluation_directory','evaluation_summary_sha256','code_sha256',
              'jobs','models_refit','comparison','artifacts',*FALSE_CLAIMS}
    if set(report)!=expected or report['comparison']!=COMPARISON:
        raise ValueError('frontier comparison contract mismatch')
    if report['directory']!=str(directory) or set(report['artifacts'])!={'frontier.json','development_frontier.csv'}:
        raise ValueError('frontier artifact scope mismatch')
    def check():
        if (digest(directory/'summary.json')!=summary_sha or digest(Path(__file__))!=report['code_sha256']
                or any(digest(directory/n)!=h for n,h in report['artifacts'].items())):
            raise ValueError('frontier artifact/source changed')
    check()
    for field in FALSE_CLAIMS:
        if report.get(field) is not False:raise ValueError('unsupported frontier claim')
    if type(report['models_refit']) is not int or report['models_refit']!=0:
        raise ValueError('frontier cannot refit models')
    if type(report['jobs']) is not int:raise ValueError('integer frontier job count required')
    rows=reconstruct(root,report['evaluation_directory'],report['evaluation_summary_sha256'])
    if rows!=json.loads((directory/'frontier.json').read_text(encoding='utf-8')) or len(rows)!=report['jobs']:
        raise ValueError('frontier reconstruction mismatch')
    expected=pd.read_csv(io.StringIO(flatten(rows).to_csv(index=False)))
    pd.testing.assert_frame_equal(pd.read_csv(directory/'development_frontier.csv'),expected,check_exact=True)
    check()
    return dict(frontier_recomputed=True,jobs=len(rows),models_refit=0,formal_H04_accepted=False)
