"""Assemble H03 diagnostic deliverables from a verified evaluated campaign.

Packaging does not supply missing legacy reproduction or formal data acceptance.
"""
import json
from pathlib import Path
import uuid
from collections import Counter

import pandas as pd

from .campaign_evaluation import verify as verify_evaluation
from .runtime import atomic_json, digest, load_plan, now
from .baseline_coverage import inventory as coverage_inventory


ACCEPTANCE_GAPS = [
    'H01/H02 formal acceptance must be supplied by runtime',
    'legacy R20/old S20/v3 original-label reproduction and matched-label attribution absent',
    'full required baseline coverage not established',
    'formal H03 acceptance not implemented; diagnostic intake cannot establish baseline coverage',
]


def reconstruct(root, evaluation_directory, evaluation_sha):
    root=Path(root).resolve()
    source=Path(evaluation_directory).resolve()
    verify_evaluation(root,source,evaluation_sha)
    inputs=load_plan(source/'inputs.json')
    campaign=load_plan(inputs['receipt_path'])
    members=json.loads((source/'metrics_by_job.json').read_text(encoding='utf-8'))
    evaluated=pd.read_parquet(source/'evaluated_predictions.parquet')
    if not campaign['jobs'] or len(campaign['jobs'])!=len(members):
        raise ValueError('baseline bundle job/member cardinality mismatch')
    if [(j['job_id'],j['fold_id']) for j in campaign['jobs']]!=[(m['job_id'],m['fold_id']) for m in members]:
        raise ValueError('baseline bundle job/member identity mismatch')
    splits=[];metrics=[];cards=[]
    for job,member in zip(campaign['jobs'],members):
        directory=Path(job['directory'])
        plan=load_plan(directory/'input.json')
        splits.append(dict(job_id=job['job_id'],fold_id=job['fold_id'],
            boundaries=plan['boundaries'],samples=plan['samples'],calendar=plan['calendar'],
            input_sha256=digest(directory/'input.json')))
        cards.append(dict(job_id=job['job_id'],fold_id=job['fold_id'],
            model_family=plan.get('model_family','logistic'),target_id=plan['target_id'],
            evidence_mode=plan['evidence_mode'],directory=str(directory),
            policy_mode=plan['policy']['mode'],
            policy_sha256=load_plan(directory/'selection_report.json')['policy_sha256'],
            summary_sha256=job['summary_sha256']))
        if 'policy_control_sources' in plan:
            evidence_path=directory/'policy_control_evidence.json'
            evidence=load_plan(evidence_path)
            cards[-1]['control_source_evidence']=dict(path=str(evidence_path),sha256=digest(evidence_path),
                method=evidence['method'],period=evidence['period'],rows=evidence['rows'],
                status_counts=dict(Counter(r['reason'] for r in evidence['audit'])),
                local_receipt_artifact_bindings_verified=evidence['local_receipt_artifact_bindings_verified'],
                external_timestamp_authenticity_proven=evidence['external_timestamp_authenticity_proven'],
                price_adjustment_and_calendar_independently_verified=evidence['price_adjustment_and_calendar_independently_verified'])
        for view in ['all_candidates','selected_candidates']:
            value=member['metrics'][view]
            metrics.append(dict(job_id=job['job_id'],fold_id=job['fold_id'],
                model_family=plan.get('model_family','logistic'),target_id=plan['target_id'],
                policy_mode=plan['policy']['mode'],
                view=view,**value['event_bounds'],
                brier_known_only=value['brier_known_only'],
                scored_mature_rows=value['scored_mature_rows'],
                known_only_diagnostic=True))
    outer_ref=campaign['outer_prediction_ledger']
    outer=pd.read_parquet(outer_ref['path'])
    # Keep the entire scored universe, with evaluation and provenance separate.
    merged=outer.merge(evaluated[['job_id','sample_id','evaluation_target','evaluation_status']],
        on=['job_id','sample_id'],how='left',validate='one_to_one',sort=False)
    if len(merged)!=len(evaluated) or merged.evaluation_status.isna().any():
        raise ValueError('baseline bundle evaluation denominator mismatch')
    verify_evaluation(root,source,evaluation_sha)
    return splits,merged,pd.DataFrame(metrics),cards


def build(root, evaluation_directory, evaluation_sha):
    root=Path(root).resolve();source=Path(evaluation_directory).resolve()
    splits,merged,metrics,cards=reconstruct(root,source,evaluation_sha)
    out=root/'output/experiments/s20_safe_v4/sources'/('baseline-bundle-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'split_manifest.json',dict(jobs=splits,formal_time_availability_accepted=False))
    merged.to_parquet(out/'baseline_oof.parquet',index=False)
    metrics.to_csv(out/'baseline_metrics.csv',index=False)
    gaps=list(ACCEPTANCE_GAPS)
    bundle=dict(status='DIAGNOSTIC_BUNDLE',at=now(),jobs=cards,
        baseline_coverage=coverage_inventory(cards),
        evaluation_directory=str(source),evaluation_summary_sha256=evaluation_sha,
        prediction_rows=len(merged),metric_rows=len(metrics),
        recorded_provenance_valid_rows=int(merged.recorded_oof_provenance_valid.sum()),
        legacy_reproduction_complete=False,formal_H03_accepted=False,
        formal_training_authorized=False,acceptance_gaps=gaps,
        artifacts={n:digest(out/n) for n in ['split_manifest.json','baseline_oof.parquet','baseline_metrics.csv']})
    atomic_json(out/'baseline_bundle.json',bundle)
    verify_evaluation(root,source,evaluation_sha)
    result=dict(directory=str(out),status='DIAGNOSTIC_BUNDLE',models_refit=0,
        formal_H03_accepted=False,formal_training_authorized=False,
        evaluation_directory=str(source),evaluation_summary_sha256=evaluation_sha,
        code_sha256=digest(Path(__file__)),acceptance_gaps=gaps,
        artifacts={n:digest(out/n) for n in ['split_manifest.json','baseline_oof.parquet',
            'baseline_metrics.csv','baseline_bundle.json']})
    atomic_json(out/'summary.json',result)
    return result


def verify(root, directory, summary_sha):
    """Rebuild H03 bundle values from evaluated campaign; does not fit models."""
    import io
    root,directory=Path(root).resolve(),Path(directory).resolve()
    if not directory.is_relative_to(root): raise ValueError('baseline bundle escapes root')
    if digest(directory/'summary.json')!=summary_sha: raise ValueError('baseline bundle summary pin mismatch')
    summary=load_plan(directory/'summary.json')
    names={'split_manifest.json','baseline_oof.parquet','baseline_metrics.csv','baseline_bundle.json'}
    if set(summary['artifacts'])!=names: raise ValueError('baseline bundle artifact scope mismatch')
    def check():
        if digest(directory/'summary.json')!=summary_sha or any(digest(directory/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('baseline bundle artifact changed')
        if digest(Path(__file__))!=summary['code_sha256']: raise ValueError('baseline bundle code changed')
    check();bundle=load_plan(directory/'baseline_bundle.json')
    if summary.get('acceptance_gaps')!=ACCEPTANCE_GAPS:
        raise ValueError('baseline bundle required acceptance gaps changed')
    for value in [summary,bundle]:
        if (value.get('status')!='DIAGNOSTIC_BUNDLE' or value.get('formal_H03_accepted') is not False
                or value.get('formal_training_authorized') is not False):
            raise ValueError('baseline bundle acceptance claim invalid')
    if summary.get('models_refit')!=0 or bundle.get('legacy_reproduction_complete') is not False:
        raise ValueError('baseline bundle unsupported reproduction claim')
    if bundle['artifacts']!={n:summary['artifacts'][n] for n in names if n!='baseline_bundle.json'}:
        raise ValueError('baseline bundle nested artifact pins differ')
    for key in ['evaluation_directory','evaluation_summary_sha256','acceptance_gaps']:
        if summary[key]!=bundle[key]: raise ValueError('baseline bundle parent/gaps mismatch')
    source=Path(summary['evaluation_directory']).resolve()
    if not source.is_relative_to(root): raise ValueError('baseline evaluation escapes root')
    splits,merged,metrics,cards=reconstruct(root,source,summary['evaluation_summary_sha256'])
    if load_plan(directory/'split_manifest.json')!=dict(jobs=splits,formal_time_availability_accepted=False):
        raise ValueError('baseline split reconstruction mismatch')
    pd.testing.assert_frame_equal(pd.read_parquet(directory/'baseline_oof.parquet'),merged,check_exact=True)
    expected=pd.read_csv(io.StringIO(metrics.to_csv(index=False)))
    pd.testing.assert_frame_equal(pd.read_csv(directory/'baseline_metrics.csv'),expected,check_exact=True)
    if (bundle['jobs']!=cards or bundle['prediction_rows']!=len(merged) or bundle['metric_rows']!=len(metrics)
            or bundle['recorded_provenance_valid_rows']!=int(merged.recorded_oof_provenance_valid.sum())):
        raise ValueError('baseline bundle semantic counts differ')
    if bundle.get('baseline_coverage')!=coverage_inventory(cards):
        raise ValueError('baseline coverage inventory differs from verified jobs')
    check()
    return dict(rows=len(merged),jobs=len(cards),bundle_values_recomputed=True,models_refit=0,
        formal_H03_accepted=False,formal_training_authorized=False)
