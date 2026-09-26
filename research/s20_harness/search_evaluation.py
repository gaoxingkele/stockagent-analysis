"""Evaluate the complete registered synthetic search without additional fits."""
from pathlib import Path
import uuid
import pandas as pd

from .registered_search_run import verify_receipt
from .joint_evaluation import evaluate
from .joint_replay import replay
from .runtime import atomic_json,digest,load_plan
from .label_availability import _instant
from .trial_budget import canonical
from .reliability_export import csv_text as reliability_csv


def reconstruct(root,receipt_path,receipt_sha,outcome_refs):
    checked=verify_receipt(root,receipt_path,receipt_sha)
    if checked['status']!='COMPLETED_SYNTHETIC':raise ValueError('complete registered search required for comparison')
    receipt=load_plan(receipt_path);folds={j['fold_id'] for j in receipt['jobs']}
    if not isinstance(outcome_refs,list) or any(set(r)!={'fold_id','path','sha256'} for r in outcome_refs):
        raise ValueError('exact search outcome references required')
    if len(outcome_refs)!=len(folds) or {r['fold_id'] for r in outcome_refs}!=folds:
        raise ValueError('one shared outcome snapshot per registered fold required')
    refs={r['fold_id']:r for r in outcome_refs};payloads={};pins={}
    for fold,ref in refs.items():
        path=Path(ref['path']).resolve()
        if digest(path)!=ref['sha256']:raise ValueError('search outcome pin mismatch')
        pins[path]=ref['sha256'];payload=load_plan(path)
        if set(payload)!={'target_id','evaluation_at','outcomes'} or payload['target_id']!='P.joint.v4':
            raise ValueError('exact joint search outcomes required')
        payloads[fold]=payload
    if len({_instant(p['evaluation_at']) for p in payloads.values()})!=1:
        raise ValueError('shared search evaluation cutoff required')
    tables=[];metrics=[]
    for job in receipt['jobs']:
        artifact=job['artifact'];directory=Path(artifact['directory'])
        replay(directory,artifact['summary_sha256'])
        plan=load_plan(directory/'input.json');payload=payloads[job['fold_id']]
        rows,report=evaluate(pd.read_parquet(directory/'candidate_ledger.parquet'),pd.DataFrame(plan['samples']),
            pd.DataFrame(payload['outcomes']),target_id=payload['target_id'],evaluation_at=payload['evaluation_at'],calendar=plan['calendar'])
        rows['candidate_id']=job['candidate_id'];rows['job_id']=job['shared_trial_id']
        rows['fold_id']=job['fold_id'];rows['random_seed']=plan['random_seed']
        tables.append(rows)
        metrics.append(dict(candidate_id=job['candidate_id'],job_id=job['shared_trial_id'],fold_id=job['fold_id'],
            random_seed=plan['random_seed'],input_sha256=job['input_sha256'],metrics=report))
    joined=pd.concat(tables,ignore_index=True)
    if joined.duplicated(['job_id','sample_id']).any():raise ValueError('duplicate search prediction identity')
    verify_receipt(root,receipt_path,receipt_sha)
    if any(digest(p)!=h for p,h in pins.items()):raise ValueError('search outcomes changed')
    return joined,metrics


def build(root,receipt_path,receipt_sha,outcome_refs):
    rows,metrics=reconstruct(root,receipt_path,receipt_sha,outcome_refs)
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('search-evaluation-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    rows.to_parquet(out/'evaluated_predictions.parquet',index=False)
    atomic_json(out/'metrics_by_job.json',metrics)
    (out/'selected_reliability.csv').write_text(reliability_csv(metrics),encoding='utf-8',newline='')
    report=dict(directory=str(out),receipt_path=str(Path(receipt_path).resolve()),receipt_sha256=receipt_sha,
        outcomes=outcome_refs,rows=len(rows),jobs=len(metrics),code_sha256=digest(Path(__file__)),
        model_fits=0,calibrator_fits=0,formal_H04_accepted=False,finalists_selected=False,
        artifacts={n:digest(out/n) for n in ['evaluated_predictions.parquet','metrics_by_job.json','selected_reliability.csv']})
    atomic_json(out/'summary.json',report)
    verify(root,out,digest(out/'summary.json'))
    return report


def verify(root,directory,sha):
    directory=Path(directory).resolve();scope=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'
    if not directory.is_relative_to(scope) or digest(directory/'summary.json')!=sha:
        raise ValueError('search evaluation pin/scope mismatch')
    report=load_plan(directory/'summary.json')
    if set(report)!={'directory','receipt_path','receipt_sha256','outcomes','rows','jobs','code_sha256',
                      'model_fits','calibrator_fits','formal_H04_accepted','finalists_selected','artifacts'}:
        raise ValueError('exact search evaluation summary required')
    if report['directory']!=str(directory) or set(report['artifacts'])!={'evaluated_predictions.parquet','metrics_by_job.json','selected_reliability.csv'}:
        raise ValueError('search evaluation artifact scope mismatch')
    if any(type(report[k])is not int or report[k]!=0 for k in ['model_fits','calibrator_fits']) or any(report[k] is not False for k in ['formal_H04_accepted','finalists_selected']):
        raise ValueError('unsupported search evaluation claim')
    def check():
        if digest(directory/'summary.json')!=sha or digest(Path(__file__))!=report['code_sha256'] or any(digest(directory/n)!=h for n,h in report['artifacts'].items()):
            raise ValueError('search evaluation source/artifact changed')
    check()
    rows,metrics=reconstruct(root,report['receipt_path'],report['receipt_sha256'],report['outcomes'])
    pd.testing.assert_frame_equal(rows,pd.read_parquet(directory/'evaluated_predictions.parquet'),check_exact=True)
    if canonical(metrics)!=canonical(load_list(directory/'metrics_by_job.json')) or type(report['rows'])is not int or type(report['jobs'])is not int or report['rows']!=len(rows) or report['jobs']!=len(metrics):
        raise ValueError('search evaluation metrics/count mismatch')
    if (directory/'selected_reliability.csv').read_bytes()!=reliability_csv(metrics).encode('utf-8'):
        raise ValueError('search reliability reconstruction mismatch')
    check()
    return dict(search_evaluation_recomputed=True,rows=len(rows),jobs=len(metrics),models_refit=0,formal_H04_accepted=False)


def load_list(path):
    import json
    return json.loads(Path(path).read_text(encoding='utf-8'))
