"""Evaluate all registered baselines using one outcome snapshot per outer fold."""
import json
from pathlib import Path
import uuid

import pandas as pd

from .baseline_outer_ledger import verify_receipt
from .baseline_evaluation import build as evaluate_job
from .runtime import atomic_json, digest, load_plan, now


def build(root, receipt_path, receipt_sha, outcome_refs):
    root,receipt_path=Path(root).resolve(),Path(receipt_path).resolve()
    verify_receipt(root,receipt_path,receipt_sha)
    campaign=load_plan(receipt_path)
    folds={j['fold_id'] for j in campaign['jobs']}
    if not isinstance(outcome_refs,list) or any(not isinstance(r,dict) or set(r)!={'fold_id','path','sha256'} for r in outcome_refs):
        raise ValueError('exact fold outcome references required')
    if len(outcome_refs)!=len(folds) or {r['fold_id'] for r in outcome_refs}!=folds:
        raise ValueError('one shared outcome snapshot per fold required')
    refs={r['fold_id']:r for r in outcome_refs}
    code_dir=Path(__file__).resolve().parent
    code_paths={str(p.resolve()) for p in code_dir.glob('*.py')}
    pins={str(receipt_path):receipt_sha,**{p:digest(Path(p)) for p in code_paths}}
    payloads={}
    for fold,ref in refs.items():
        path=Path(ref['path']).resolve()
        if digest(path)!=ref['sha256']: raise ValueError('fold outcome pin mismatch')
        payload=load_plan(path)
        if set(payload)!={'target_id','evaluation_at','outcomes'}:
            raise ValueError('exact fold outcome schema required')
        pins[str(path)]=ref['sha256'];payloads[fold]=payload
    # Fix evaluation time across the whole comparison; no favorable cutoff per model.
    from .label_availability import _instant
    if len({_instant(p['evaluation_at']) for p in payloads.values()})!=1:
        raise ValueError('shared campaign evaluation cutoff required')
    def check():
        if {str(p.resolve()) for p in code_dir.glob('*.py')}!=code_paths:
            raise ValueError('campaign evaluation code inventory changed')
        if any(digest(Path(p))!=h for p,h in pins.items()):
            raise ValueError('campaign evaluation source changed')
    check()
    tables=[];members=[]
    for job in campaign['jobs']:
        ref=refs[job['fold_id']]
        pipeline=job.get('pipeline_kind','baseline')
        if pipeline=='joint':
            from .joint_evaluation import build as evaluator
        elif pipeline=='baseline':evaluator=evaluate_job
        else:raise ValueError('unknown evaluation pipeline identity')
        report=evaluator(root,job['directory'],job['summary_sha256'],ref['path'],ref['sha256'])
        directory=Path(report['directory'])
        pins[str(directory/'summary.json')]=digest(directory/'summary.json')
        if load_plan(directory/'summary.json')!=report:
            raise ValueError('evaluation member summary mismatch')
        for name,sha in report['artifacts'].items():
            path=(directory/name).resolve()
            if path.parent!=directory.resolve():
                raise ValueError('evaluation member artifact outside directory')
            pins[str(path)]=sha
        check()
        table=pd.read_parquet(directory/'evaluated_candidates.parquet')
        table['job_id']=job['job_id'];table['fold_id']=job['fold_id']
        tables.append(table)
        members.append(dict(job_id=job['job_id'],fold_id=job['fold_id'],directory=str(directory),
            summary_sha256=digest(directory/'summary.json'),metrics=load_plan(directory/'metrics.json')))
        check()
    joined=pd.concat(tables,ignore_index=True)
    if joined.duplicated(['job_id','sample_id']).any(): raise ValueError('duplicate evaluated job/sample')
    verify_receipt(root,receipt_path,receipt_sha);check()
    out=root/'output/experiments/s20_safe_v4/sources'/('campaign-evaluation-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    joined.to_parquet(out/'evaluated_predictions.parquet',index=False)
    atomic_json(out/'metrics_by_job.json',members)
    atomic_json(out/'inputs.json',dict(source_pins=pins,outcomes=outcome_refs,
        receipt_path=str(receipt_path),receipt_sha256=receipt_sha))
    report=dict(directory=str(out),at=now(),rows=len(joined),jobs=len(members),outer_folds=len(folds),
        evaluation_at=next(iter(payloads.values()))['evaluation_at'],
        unique_signal_dates=int(joined.signal_date.nunique()),
        models_refit=0,all_candidate_rows_retained=True,shared_outcomes_per_fold=True,
        model_rows_are_independent_observations=False,confidence_intervals_computed=False,
        formal_H03_accepted=False,formal_promotion_authorized=False,
        artifacts={n:digest(out/n) for n in ['evaluated_predictions.parquet','metrics_by_job.json','inputs.json']})
    check()
    atomic_json(out/'summary.json',report)
    return report


def verify(root, directory, summary_sha):
    """Recompute outcome attachment and metrics, without fitting any model."""
    from .baseline_evaluation import evaluate
    from .label_availability import _instant
    root,directory=Path(root).resolve(),Path(directory).resolve()
    scope=root/'output/experiments/s20_safe_v4/sources'
    summary=directory/'summary.json'
    if not directory.is_relative_to(scope) or digest(summary)!=summary_sha:
        raise ValueError('campaign evaluation pin/scope mismatch')
    report=load_plan(summary)
    if Path(report['directory']).resolve()!=directory:
        raise ValueError('campaign evaluation directory mismatch')
    names={'evaluated_predictions.parquet','metrics_by_job.json','inputs.json'}
    if set(report['artifacts'])!=names:
        raise ValueError('exact campaign evaluation artifacts required')
    pins={str(summary):summary_sha,**{str(directory/n):report['artifacts'][n] for n in names}}
    def check():
        if any(digest(Path(p))!=h for p,h in pins.items()):
            raise ValueError('campaign evaluation verification pin mismatch')
    check()
    inputs=load_plan(directory/'inputs.json')
    code_dir=Path(__file__).resolve().parent
    expected_code={str(p.resolve()):digest(p) for p in code_dir.glob('*.py')}
    recorded_code={p:h for p,h in inputs['source_pins'].items() if Path(p).suffix=='.py'}
    if recorded_code!=expected_code:
        raise ValueError('campaign evaluation code inventory mismatch')
    for path,sha in inputs['source_pins'].items():
        if path in pins and pins[path]!=sha:
            raise ValueError('conflicting campaign evaluation pins')
        pins[path]=sha
    check()
    receipt=Path(inputs['receipt_path']).resolve()
    if pins.get(str(receipt))!=inputs['receipt_sha256']:
        raise ValueError('unbound campaign receipt')
    verify_receipt(root,receipt,inputs['receipt_sha256'])
    campaign=load_plan(receipt)
    refs=inputs['outcomes'];folds={j['fold_id'] for j in campaign['jobs']}
    if len(refs)!=len(folds) or {r['fold_id'] for r in refs}!=folds:
        raise ValueError('one shared outcome snapshot per fold required')
    refs={r['fold_id']:r for r in refs}
    members=json.loads((directory/'metrics_by_job.json').read_text(encoding='utf-8'))
    if [(m['job_id'],m['fold_id']) for m in members]!=[(j['job_id'],j['fold_id']) for j in campaign['jobs']]:
        raise ValueError('evaluation job denominator/order mismatch')
    tables=[]
    for job,member in zip(campaign['jobs'],members):
        child=Path(member['directory']).resolve()
        if not child.is_relative_to(scope) or pins.get(str(child/'summary.json'))!=member['summary_sha256']:
            raise ValueError('unbound evaluation member')
        child_summary=load_plan(child/'summary.json')
        if Path(child_summary['run_directory']).resolve()!=Path(job['directory']).resolve() or child_summary['run_summary_sha256']!=job['summary_sha256']:
            raise ValueError('evaluation member parent mismatch')
        ref=refs[job['fold_id']];path=Path(ref['path']).resolve()
        if pins.get(str(path))!=ref['sha256']:
            raise ValueError('unbound campaign outcome')
        payload=load_plan(path);plan=load_plan(Path(job['directory'])/'input.json')
        if payload['target_id']!=plan['target_id'] or _instant(payload['evaluation_at'])!=_instant(report['evaluation_at']):
            raise ValueError('campaign evaluation target/cutoff mismatch')
        pipeline=job.get('pipeline_kind','baseline')
        kwargs={}
        if pipeline=='joint':
            from .joint_evaluation import evaluate as evaluator
            kwargs['target_id']=payload['target_id']
        elif pipeline=='baseline':evaluator=evaluate
        else:raise ValueError('unknown evaluation pipeline identity')
        rows,metrics=evaluator(pd.read_parquet(Path(job['directory'])/'candidate_ledger.parquet'),
            pd.DataFrame(plan['samples']),
            pd.DataFrame(payload['outcomes'],columns=['sample_id','target','label_available_at']),
            evaluation_at=payload['evaluation_at'],calendar=plan['calendar'],**kwargs)
        if metrics!=member['metrics']:
            raise ValueError('campaign evaluation metrics reconstruction mismatch')
        rows['job_id']=job['job_id'];rows['fold_id']=job['fold_id'];tables.append(rows)
    joined=pd.concat(tables,ignore_index=True)
    saved=pd.read_parquet(directory/'evaluated_predictions.parquet')
    # Parquet infers bool when every outcome is mature, but object when nulls
    # remain. Compare the explicit nullable-boolean contract in both cases.
    if 'evaluation_target' in joined.columns:
        for frame in (saved,joined):
            frame['evaluation_target']=frame['evaluation_target'].astype('boolean')
    pd.testing.assert_frame_equal(saved,joined,check_exact=True)
    expected=dict(rows=len(joined),jobs=len(members),outer_folds=len(folds),
        unique_signal_dates=int(joined.signal_date.nunique()),models_refit=0,
        all_candidate_rows_retained=True,shared_outcomes_per_fold=True,
        model_rows_are_independent_observations=False,confidence_intervals_computed=False,
        formal_H03_accepted=False,formal_promotion_authorized=False)
    if any(report.get(k)!=v for k,v in expected.items()):
        raise ValueError('campaign evaluation summary mismatch')
    check()
    return dict(rows=len(joined),metrics_recomputed=True,outcome_attachment_recomputed=True,
        models_refit=0,formal_H03_accepted=False,formal_promotion_authorized=False)
