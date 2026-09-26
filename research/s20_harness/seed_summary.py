"""Descriptive fold aggregation per seed; seeds are not market replicates."""
from pathlib import Path
import json
import uuid
from itertools import combinations
import numpy as np
import pandas as pd

from .development_frontier import reconstruct as reconstruct_jobs
from .campaign_evaluation import verify as verify_evaluation
from .runtime import atomic_json, digest, load_plan


def prediction_disagreement(predictions, rows, seeds, folds):
    """Compare aligned candidates, not outcome labels, within each fold."""
    reports=[]
    for fold in folds:
        members={r['random_seed']:r['job_id'] for r in rows if r['fold_id']==fold}
        if set(members)!=set(seeds) or len([r for r in rows if r['fold_id']==fold])!=len(seeds):
            raise ValueError('complete unique prediction seed grid required')
        tables={}
        for seed in seeds:
            table=predictions.loc[predictions.job_id.eq(members[seed])].copy()
            if table.sample_id.isna().any() or table.sample_id.duplicated().any():
                raise ValueError('unique seed candidate identities required')
            if not table.fold_id.eq(fold).all():raise ValueError('seed prediction fold mismatch')
            if not table.selected.map(lambda v:isinstance(v,(bool,np.bool_))).all():
                raise ValueError('boolean seed selection required')
            tables[seed]=table.set_index('sample_id').sort_index()
        for left,right in combinations(seeds,2):
            a,b=tables[left],tables[right]
            if not a.index.equals(b.index):raise ValueError('seed candidate universe mismatch')
            pd.testing.assert_frame_equal(a[['entity_id','signal_date']],b[['entity_id','signal_date']],check_exact=True)
            channels={}
            for name,columns in [('raw',['raw_p_'+c for c in 'ABCD']),('calibrated',['p_'+c for c in 'ABCD'])]:
                x=a[columns].to_numpy(dtype=float);y=b[columns].to_numpy(dtype=float)
                if np.isinf(x).any() or np.isinf(y).any():raise ValueError('infinite seed probability')
                known_x=np.isfinite(x).all(axis=1);known_y=np.isfinite(y).all(axis=1)
                if ((np.isfinite(x).any(axis=1)!=known_x).any() or (np.isfinite(y).any(axis=1)!=known_y).any()):
                    raise ValueError('partial joint probability vector')
                both=known_x&known_y;delta=np.abs(x[both]-y[both])
                identical=np.array_equal(x,y,equal_nan=True)
                channels[name]=dict(both_available=int(both.sum()),both_unavailable=int((~known_x&~known_y).sum()),
                    availability_mismatch=int((known_x!=known_y).sum()),
                    changed_candidates=int((delta!=0).any(axis=1).sum()),
                    max_absolute_difference=float(delta.max()) if delta.size else None,
                    mean_absolute_difference=float(delta.mean()) if delta.size else None,
                    exactly_identical=identical)
            picked_a=set(a.index[a.selected]);picked_b=set(b.index[b.selected]);union=picked_a|picked_b
            reports.append(dict(fold_id=fold,left_seed=left,right_seed=right,left_job=members[left],right_job=members[right],
                candidates=len(a),probabilities=channels,selection_disagreements=len(picked_a^picked_b),
                selected_intersection=len(picked_a&picked_b),selected_union=len(union),
                selection_jaccard=len(picked_a&picked_b)/len(union) if union else None,
                same_predictions_and_selection=all(c['exactly_identical'] for c in channels.values()) and picked_a==picked_b,
                independent_market_evidence=False,disagreement_is_confidence_interval=False))
    return reports


def aggregate(rows, seeds, folds):
    expected={(f,s) for f in folds for s in seeds}
    cells=[(r['fold_id'],r['random_seed']) for r in rows]
    if len(cells)!=len(set(cells)) or set(cells)!=expected:
        raise ValueError('complete unique fold/seed result grid required')
    summaries=[]
    for seed in seeds:
        selected_rows=[next(r for r in rows if r['fold_id']==fold and r['random_seed']==seed) for fold in folds]
        result=dict(random_seed=seed,folds=list(folds),job_ids=[r['job_id'] for r in selected_rows],
            candidates=sum(r['candidates'] for r in selected_rows),selected=sum(r['selected'] for r in selected_rows),
            zero_selection_folds=[r['fold_id'] for r in selected_rows if r['selected']==0])
        result['selection_fraction']=result['selected']/result['candidates'] if result['candidates'] else None
        for event in ['safe_profit','down5']:
            parts=[r[event] for r in selected_rows]
            counts={k:sum(p[k] for p in parts) for k in ['denominator','known','positive','unknown']}
            n=counts['denominator'];positive=counts['positive'];unknown=counts['unknown'];known=counts['known']
            if n!=result['selected'] or known+unknown!=n or not 0<=positive<=known:
                raise ValueError('seed aggregate outcome denominator mismatch')
            result[event]=dict(counts,rate_lower=positive/n if n else None,
                rate_upper=(positive+unknown)/n if n else None,known_only_rate=positive/known if known else None,
                bounds_are_confidence_intervals=False)
            # A zero-selection fold is undefined, never silently omitted from a macro mean.
            result[event]['equal_fold_lower']=sum(p['rate_lower'] for p in parts)/len(parts) if all(p['denominator'] for p in parts) else None
            result[event]['equal_fold_upper']=sum(p['rate_upper'] for p in parts)/len(parts) if all(p['denominator'] for p in parts) else None
        summaries.append(result)
    return summaries


def reconstruct(root,directory,summary_sha):
    directory=Path(directory).resolve()
    rows=reconstruct_jobs(root,directory,summary_sha)
    inputs=load_plan(directory/'inputs.json');campaign=load_plan(inputs['receipt_path'])
    manifest=load_plan(Path(campaign['directory'])/'manifest.json')
    if manifest.get('schema_version')!='7':raise ValueError('registered seed campaign required')
    jobs={j['job_id']:j for j in campaign['jobs']}
    for row in rows:
        path=Path(jobs[row['job_id']]['directory'])
        plan=load_plan(path/'input.json');card=load_plan(path/'joint_card.json')
        row['random_seed']=plan['random_seed']
        expected_effect='deterministic_frequency' if plan.get('model_family')=='mature_frequency' else 'unused_by_lbfgs'
        if plan.get('model_family')=='shallow_joint':expected_effect='feature_permutation_and_split_ties'
        expected=dict(requested_seed=plan['random_seed'],seed_effect=expected_effect,independent_market_evidence=False)
        if card.get('randomness')!=expected:raise ValueError('seed model-card identity mismatch')
        if expected_effect!='deterministic_frequency' and card.get('parameters',{}).get('random_state')!=plan['random_seed']:
            raise ValueError('seed estimator identity mismatch')
        row['randomness']=expected
    result=dict(seed_grid=manifest['seed_grid'],fold_grid=manifest['fold_grid'],jobs=rows,
        per_seed=aggregate(rows,manifest['seed_grid'],manifest['fold_grid']),
        prediction_disagreement=prediction_disagreement(pd.read_parquet(campaign['outer_prediction_ledger']['path']),
            rows,manifest['seed_grid'],manifest['fold_grid']),
        seed_pooling_performed=False,independent_market_replicates=False,
        bounds_are_confidence_intervals=False,formal_H04_accepted=False,finalists_selected=False,
        risk10_evaluated=False,models_refit=0,
        scope='one fixed joint-model campaign; descriptive per-seed fold aggregation')
    verify_evaluation(root,directory,summary_sha)
    return result


def build(root,directory,summary_sha):
    directory=Path(directory).resolve();root=Path(root).resolve()
    result=reconstruct(root,directory,summary_sha)
    out=root/'output/experiments/s20_safe_v4/sources'/('seed-summary-'+uuid.uuid4().hex)
    out.mkdir(parents=True);atomic_json(out/'seed_summary.json',result)
    report=dict(directory=str(out),evaluation_directory=str(directory),evaluation_summary_sha256=summary_sha,
        code_sha256=digest(Path(__file__)),artifacts={'seed_summary.json':digest(out/'seed_summary.json')},
        models_refit=0,formal_H04_accepted=False)
    verify_evaluation(root,directory,summary_sha)
    atomic_json(out/'summary.json',report)
    return report


def verify(root,directory,summary_sha):
    root=Path(root).resolve();directory=Path(directory).resolve()
    if not directory.is_relative_to(root/'output/experiments/s20_safe_v4/sources'):
        raise ValueError('seed summary outside research scope')
    if digest(directory/'summary.json')!=summary_sha:raise ValueError('seed summary pin mismatch')
    report=load_plan(directory/'summary.json')
    if set(report)!={'directory','evaluation_directory','evaluation_summary_sha256','code_sha256','artifacts','models_refit','formal_H04_accepted'}:
        raise ValueError('exact seed summary contract required')
    if report['directory']!=str(directory) or set(report['artifacts'])!={'seed_summary.json'}:
        raise ValueError('seed summary artifact scope mismatch')
    if type(report['models_refit']) is not int or report['models_refit']!=0 or report['formal_H04_accepted'] is not False:
        raise ValueError('unsupported seed summary claim')
    def check():
        if (digest(directory/'summary.json')!=summary_sha or digest(Path(__file__))!=report['code_sha256']
                or digest(directory/'seed_summary.json')!=report['artifacts']['seed_summary.json']):
            raise ValueError('seed summary source/artifact changed')
    check()
    result=reconstruct(root,report['evaluation_directory'],report['evaluation_summary_sha256'])
    # Canonical serialization also distinguishes false from 0 and true from 1.
    if json.dumps(result,sort_keys=True,allow_nan=False)!=json.dumps(load_plan(directory/'seed_summary.json'),sort_keys=True,allow_nan=False):
        raise ValueError('seed summary reconstruction mismatch')
    check()
    return dict(seed_summary_recomputed=True,models_refit=0,formal_H04_accepted=False)
