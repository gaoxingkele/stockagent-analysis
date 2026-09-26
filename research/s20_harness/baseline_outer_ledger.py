"""Retain forward candidates and audit recorded fit/calibration dependencies."""
import json
from pathlib import Path

import pandas as pd

from .baseline_run import verify
from .oof_audit import audit, membership_hash
from .runtime import digest, load_plan
from .label_availability import _instant


def collect(jobs):
    tables=[];seen=set()
    for job in jobs:
        if job['job_id'] in seen: raise ValueError('duplicate outer ledger job')
        seen.add(job['job_id'])
        if job.get('pipeline_kind')=='joint':
            from .joint_outer_ledger import collect_one
            tables.append(collect_one(job))
            continue
        if job.get('pipeline_kind','baseline')!='baseline':
            raise ValueError('unknown outer pipeline identity')
        directory=Path(job['directory']).resolve()
        verify(directory,job['summary_sha256'])
        read=lambda name:json.loads((directory/name).read_text(encoding='utf-8'))
        plan=read('input.json');base=read('baseline_card.json');cal=read('calibration_card.json')
        if plan['policy']['mode'] not in {'score_only_control', 'atr_liquidity_control'}:
            raise ValueError('outer ledger requires fixed policy with no learned policy labels')
        predictions=pd.read_parquet(directory/'calibrated_predictions.parquet')
        outer=predictions.loc[predictions.segment.eq('outer-test')].copy()
        ledger=pd.read_parquet(directory/'candidate_ledger.parquet')
        if outer.sample_id.tolist()!=ledger.sample_id.tolist():
            raise ValueError('outer candidate denominator/order mismatch')
        pd.testing.assert_series_equal(outer.calibrated_probability.reset_index(drop=True),
            ledger.score.reset_index(drop=True),check_names=False,check_exact=True)
        if plan['policy']['mode'] == 'atr_liquidity_control':
            from .recommendation_policy import apply
            columns = ['sample_id', 'entity_id', 'signal_date', 'prediction_at', 'score', 'risk']
            rebuilt, selection = apply(ledger[columns], plan['policy'], plan['calendar'],
                                       controls=pd.DataFrame(plan['policy_controls']))
            pd.testing.assert_frame_equal(ledger, rebuilt, check_exact=True)
            if selection != read('selection_report.json'):
                raise ValueError('ATR/liquidity policy reconstruction mismatch')
        dependencies=base['fit_sample_ids']+cal['calibration_sample_ids']
        # No tune or policy labels are consumed by this fixed pipeline. The
        # supplied frozen policy time is still a dependency, not a live receipt.
        cutoff=max(_instant(plan['boundaries'][3]['start_at']),_instant(plan['policy']['frozen_at'])).isoformat()
        model_id=job['job_id']
        models={model_id:dict(model_path=str(directory/'calibration_card.json'),
            model_sha256=digest(directory/'calibration_card.json'),
            information_cutoff_at=cutoff,dependency_sha256=membership_hash(dependencies))}
        checked,_=audit(pd.DataFrame(plan['samples']),
            pd.DataFrame(dict(sample_id=outer.sample_id.tolist(),model_id=model_id,
                              score=outer.calibrated_probability.tolist())),
            models,{model_id:dependencies})
        anchor_details=None
        if 'anchor_context' in plan:
            from .anchor_outer_audit import audit_outer
            checked,anchor_details=audit_outer(plan,base,cal,outer,directory/'calibration_card.json',cutoff)
        table=ledger.copy()
        table['job_id']=model_id;table['fold_id']=job['fold_id'];table['model_family']=plan.get('model_family','logistic')
        table['target_id']=plan['target_id'];table['raw_probability']=outer.raw_probability.to_numpy()
        table['recorded_oof_provenance_valid']=checked.recorded_oof_provenance_valid.to_numpy()
        table['provenance_reasons_json']=checked.reasons.map(json.dumps).to_numpy()
        table['information_cutoff_at']=cutoff
        if anchor_details is not None:
            for column in anchor_details:
                table[column]=anchor_details[column].to_numpy()
        table['source_summary_sha256']=job['summary_sha256']
        table['historical_availability_proven']=False
        verify(directory,job['summary_sha256'])
        tables.append(table)
    if not tables: raise ValueError('nonempty outer jobs required')
    result=pd.concat(tables,ignore_index=True)
    if result.duplicated(['job_id','sample_id']).any(): raise ValueError('duplicate job/sample prediction')
    return result


def verify_receipt(root, receipt_path, receipt_sha):
    """Reconstruct a campaign's saved outer table; no training or label access."""
    from .bounded_baseline import verify_process
    from .process_runner import Limits
    root,receipt_path=Path(root).resolve(),Path(receipt_path).resolve()
    scope=root/'output/experiments/s20_safe_v4/sources'
    if not receipt_path.is_relative_to(scope) or digest(receipt_path)!=receipt_sha:
        raise ValueError('outer campaign receipt pin/scope mismatch')
    report=load_plan(receipt_path)
    directory=Path(report['directory']).resolve()
    if directory!=receipt_path.parent:
        raise ValueError('campaign receipt directory mismatch')
    manifest_path=directory/'manifest.json'
    if digest(manifest_path)!=report['manifest_sha256']:
        raise ValueError('outer campaign manifest mismatch')
    manifest=load_plan(manifest_path)
    from .multifold_run import comparison_variation,comparison_scope,check_control_scope,check_seed_grid
    variation=comparison_variation(manifest);scopes={};control_scopes={}
    input_plans=[]
    for registered in manifest['jobs']:
        plan=load_plan(registered['input_path'])
        input_plans.append(plan)
        sha=comparison_scope(plan,variation)
        check_control_scope(plan,registered['fold_id'],variation,control_scopes)
        fid=registered['fold_id']
        if fid in scopes and scopes[fid]!=sha:raise ValueError('outer same-fold comparison scope mismatch')
        scopes[fid]=sha
    check_seed_grid(manifest,input_plans)
    if report['same_fold_variation_allowed']!=variation or report['same_fold_comparison_scope_sha256']!=scopes:
        raise ValueError('outer comparison contract/scope report mismatch')
    if report['formal_H03_accepted'] is not False or report['outer_performance_evaluated'] is not False:
        raise ValueError('outer receipt claims unsupported acceptance')
    jobs=report['jobs'];expected=manifest['jobs']
    if [(j['job_id'],j['fold_id']) for j in jobs]!=[(j['job_id'],j['fold_id']) for j in expected]:
        raise ValueError('outer campaign job denominator/order mismatch')
    limits=Limits(**manifest['limits'])
    for actual,registered in zip(jobs,expected):
        pipeline=registered.get('pipeline_kind','baseline')
        verify_process(root,actual,registered['input_path'],registered['input_sha256'],limits,pipeline=pipeline)
    reference=report['outer_prediction_ledger'];path=Path(reference['path']).resolve()
    if path.parent!=directory or digest(path)!=reference['sha256']:
        raise ValueError('outer ledger pin/scope mismatch')
    reconstructed=collect(jobs)
    pd.testing.assert_frame_equal(pd.read_parquet(path),reconstructed,check_exact=True)
    if reference['rows']!=len(reconstructed) or reference['recorded_provenance_valid_rows']!=int(reconstructed.recorded_oof_provenance_valid.sum()):
        raise ValueError('outer ledger summary counts mismatch')
    if reference['historical_availability_proven'] is not False:
        raise ValueError('unsupported outer historical availability')
    if report['distinct_outer_folds']!=len({j['fold_id'] for j in expected}):
        raise ValueError('outer fold count mismatch')
    if digest(receipt_path)!=receipt_sha or digest(path)!=reference['sha256'] or digest(manifest_path)!=report['manifest_sha256']:
        raise ValueError('outer campaign changed during verification')
    return dict(rows=len(reconstructed),outer_ledger_reconstructed=True,models_refit=0,
        recorded_process_inputs_verified=True,prediction_algorithm_independently_verified=False,
        historical_availability_proven=False,formal_H03_accepted=False)
