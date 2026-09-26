"""Bound paired diagnostics of two saved joint policies, not formal G3."""
from pathlib import Path
import uuid
import pandas as pd
from .joint_endpoints import verify_evaluation
from .paired_comparison import compare
from .runtime import load_plan,atomic_json,digest,now


def load(root,ref):
    if not isinstance(ref,dict) or set(ref)!={'directory','summary_sha256'}:
        raise ValueError('exact pinned joint endpoint reference required')
    directory=Path(ref['directory']).resolve()
    validation=verify_evaluation(root,directory,ref['summary_sha256'])
    binding=load_plan(directory/'bindings.json');payload=load_plan(directory/'input.json')
    plan=load_plan(Path(binding['run_directory'])/'input.json')
    saved=pd.read_parquet(directory/'evaluated_endpoints.parquet')
    rows=saved[['sample_id','entity_id','signal_date','selected']].copy()
    rows['safe_profit']=saved.safe_profit_target
    rows['risk10']=saved.risk10_target
    return rows,plan,payload,validation


def compare_bound(root,candidate_ref,baseline_ref,*,draws=2000,seed=20):
    left,lp,lo,lv=load(root,candidate_ref)
    right,rp,ro,rv=load(root,baseline_ref)
    # This adapter studies policy differences on the same joint experiment.
    # Cross-target or different feature/model searches need a separate scope
    # contract rather than silently weakening this one.
    if {k:v for k,v in lp.items() if k!='policy'}!={k:v for k,v in rp.items() if k!='policy'}:
        raise ValueError('paired joint experiment scope mismatch')
    if lo!=ro:raise ValueError('paired joint endpoint snapshot/cutoff mismatch')
    result=compare(left,right,lp['calendar'],draws=draws,seed=seed)
    verify_evaluation(root,candidate_ref['directory'],candidate_ref['summary_sha256'])
    verify_evaluation(root,baseline_ref['directory'],baseline_ref['summary_sha256'])
    return dict(candidate_reference=candidate_ref,baseline_reference=baseline_ref,
        endpoint_validation={'candidate':lv,'baseline':rv},comparison=result,
        saved_maturity_and_selection_recomputed=True,shared_joint_experiment_verified=True,
        shared_endpoint_snapshot_verified=True,model_or_feature_search_compared=False,
        policy_choice_preregistered_proven=False,market_label_semantics_proven=False,
        formal_G3_passed=False,models_refit=0)


def _execute(root,plan):
    if plan.get('schema_version')=='2':
        if set(plan)!={'schema_version','folds','draws','seed'}:
            raise ValueError('exact multi-fold paired plan required')
        return aggregate_bound(root,plan['folds'],draws=plan['draws'],seed=plan['seed'])
    if set(plan)!={'schema_version','candidate','baseline','draws','seed'} or plan['schema_version']!='1':
        raise ValueError('exact paired comparison plan required')
    return compare_bound(root,plan['candidate'],plan['baseline'],draws=plan['draws'],seed=plan['seed'])


def aggregate_bound(root,folds,*,draws=2000,seed=20):
    from .fold_comparison import aggregate
    from .label_availability import _instant
    if not isinstance(folds,list) or not 1<=len(folds)<=10 or any(
        not isinstance(f,dict) or set(f)!={'fold_id','candidate','baseline'} for f in folds):
        raise ValueError('bounded exact paired fold references required')
    tables=[];bindings=[];clock=None;evidence=None;seen_dates=set()
    for fold in folds:
        left,lp,lo,lv=load(root,fold['candidate']);right,rp,ro,rv=load(root,fold['baseline'])
        if {k:v for k,v in lp.items() if k!='policy'}!={k:v for k,v in rp.items() if k!='policy'}:
            raise ValueError('paired fold joint experiment scope mismatch')
        if lo!=ro:raise ValueError('paired fold endpoint snapshot mismatch')
        current_clock=_instant(lo['evaluation_at'])
        if clock is not None and (current_clock!=clock or lp['evidence_mode']!=evidence):
            raise ValueError('cross-fold evaluation clock/evidence mismatch')
        clock=current_clock;evidence=lp['evidence_mode']
        if seen_dates&set(lp['calendar']):raise ValueError('overlapping outer dates; seeds are not new evidence')
        seen_dates.update(lp['calendar'])
        tables.append(dict(fold_id=fold['fold_id'],calendar=lp['calendar'],candidate=left,baseline=right))
        bindings.append(dict(fold_id=fold['fold_id'],candidate=fold['candidate'],baseline=fold['baseline'],
                             candidate_validation=lv,baseline_validation=rv))
    result=aggregate(tables,draws=draws,seed=seed)
    for fold in folds:
        for side in ['candidate','baseline']:
            ref=fold[side];verify_evaluation(root,ref['directory'],ref['summary_sha256'])
    return dict(fold_bindings=bindings,aggregation=result,evaluation_at=clock.isoformat(),evidence_mode=evidence,
        models_refit=0,cross_fold_policy_lineage_preregistered=False,
        market_label_semantics_proven=False,formal_G3_passed=False)


def _code():
    return {str(p.resolve()):digest(p) for p in Path(__file__).parent.glob('*.py')}


def build(root,input_path,input_sha):
    source=Path(input_path).resolve()
    if digest(source)!=input_sha:raise ValueError('paired comparison input pin mismatch')
    pins={str(source):input_sha,**_code()};plan=load_plan(source)
    result=_execute(root,plan)
    if any(digest(Path(p))!=h for p,h in pins.items()):raise ValueError('paired comparison source changed')
    out=Path(root).resolve()/'output/experiments/s20_safe_v4/sources'/('joint-paired-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'input.json',plan);atomic_json(out/'result.json',result)
    atomic_json(out/'bindings.json',dict(input_path=str(source),input_sha256=input_sha,source_pins=pins))
    report=dict(directory=str(out),at=now(),models_refit=0,formal_G3_passed=False,
        registered_search_budget_proven=False,
        artifacts={n:digest(out/n) for n in ['input.json','result.json','bindings.json']})
    if any(digest(Path(p))!=h for p,h in pins.items()):raise ValueError('paired comparison source changed')
    atomic_json(out/'summary.json',report)
    return report


def verify(root,directory,summary_sha):
    directory=Path(directory).resolve();summary=directory/'summary.json'
    if not directory.is_relative_to(Path(root).resolve()/'output/experiments/s20_safe_v4/sources') or digest(summary)!=summary_sha:
        raise ValueError('paired comparison pin/scope mismatch')
    report=load_plan(summary)
    if report['directory']!=str(directory) or set(report['artifacts'])!={'input.json','result.json','bindings.json'}:
        raise ValueError('paired comparison artifact contract mismatch')
    pins={str(summary):summary_sha,**{str(directory/n):h for n,h in report['artifacts'].items()}}
    def check():
        if any(digest(Path(p))!=h for p,h in pins.items()):raise ValueError('paired comparison artifact/source mismatch')
    check();binding=load_plan(directory/'bindings.json')
    source=Path(binding['input_path']).resolve()
    if binding['source_pins'].get(str(source))!=binding['input_sha256']:
        raise ValueError('paired comparison original input not bound')
    if {p:h for p,h in binding['source_pins'].items() if Path(p).suffix=='.py'}!=_code():
        raise ValueError('paired comparison code inventory mismatch')
    for p,h in binding['source_pins'].items():
        if p in pins and pins[p]!=h:raise ValueError('conflicting paired comparison pins')
        pins[p]=h
    check()
    plan=load_plan(directory/'input.json')
    if plan!=load_plan(source):raise ValueError('paired comparison saved/original input mismatch')
    result=_execute(root,plan)
    if load_plan(directory/'result.json')!=result:raise ValueError('paired comparison reconstruction mismatch')
    expected=dict(models_refit=0,formal_G3_passed=False,registered_search_budget_proven=False)
    if any(report.get(k)!=v for k,v in expected.items()):raise ValueError('paired comparison unsupported summary claims')
    check()
    return dict(comparison_recomputed=True,models_refit=0,formal_G3_passed=False)
