"""Assemble H02 diagnostic deliverables without bypassing formal H01 admission."""
from pathlib import Path
import uuid

import pandas as pd

from .dataset_collection import verify_collection
from . import opportunity_partition,execution_golden,stock_session_compat
from .runtime import atomic_json,digest,load_plan,now


def join_tracks(samples,opportunity,compatibility=None):
    keys=['sample_id','entity_id','signal_date']
    if samples[keys].to_dict('records')!=opportunity[keys].to_dict('records'):
        raise ValueError('H02 O/P candidate identity or order mismatch')
    if not samples.p_class.fillna('UNKNOWN').eq(opportunity.p_class.fillna('UNKNOWN')).all():
        raise ValueError('H02 original P classes differ')
    result=samples.copy()
    for name in ['o_class','o_label_realized','o_reason','o_safe_opportunity','o_payload_json']:
        result[name]=opportunity[name].array
    if compatibility is not None:
        if samples[keys].to_dict('records')!=compatibility[keys].to_dict('records'):
            raise ValueError('H02 compatibility candidate identity or order mismatch')
        for name in ['compat_class','compat_reason','compat_entry_date','compat_horizon_end','compat_payload_json','different_horizon',
                     'compat_label_realized','raw_calendar_class','raw_calendar_reason','raw_calendar_payload_json']:
            result[name]=compatibility[name].array
    return result


def build(root,collection,collection_sha,plan_path,plan_sha):
    root,collection,plan_path=Path(root).resolve(),Path(collection).resolve(),Path(plan_path).resolve()
    if digest(plan_path)!=plan_sha: raise ValueError('H02 plan pin mismatch')
    plan=load_plan(plan_path)
    samples,validation=verify_collection(root,collection,collection_sha)
    members=load_plan(collection/'inputs.json')['assemblies']
    rows=[];op_bindings=[];compat_rows=[];compat_bindings=[]
    for member in members:
        inputs=load_plan(Path(member['directory'])/'inputs.json')
        report=opportunity_partition.build(root,inputs['label_directory'],inputs['label_summary_sha256'])
        directory=Path(report['directory'])
        rows.append(pd.read_parquet(directory/'labels.parquet'))
        op_bindings.append(dict(directory=str(directory),summary_sha256=digest(directory/'summary.json')))
        compat=stock_session_compat.build(root,inputs['label_directory'],inputs['label_summary_sha256'])
        compat_dir=Path(compat['directory'])
        compat_rows.append(pd.read_parquet(compat_dir/'labels.parquet'))
        compat_bindings.append(dict(directory=str(compat_dir),summary_sha256=digest(compat_dir/'summary.json')))
    joined=join_tracks(samples,pd.concat(rows,ignore_index=True),pd.concat(compat_rows,ignore_index=True))
    golden=execution_golden.build(root);golden_dir=Path(golden['directory'])
    golden_sha=digest(golden_dir/'summary.json')
    golden_check=execution_golden.verify(golden_dir,golden_sha)
    # Revalidate the entire input collection after O computation.
    verified,_=verify_collection(root,collection,collection_sha)
    pd.testing.assert_frame_equal(samples,verified,check_exact=True)
    if digest(plan_path)!=plan_sha: raise ValueError('H02 plan changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('h02-preparation-'+uuid.uuid4().hex);out.mkdir(parents=True)
    joined.to_parquet(out/'labels.parquet',index=False)
    pd.crosstab(joined.o_reason,joined.p_class.fillna('UNKNOWN'),dropna=False).to_csv(out/'label_transfer.csv')
    pd.crosstab(joined.compat_reason,joined.o_reason,dropna=False).to_csv(out/'compatibility_transfer.csv')
    pd.crosstab(joined.compat_reason,joined.raw_calendar_reason,dropna=False).to_csv(out/'calendar_effect_transfer.csv')
    pd.crosstab(joined.raw_calendar_reason,joined.o_reason,dropna=False).to_csv(out/'event_treatment_transfer.csv')
    atomic_json(out/'label_contracts.json',dict(frozen_contracts=plan['label_tracks'],
        actual_tracks=['raw stock-session O formula diagnostic','raw calendar O diagnostic','calendar O diagnostic','gross-cash P diagnostic'],
        stock_session_formula_computed=True,legacy_model_or_cache_reproduced=False,formal_training_authorized=False))
    atomic_json(out/'execution_contract.json',dict(contract=plan['label_tracks']['execution_book_v4'],
        actual_execution_evidence=False,formal_training_authorized=False))
    atomic_json(out/'golden_cases.json',load_plan(golden_dir/'golden_cases.json'))
    atomic_json(out/'inputs.json',dict(collection=str(collection),collection_sha256=collection_sha,
        plan_path=str(plan_path),plan_sha256=plan_sha,opportunity_partitions=op_bindings,
        compatibility_partitions=compat_bindings,
        golden_directory=str(golden_dir),golden_summary_sha256=golden_sha,code_sha256=digest(Path(__file__))))
    gaps=['H01 source admission has not passed','historical availability not established',
          'legacy model/cache baseline not reproduced; compatibility formula alone is insufficient',
          'economic event coverage unproved','real execution/market rules and full golden coverage incomplete']
    report=dict(directory=str(out),at=now(),rows=len(joined),dates=sorted(joined.signal_date.unique().tolist()),
        all_candidates_retained=True,collection_validation=validation,golden_validation=golden_check,
        o_resolved_rows=int(joined.o_label_realized.sum()),o_safe_rows=int(joined.o_safe_opportunity.fillna(False).sum()),
        compatibility_unknown_rows=int((~joined.compat_label_realized).sum()),different_horizon_rows=int(joined.different_horizon.sum()),
        formal_gate_passed=False,formal_training_authorized=False,status='DIAGNOSTIC_PREPARATION',acceptance_gaps=gaps,
        artifacts={n:digest(out/n) for n in ['labels.parquet','label_transfer.csv','label_contracts.json',
            'execution_contract.json','golden_cases.json','inputs.json','compatibility_transfer.csv',
            'calendar_effect_transfer.csv','event_treatment_transfer.csv']})
    atomic_json(out/'summary.json',report)
    return report
