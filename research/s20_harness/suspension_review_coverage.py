"""Retained-row coverage of reviewed notices, separate from raw event semantics."""
import json
from pathlib import Path
import uuid

import pandas as pd

from .runtime import atomic_json, digest, load_plan, now


def attach(overlay, cases):
    keys=['ts_code','trade_date']
    added=['notice_support_status','notice_direct_support','notice_documents_json']
    if set(added)&set(overlay): raise ValueError('notice coverage already attached')
    case_keys=[tuple(c[k] for k in keys) for c in cases]
    if len(case_keys)!=len(set(case_keys)): raise ValueError('duplicate notice coverage case')
    rows=[]
    for case in cases:
        selected=overlay.loc[overlay.ts_code.eq(case['ts_code']) & overlay.trade_date.eq(case['trade_date'])]
        if selected.empty or not selected.review_state.eq(case['original_review_state']).all():
            raise ValueError('notice case absent or contradicts raw review overlay')
        if case['historical_prediction_eligible'] is not False or case['executable_fill_proven'] is not False:
            raise ValueError('unsupported notice eligibility')
        rows.append({**{k:case[k] for k in keys},'notice_support_status':case['support_status'],
            'notice_direct_support':case['retrospective_direct_support'],
            'notice_documents_json':json.dumps(case['documents'],ensure_ascii=False,sort_keys=True)})
    expected=set(map(tuple,overlay.loc[overlay.review_state.ne('not_event_reviewed'),keys].drop_duplicates().to_numpy()))
    if set(case_keys)!=expected: raise ValueError('reviewed-case coverage denominator mismatch')
    result=overlay.merge(pd.DataFrame(rows,columns=keys+added),on=keys,how='left',sort=False,validate='many_to_one')
    pd.testing.assert_frame_equal(result[overlay.columns],overlay.reset_index(drop=True),check_exact=True)
    result['notice_support_status']=result.notice_support_status.fillna('not_notice_reviewed')
    result['notice_direct_support']=pd.array(result.notice_direct_support,dtype='boolean')
    return result


def build(root):
    from .suspension_daily_verify import verify
    from .suspension_evidence_merge import build as merge_findings
    from .suspension_semantics import resolve
    root=Path(root).resolve()
    inventory_path=root/'config/s20_v4_data_sources.json'
    review_path=root/'config/s20_v4_daily_suspension_reviews.json'
    pins={str(p):digest(p) for p in [inventory_path,review_path,Path(__file__).resolve()]}
    inventory=load_plan(inventory_path)
    sources=[r for r in inventory['sources'] if r['role']=='daily_suspension_events']
    if len(sources)!=1: raise ValueError('unique suspension source required')
    source=sources[0];audit_path=root/source['audit']
    verified=verify(root,audit_path.parent,source['audit_sha256'],root/source['path'])
    pins.update({r['path']:r['sha256'] for r in verified.pop('evidence_files')})
    audit_summary=load_plan(audit_path)
    raw=pd.read_parquet(audit_path.parent/'rows.parquet')
    overlay=resolve(raw,load_plan(review_path),audit_summary['rows_sha256'],now())
    merged=merge_findings(root,root/'config/s20_v4_suspension_notice_findings.json',
                         [root/'config/s20_v4_suspension_supplemental_findings.json'])
    member=Path(merged['directory'])
    cases=json.loads((member/'cases.json').read_text(encoding='utf-8'))
    result=attach(overlay,cases)
    if any(digest(Path(p))!=h for p,h in pins.items()): raise ValueError('coverage input changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('suspension-review-coverage-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    result.to_parquet(out/'rows.parquet',index=False)
    atomic_json(out/'inputs.json',dict(source_pins=pins,merged_findings=str(member),
        merged_summary_sha256=digest(member/'summary.json')))
    report=dict(directory=str(out),at=now(),rows=len(result),reviewed_cases=len(cases),
        status_counts=result.notice_support_status.value_counts().to_dict(),
        direct_supported_rows=int(result.notice_direct_support.fillna(False).sum()),
        unknown_notice_rows=int(result.notice_direct_support.isna().sum()),
        all_raw_rows_preserved=True,formal_H01_gate_passed=False,
        historical_availability_proven=False,actual_fills_proven=False,
        artifacts={n:digest(out/n) for n in ['rows.parquet','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
