"""Retain per-document findings when adding corroboration to case coverage."""


def merge(primary, supplements):
    keys=[(r['ts_code'],r['trade_date']) for r in primary]
    if len(set(keys))!=len(keys):
        raise ValueError('duplicate original case')
    grouped={key:[row] for key,row in zip(keys,primary)}
    for rows in supplements:
        seen=set()
        for row in rows:
            key=(row['ts_code'],row['trade_date'])
            if key not in grouped or key in seen:
                raise ValueError('supplement outside original cases or duplicated')
            seen.add(key)
            if row['original_review_state']!=grouped[key][0]['original_review_state']:
                raise ValueError('conflicting retrospective states require explicit review')
            if row['source_url'] in {r['source_url'] for r in grouped[key]}:
                raise ValueError('duplicate source is not corroboration')
            grouped[key].append(row)
    result=[]
    for key in keys:
        rows=grouped[key]
        for row in rows:
            if row.get('page_evidence_bound') is not True:
                raise ValueError('unbound page evidence')
            if row.get('historical_prediction_eligible') is not False or row.get('executable_fill_proven') is not False:
                raise ValueError('unsupported eligibility claim')
        direct=[r for r in rows if r['basis']=='direct_disclosure' and r['host_role']=='disclosure_host']
        result.append(dict(ts_code=key[0],trade_date=key[1],
            original_review_state=rows[0]['original_review_state'],documents=rows,
            retrospective_direct_support=bool(direct),
            original_document_basis=rows[0]['basis'],
            support_status='direct_disclosure_supported' if direct else 'interval_or_mirror_unresolved',
            historical_prediction_eligible=False,executable_fill_proven=False))
    return result


def build(root, primary_spec, supplement_specs):
    """Freshly validate each source finding before merging; no status override."""
    import json
    from pathlib import Path
    import uuid
    from .suspension_notice_findings import build as build_findings
    from .runtime import atomic_json, digest, now
    root=Path(root).resolve()
    specs=[Path(primary_spec).resolve()]+[Path(p).resolve() for p in supplement_specs]
    pins={str(p):digest(p) for p in specs}
    pins[str(Path(__file__).resolve())]=digest(Path(__file__))
    members=[];tables=[]
    for spec in specs:
        report=build_findings(root,spec)
        directory=Path(report['directory'])
        members.append(dict(directory=str(directory),summary_sha256=digest(directory/'summary.json')))
        tables.append(json.loads((directory/'cases.json').read_text(encoding='utf-8')))
    cases=merge(tables[0],tables[1:])
    if any(digest(Path(p))!=h for p,h in pins.items()):
        raise ValueError('merge source changed')
    out=root/'output/experiments/s20_safe_v4/sources'/('suspension-evidence-merge-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'cases.json',cases)
    atomic_json(out/'inputs.json',dict(source_pins=pins,members=members))
    report=dict(at=now(),directory=str(out),cases=len(cases),
        direct_supported=sum(r['retrospective_direct_support'] for r in cases),
        unresolved=sum(not r['retrospective_direct_support'] for r in cases),
        original_document_findings_preserved=True,formal_H01_gate_passed=False,
        historical_prediction_eligible=False,executable_fill_proven=False,
        artifacts={n:digest(out/n) for n in ['cases.json','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
