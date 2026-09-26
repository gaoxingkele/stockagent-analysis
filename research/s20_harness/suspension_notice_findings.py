"""Validate explicit retrospective case reviews against bound page evidence."""
import json
from pathlib import Path
import re
import uuid

from .runtime import atomic_json, digest, load_plan, now


def validate(cases, pages, findings):
    keys=lambda rows:[(r['ts_code'],r['trade_date']) for r in rows]
    if keys(cases)!=keys(findings) or len(set(keys(cases)))!=len(cases):
        raise ValueError('notice findings denominator/order mismatch')
    texts={p['url']:{r['page']:re.sub(r'\s+','',r['text']) for r in p['pages']} for p in pages}
    result=[]
    for original,review in zip(cases,findings):
        basis=review['basis']
        if basis not in {'direct_disclosure','interval_inference','mirror_pending_primary'}:
            raise ValueError('unknown interpretation basis')
        if original['host_role']=='mirror' and basis!='mirror_pending_primary':
            raise ValueError('mirror cannot be promoted to primary disclosure')
        if not original['document_identity_matched'] or not review['finding'].strip() or not review['limit'].strip() or not review['evidence']:
            raise ValueError('missing case identity/finding/evidence')
        for evidence in review['evidence']:
            anchor=evidence['anchor']
            if not anchor or anchor not in texts.get(original['source_url'],{}).get(evidence['page'],''):
                raise ValueError('case evidence anchor absent')
        result.append(dict(original,**{k:review[k] for k in ['basis','finding','limit','evidence']},
            page_evidence_bound=True,retrospective_direct_disclosure_supported=basis=='direct_disclosure',
            historical_prediction_eligible=False,executable_fill_proven=False))
    return result


def build(root, findings_path):
    root,findings_path=Path(root).resolve(),Path(findings_path).resolve()
    spec_sha=digest(findings_path);spec=load_plan(findings_path)
    if spec['historical_availability_proven'] is not False or spec['automatic_trading_override'] is not False:
        raise ValueError('unsupported notice authority')
    directory=(root/spec['binding_directory']).resolve()
    if not directory.is_relative_to(root) or digest(directory/'summary.json')!=spec['binding_summary_sha256']:
        raise ValueError('notice binding pin mismatch')
    summary=load_plan(directory/'summary.json')
    pins={str(findings_path):spec_sha,str(directory/'summary.json'):spec['binding_summary_sha256']}
    for name,sha in summary['artifacts'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory): raise ValueError('binding artifact escapes directory')
        pins[str(path)]=sha
    def check():
        if any(digest(Path(p))!=sha for p,sha in pins.items()): raise ValueError('notice finding input changed')
    check()
    # Original code blobs are historical provenance, not current executable pins.
    for p,sha in load_plan(directory/'inputs.json').items():
        if Path(p).suffix!='.py': pins[p]=sha
    pins[str(Path(__file__).resolve())]=digest(Path(__file__))
    check()
    cases=json.loads((directory/'cases.json').read_text(encoding='utf-8'))
    pages=json.loads((directory/'pages.json').read_text(encoding='utf-8'))
    result=validate(cases,pages,spec['cases'])
    check()
    out=root/'output/experiments/s20_safe_v4/sources'/('suspension-notice-findings-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'cases.json',result);atomic_json(out/'inputs.json',pins)
    from collections import Counter
    report=dict(directory=str(out),at=now(),cases=len(result),basis_counts=dict(Counter(r['basis'] for r in result)),
        page_evidence_bound=True,formal_H01_gate_passed=False,historical_availability_proven=False,
        actual_fills_proven=False,artifacts={n:digest(out/n) for n in ['cases.json','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
