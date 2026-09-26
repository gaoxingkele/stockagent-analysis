"""Bind suspension reviews to archived PDF identity and searchable page text.

Identity matches are not automated proof of the review's interpretation.
"""
import hashlib
from pathlib import Path
import re
import uuid

import pymupdf

from .runtime import atomic_json, digest, load_plan, now
from .label_availability import _instant


def identity_pages(pages, code, notice):
    normalized = [re.sub(r'\s+', '', text) for text in pages]
    code_pattern = r'证券代码：'+re.escape(code.split('.')[0])+r'(?!\d)'
    notice_pattern = r'公告编号：'+re.escape(notice)+r'(?!\d)'
    code_hits = [i+1 for i, text in enumerate(normalized) if re.search(code_pattern, text)]
    notice_hits = [i+1 for i, text in enumerate(normalized) if re.search(notice_pattern, text)]
    return dict(code_pages=code_hits, notice_pages=notice_hits,
                document_identity_matched=bool(set(code_hits)&set(notice_hits)))


def build(root, archive, archive_sha):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if digest(archive/'summary.json') != archive_sha:
        raise ValueError('notice archive summary mismatch')
    summary = load_plan(archive/'summary.json')
    pins = {str(archive/'summary.json'):archive_sha,
            str(archive/'plan.json'):summary['plan_sha256'],
            str(archive/'receipts.json'):summary['receipts_sha256']}
    def check():
        if any(digest(Path(p))!=h for p,h in pins.items()):
            raise ValueError('notice binding source changed')
    check()
    plan=load_plan(archive/'plan.json')
    review_path=Path(plan['review_path'])
    pins[str(review_path)]=plan['review_sha256']
    pins[str(Path(__file__).resolve())]=digest(Path(__file__))
    check()
    import json
    receipts=json.loads((archive/'receipts.json').read_text(encoding='utf-8'))
    if sorted(r['url'] for r in receipts)!=sorted(plan['urls']) or len(set(plan['urls']))!=len(plan['urls']):
        raise ValueError('notice receipt URL denominator mismatch')
    review=load_plan(review_path)
    if sorted({c['source_url'] for c in review['reviewed_cases']})!=sorted(plan['urls']):
        raise ValueError('review URL binding mismatch')
    documents={};receipt_map={r['url']:r for r in receipts}
    for receipt in receipts:
        if _instant(receipt['received_at'])<_instant(receipt['requested_at']):
            raise ValueError('reversed notice acquisition times')
        if receipt['status']!='ARCHIVED_NOT_REVIEWED':
            continue
        path=(archive/receipt['file']).resolve()
        if not path.is_relative_to(archive) or digest(path)!=receipt['sha256']:
            raise ValueError('notice PDF binding mismatch')
        pins[str(path)]=receipt['sha256']
        with pymupdf.open(path) as doc:
            documents[receipt['url']]=[p.get_text() for p in doc]
    cases=[]
    for case in review['reviewed_cases']:
        receipt=receipt_map[case['source_url']];pages=documents.get(case['source_url'],[])
        result=identity_pages(pages,case['ts_code'],case['source_notice'])
        cases.append(dict(ts_code=case['ts_code'],trade_date=case['trade_date'],
            source_url=case['source_url'],source_notice=case['source_notice'],
            pdf_sha256=receipt.get('sha256'),received_at=receipt['received_at'],host_role=receipt['host_role'],
            **result,reviewed_finding=case['semantic_finding'],scope_limit=case['scope_limit'],
            original_review_state=case['retrospective_state'],
            semantic_finding_automatically_verified=False,historical_prediction_eligible=False))
    check()
    out=root/'output/experiments/s20_safe_v4/sources'/('suspension-notice-binding-'+uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out/'cases.json',cases)
    atomic_json(out/'pages.json',[dict(url=url,pages=[dict(page=i+1,text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest()) for i,text in enumerate(pages)])
        for url,pages in sorted(documents.items())])
    atomic_json(out/'inputs.json',pins)
    report=dict(directory=str(out),at=now(),review_cases=len(cases),parsed_documents=len(documents),
        identity_matched=sum(c['document_identity_matched'] for c in cases),
        parser_version=pymupdf.VersionBind,semantic_acceptance=False,historical_availability_proven=False,
        formal_training_authorized=False,artifacts={n:digest(out/n) for n in ['cases.json','pages.json','inputs.json']})
    atomic_json(out/'summary.json',report)
    return report
