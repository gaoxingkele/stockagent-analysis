"""Bounded archival of reviewed suspension URLs; retrieval is not acceptance."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
import uuid

import requests

from .runtime import atomic_json, digest, load_plan, now

HOSTS = {'static.cninfo.com.cn', 'star.sse.com.cn', 'disc.static.szse.cn', 'stockmc.xueqiu.com'}


def targets(review):
    cases = review['reviewed_cases']
    keys = [(r['ts_code'], r['trade_date']) for r in cases]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate notice review identity')
    urls = sorted({r['source_url'] for r in cases})
    if not 1 <= len(urls) <= 20:
        raise ValueError('notice request budget exceeded')
    for url in urls:
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.port is not None
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not parsed.path.lower().endswith('.pdf') or '..' in parsed.path.split('/')):
            raise ValueError('unapproved notice URL')
    return urls


def acquire(out, url):
    receipt = dict(url=url, requested_at=now(), historical_availability_proven=False,
                   semantic_review_accepted=False, host_role='mirror' if urlsplit(url).hostname=='stockmc.xueqiu.com' else 'disclosure_host')
    try:
        with requests.get(url, timeout=(10, 20), allow_redirects=False, stream=True) as response:
            receipt['http_status'] = response.status_code
            if response.status_code != 200:
                raise ValueError('unexpected HTTP status')
            chunks=[];size=0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 20_000_000:
                    raise ValueError('notice byte budget exceeded')
                chunks.append(chunk)
        payload=b''.join(chunks)
        if not payload.startswith(b'%PDF-'):
            raise ValueError('not PDF bytes')
        name=hashlib.sha256(url.encode()).hexdigest()+'.pdf'
        path=out/'pdf'/name
        path.write_bytes(payload)
        receipt.update(status='ARCHIVED_NOT_REVIEWED', file='pdf/'+name, sha256=digest(path), bytes=size)
    except (requests.RequestException, ValueError, OSError) as exc:
        receipt.update(status='FAILED', error_type=type(exc).__name__)
    receipt['received_at']=now()
    return receipt


def build(root, review_path, review_sha):
    root, review_path=Path(root).resolve(),Path(review_path).resolve()
    if digest(review_path)!=review_sha:
        raise ValueError('notice review pin mismatch')
    review=load_plan(review_path);urls=targets(review)
    code_sha=digest(Path(__file__))
    out=root/'output/experiments/s20_safe_v4/sources'/('suspension-notice-archive-'+uuid.uuid4().hex)
    (out/'pdf').mkdir(parents=True)
    atomic_json(out/'plan.json',dict(review_path=str(review_path),review_sha256=review_sha,
        urls=urls,code_sha256=code_sha,workers=3,max_documents=20))
    receipts=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(acquire,out,url) for url in urls]):
            receipts.append(future.result())
            atomic_json(out/'receipts.json',sorted(receipts,key=lambda r:r['url']))
    if digest(review_path)!=review_sha or digest(Path(__file__))!=code_sha:
        raise ValueError('notice archive inputs changed')
    report=dict(directory=str(out),review_cases=len(review['reviewed_cases']),requested_documents=len(urls),
        states=dict(Counter(r['status'] for r in receipts)),plan_sha256=digest(out/'plan.json'),
        receipts_sha256=digest(out/'receipts.json'),historical_availability_proven=False,
        formal_training_authorized=False,semantic_review_accepted=False)
    atomic_json(out/'summary.json',report)
    return report
