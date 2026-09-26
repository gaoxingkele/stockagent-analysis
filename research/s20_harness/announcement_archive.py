"""Source-bound candidate matching and bounded PDF archival, not approval."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
from pathlib import Path
import re
import uuid

import pandas as pd
import requests

from .distribution_adapter import row_fingerprint
from .announcement_discovery import validate_page
from .runtime import atomic_json, digest, now


def match_events(frame, announcements, dates):
    if frame.normalized_event_id.isna().any() or frame.normalized_event_id.duplicated().any():
        raise ValueError("unique event IDs required")
    index = {}
    for a in announcements:
        title = html.unescape(re.sub(r"<[^>]+>", "", a.get("announcementTitle", ""))).strip()
        # Known issuer title variants only. A correction/cancellation notice
        # is not the implementation notice; ambiguity still survives by URL.
        if not re.fullmatch(r".*(?:权益分派|分红派息)实施(?:的)?公告(?:[（(]\d{4}-\d+[）)])?", title):
            continue
        path = a.get("adjunctUrl", "")
        if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/\d+\.[Pp][Dd][Ff]", path):
            raise ValueError("unsafe announcement path")
        key = (str(a["secCode"]), a["query_date"].replace("-", ""))
        index.setdefault(key, set()).add(path)
    result = []
    for _, r in frame.loc[frame.imp_ann_date.isin([d.replace("-", "") for d in dates])].iterrows():
        paths = sorted(index.get((r.ts_code.split(".")[0], r.imp_ann_date), ()))
        result.append(dict(normalized_event_id=r.normalized_event_id, row_sha256=row_fingerprint(r),
                           ts_code=r.ts_code, imp_ann_date=r.imp_ann_date, candidate_paths=paths,
                           status="unique_candidate" if len(paths) == 1 else "unmatched" if not paths else "ambiguous",
                           beneficiary_approved=False))
    return result


def _download(out, relative):
    url = "https://static.cninfo.com.cn/" + relative
    receipt = dict(relative_url=relative, url=url, requested_at=now(), beneficiary_approved=False)
    try:
        with requests.get(url, timeout=(10, 20), allow_redirects=False, stream=True) as response:
            if response.status_code != 200:
                raise ValueError("unexpected HTTP status")
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 20_000_000:
                    raise ValueError("document byte budget exceeded")
                chunks.append(chunk)
        content = b"".join(chunks)
        if not content.startswith(b"%PDF-"):
            raise ValueError("not PDF bytes")
        name = relative.split("/")[-2] + "-" + relative.split("/")[-1]
        destination = out / "pdf" / name
        destination.write_bytes(content)
        receipt.update(status="ARCHIVED_NOT_REVIEWED", received_at=now(), file="pdf/" + name,
                       sha256=digest(destination), bytes=size)
    except Exception as exc:
        receipt.update(status="FAILED", received_at=now(), error_type=type(exc).__name__)
    return receipt


def build(root, normalized_path, normalized_sha, discovery, discovery_sha, *, max_documents=100):
    root, discovery = Path(root).resolve(), Path(discovery).resolve()
    summary_path = discovery / "summary.json"
    pins = [(Path(normalized_path), normalized_sha), (summary_path, discovery_sha)]
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("source pin mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pins += [(discovery / "announcements.json", summary["announcements_sha256"]),
             (discovery / "receipts.json", summary["receipts_sha256"])]
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("discovery changed")
    receipts = json.loads((discovery / "receipts.json").read_text(encoding="utf-8"))
    rebuilt = []
    for receipt in receipts:
        if receipt["status"] != "ACQUIRED_NOT_REVIEWED":
            raise ValueError("failed discovery page")
        p = (discovery / receipt["raw_file"]).resolve()
        if not p.is_relative_to(discovery) or digest(p) != receipt["raw_sha256"]:
            raise ValueError("raw discovery mismatch")
        pins.append((p, receipt["raw_sha256"]))
        payload = json.loads(p.read_bytes())
        rebuilt.extend(dict(query_date=receipt["date"], query_page=receipt["page"], raw_sha256=receipt["raw_sha256"], **a)
                       for a in validate_page(payload)[0])
    announcements = json.loads((discovery / "announcements.json").read_text(encoding="utf-8"))
    if rebuilt != announcements:
        raise ValueError("discovery reconstruction mismatch")
    matches = match_events(pd.read_parquet(normalized_path), announcements, [q["date"] for q in summary["queries"]])
    targets = sorted({r["candidate_paths"][0] for r in matches if r["status"] == "unique_candidate"})
    if not 1 <= max_documents <= 100 or len(targets) > max_documents:
        raise ValueError("document request budget exceeded")
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("input changed during matching")
    out = root / "output/experiments/s20_safe_v4/sources" / ("announcement-archive-" + uuid.uuid4().hex)
    (out / "pdf").mkdir(parents=True)
    atomic_json(out / "matches.json", matches)
    atomic_json(out / "inputs.json", [{"path": str(p.resolve()), "sha256": sha} for p, sha in pins])
    atomic_json(out / "plan.json", dict(targets=targets, max_documents=max_documents, workers=4, code_sha256=digest(Path(__file__))))
    acquired = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_download, out, target) for target in targets]
        for future in as_completed(futures):
            acquired.append(future.result())
            atomic_json(out / "receipts.json", sorted(acquired, key=lambda r:r["relative_url"]))
    from collections import Counter
    report = dict(directory=str(out), event_rows=len(matches), match_states=dict(Counter(r["status"] for r in matches)),
                  requested_documents=len(targets), archive_states=dict(Counter(r["status"] for r in acquired)),
                  matches_sha256=digest(out / "matches.json"), receipts_sha256=digest(out / "receipts.json"),
                  formal_event_acceptance=False, historical_availability_proven=False)
    atomic_json(out / "summary.json", report)
    return report
