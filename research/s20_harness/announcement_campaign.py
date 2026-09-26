"""Bounded sequential disclosure batches with durable stage receipts.

This is source preparation, not a formal H01 executor. No automatic retries or
resumption: a live/uncertain child must be reconciled before another campaign.
"""
import json
import os
from pathlib import Path
import uuid

import pandas as pd

from . import announcement_discovery, announcement_archive, announcement_extract
from . import announcement_adjudicate, review_merge, distribution_adapter
from .runtime import atomic_json, digest, now


def run(root, dates, normalized_path, normalized_sha, review_path, review_sha,
        unit_path, unit_sha):
    if not 1 <= len(dates) <= 3 or len(set(dates)) != len(dates):
        raise ValueError("one to three unique announcement dates required")
    for date in dates:
        if pd.Timestamp(date).strftime("%Y-%m-%d") != date:
            raise ValueError("ISO date required")
    root, normalized_path, review_path, unit_path = map(lambda p: Path(p).resolve(),
                                                       [root, normalized_path, review_path, unit_path])
    pins = [(normalized_path, normalized_sha), (review_path, review_sha), (unit_path, unit_sha)]
    pins.extend((p, digest(p)) for p in sorted(Path(__file__).parent.glob("*.py")))
    def check():
        if any(digest(p) != sha for p, sha in pins):
            raise ValueError("campaign source/code changed")
    check()
    out = root / "output/experiments/s20_safe_v4/sources" / ("announcement-campaign-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out / "plan.json", dict(dates=dates, max_pages_per_date=10, max_documents_per_date=100,
                                       max_download_requests=100*len(dates), at=now(),
                                       pins=[dict(path=str(p), sha256=s) for p,s in pins],
                                       formal_training_authorized=False))
    records = []
    state = dict(status="RUNNING", pid=os.getpid(), at=now(), date=None, stage=None)
    def publish_state():
        atomic_json(out / "state.json", dict(state, at=now()))
    def stage(name, call):
        state["stage"] = name
        publish_state()
        check()
        result = call()
        child = Path(result["directory"])
        sha = digest(child / "summary.json")
        # Keep completed child receipts even when input checks subsequently fail.
        records.append(dict(date=state["date"], stage=name, directory=str(child), summary_sha256=sha))
        atomic_json(out / "stages.json", records)
        check()
        return result, child, sha
    publish_state()
    try:
        for date in dates:
            state["date"] = date
            result, discovery, sha = stage("discovery", lambda: announcement_discovery.collect(root, [date]))
            if not all(q["complete_search"] for q in result["queries"]):
                raise ValueError("incomplete discovery; no downstream execution")
            _, archive, archive_sha = stage("archive", lambda: announcement_archive.build(
                root, normalized_path, normalized_sha, discovery, sha, max_documents=100))
            _, extraction, extraction_sha = stage("extract", lambda: announcement_extract.build(
                root, archive, archive_sha, normalized_path, normalized_sha))
            judgment, judged, _ = stage("adjudicate", lambda: announcement_adjudicate.build(
                root, extraction, extraction_sha, normalized_path, normalized_sha))
            merged, merged_path, _ = stage("merge", lambda: review_merge.build(
                root, normalized_path, normalized_sha, [(review_path, review_sha),
                (judged / "reviews.json", judgment["artifacts"]["reviews.json"])]))
            review_path, review_sha = merged_path / "reviews.json", merged["artifacts"]["reviews.json"]
            pins.append((review_path, review_sha))
        ledger, ledger_path, ledger_sha = stage("accounting", lambda: distribution_adapter.run(
            root, normalized_path, review_path, unit_review_path=unit_path))
        report = dict(directory=str(out), dates=dates, stages=len(records),
                      final_review_path=str(review_path), final_review_sha256=review_sha,
                      accounting_path=str(ledger_path), accounting_summary_sha256=ledger_sha,
                      accepted_gross_accounting_events=ledger["accepted_gross_accounting_events"],
                      unresolved_events=ledger["unresolved_events"], formal_training_authorized=False,
                      artifacts={n:digest(out/n) for n in ["plan.json", "stages.json"]})
        atomic_json(out / "summary.json", report)
        state.update(status="COMPLETED_SOURCE_PREPARATION", summary_sha256=digest(out/"summary.json"))
        publish_state()
        return report
    except Exception as exc:
        state.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        publish_state()
        raise
