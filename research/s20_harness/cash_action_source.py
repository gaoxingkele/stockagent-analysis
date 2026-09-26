"""Reconstruct source-bound accounting events before portfolio use."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import uuid

import pandas as pd

from .distribution_adapter import adapt
from .portfolio_cash_actions import record_cash_claim
from .runtime import atomic_json, digest, now


def load_ledger(ledger, summary_sha, *, require_code_pin=False):
    """Rebuild every accepted and refused event; no selected-event filtering."""
    ledger = Path(ledger).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", summary_sha or ""):
        raise ValueError("explicit ledger pin required")
    pins = {ledger/"summary.json": summary_sha}

    def check():
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("cash source pin mismatch")

    check()
    summary = json.loads((ledger/"summary.json").read_text(encoding="utf-8"))
    code = Path(__file__).with_name("distribution_adapter.py").resolve()
    if require_code_pin and not summary.get("code_sha256"):
        raise ValueError("ledger adapter code pin required")
    if summary.get("code_sha256"):
        pins[code] = summary["code_sha256"]
    for name in ("source", "review", "unit_review"):
        if summary.get(name+"_path"):
            pins[Path(summary[name+"_path"]).resolve()] = summary[name+"_sha256"]
    for name in ("events", "decisions"):
        pins[ledger/(name+".json")] = summary[name+"_sha256"]
    check()
    frame = pd.read_parquet(summary["source_path"])
    reviews = json.loads(Path(summary["review_path"]).read_text(encoding="utf-8"))["reviews"]
    unit = (json.loads(Path(summary["unit_review_path"]).read_text(encoding="utf-8"))["reviews"]
            if summary.get("unit_review_path") else None)
    # Full reconstruction preserves multi-event group constraints and quote
    # units; selecting one normalized row first would lose those checks.
    events, decisions = adapt(frame, reviews, rate_policy="gross_reference_diagnostic", unit_reviews=unit)
    stored_events = json.loads((ledger/"events.json").read_text(encoding="utf-8"))
    stored_decisions = json.loads((ledger/"decisions.json").read_text(encoding="utf-8"))
    if [asdict(e) for e in events] != stored_events or decisions != stored_decisions:
        raise ValueError("ledger reconstruction mismatch")
    if (len(frame) != summary["source_rows"] or len(events) != summary["accepted_gross_accounting_events"]
            or len(frame)-len(events) != summary["unresolved_events"]):
        raise ValueError("ledger denominator mismatch")
    check()
    return events, decisions, dict(
        source_rows=len(frame), accepted_gross_accounting_events=len(events),
        unresolved_events=len(frame)-len(events), observed_available_at=now(),
        source_pins={str(p): h for p, h in pins.items()},
        full_ledger_reconstructed=True, source_bytes_verified=True,
        availability_basis="present verification, not original historical receipt",
        historical_availability_proven=False, final_personal_tax_proven=False,
        full_corporate_action_coverage_proven=False, formal_training_authorized=False)


def load_event(ledger, summary_sha, event_id, ts_code):
    if not event_id or not ts_code:
        raise ValueError("explicit event/stock identity required")
    events, decisions, evidence = load_ledger(ledger, summary_sha)
    match = [d for d in decisions if d["normalized_event_id"] == event_id]
    if len(match) != 1 or match[0]["ts_code"] != ts_code:
        raise ValueError("event/stock identity mismatch")
    if not match[0]["accepted_for_accounting"]:
        raise ValueError("event remains unresolved")
    selected = next(e for e in events if e.event_id == event_id)
    return selected, dict(evidence, event_id=event_id, ts_code=ts_code, distribution=asdict(selected))


def build(root, ledger, summary_sha, event_id, ts_code):
    _, report = load_event(ledger, summary_sha, event_id, ts_code)
    out = Path(root).resolve()/"output/experiments/s20_safe_v4/sources"/("cash-action-source-"+uuid.uuid4().hex)
    out.mkdir(parents=True)
    report.update(directory=str(out), code_sha256=digest(Path(__file__)),
                  adapter_code_sha256=digest(Path(__file__).with_name("distribution_adapter.py")))
    atomic_json(out/"summary.json", report)
    return report


def record_from_ledger(book, *, ledger, summary_sha, distribution_id, ts_code,
                       event_id, processing_at):
    """Historical replay cannot silently replace today's verification time."""
    distribution, evidence = load_event(ledger, summary_sha, distribution_id, ts_code)
    return record_cash_claim(book, event_id=event_id, ts_code=ts_code,
        distribution=distribution, processing_at=processing_at,
        terms_available_at=evidence["observed_available_at"],
        source_id=str(Path(ledger).resolve()/"summary.json")+"#"+summary_sha)
