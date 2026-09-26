"""Source-bound, denominator-preserving event refusal audit; not H01 approval."""
import json
from collections import Counter
from pathlib import Path
import uuid

import pandas as pd

from .label_partition_verify import verify
from .runtime import atomic_json, digest, now


def route(labels, decisions):
    if labels.sample_id.isna().any() or labels.sample_id.duplicated().any():
        raise ValueError("duplicate/missing sample identity")
    indexed = {}
    for decision in decisions:
        key = decision["normalized_event_id"]
        if key in indexed:
            raise ValueError("duplicate event decision")
        indexed[key] = decision
    samples, links = [], []
    for row in labels.itertuples(index=False):
        payload = json.loads(row.payload_json)
        ids = payload.get("unresolved_event_ids", [])
        if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid unresolved event list")
        if bool(ids) != (row.label_status == "unknown_event_terms"):
            raise ValueError("event status/list mismatch")
        reasons = set()
        for key in ids:
            if key not in indexed:
                raise ValueError("event decision missing")
            decision = indexed[key]
            rs = decision.get("reasons")
            if decision.get("accepted_for_accounting") is not False or not isinstance(rs, list) or not rs or any(not isinstance(r, str) for r in rs):
                raise ValueError("unresolved event has no valid refusal")
            reasons.update(rs)
            links.append(dict(sample_id=row.sample_id, entity_id=row.entity_id,
                              signal_date=row.signal_date, event_id=key,
                              event_code=decision["ts_code"], reasons=sorted(set(rs))))
        samples.append(dict(sample_id=row.sample_id, entity_id=row.entity_id,
                            signal_date=row.signal_date, label_status=row.label_status,
                            event_ids=ids, reasons=sorted(reasons)))
    events = sorted({x["event_id"] for x in links})
    return samples, links, dict(
        candidate_rows=len(samples), unique_entities=int(labels.entity_id.nunique()),
        signal_dates=sorted(labels.signal_date.unique().tolist()),
        affected_candidates=sum(bool(s["event_ids"]) for s in samples),
        unique_unresolved_events=len(events), sample_event_links=len(links),
        status_counts=labels.label_status.value_counts().to_dict(),
        affected_candidate_reasons=dict(Counter(r for s in samples for r in s["reasons"])),
        unique_event_reasons=dict(Counter(r for e in events for r in set(indexed[e]["reasons"]))),
        reason_counts_are_nonexclusive=True, formal_training_authorized=False,
        scope="existing refusal links and integrity, not independent event coverage/PIT/path proof")


def build(root, partitions, ledger, ledger_sha):
    """Consume externally pinned partitions and their shared accounting ledger."""
    ledger = Path(ledger).resolve()
    pins = {ledger / "summary.json": ledger_sha}

    def check():
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("audit input pin mismatch")

    check()
    summary = json.loads((ledger / "summary.json").read_text(encoding="utf-8"))
    for name in ("events", "decisions"):
        pins[ledger / (name + ".json")] = summary[name + "_sha256"]
    for name in ("source", "review", "unit_review"):
        if summary.get(name + "_path"):
            pins[Path(summary[name + "_path"]).resolve()] = summary[name + "_sha256"]
    check()
    decisions = json.loads((ledger / "decisions.json").read_text(encoding="utf-8"))
    events = json.loads((ledger / "events.json").read_text(encoding="utf-8"))
    accepted = [d["normalized_event_id"] for d in decisions if d["accepted_for_accounting"]]
    event_ids = [e["event_id"] for e in events]
    if (len(decisions) != summary["source_rows"] or len(accepted) != summary["accepted_gross_accounting_events"]
            or len(decisions)-len(accepted) != summary["unresolved_events"]
            or len(event_ids) != len(set(event_ids)) or set(accepted) != set(event_ids)):
        raise ValueError("ledger denominator/event mismatch")
    frames, checks, dates, paths = [], [], set(), set()
    if not partitions:
        raise ValueError("no partitions")
    for path, sha in partitions:
        path = Path(path).resolve()
        if path in paths:
            raise ValueError("duplicate partition")
        paths.add(path)
        frame, validation = verify(path, sha)
        metadata = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        inputs = json.loads((path / "inputs.json").read_text(encoding="utf-8"))
        if metadata["signal_date"] in dates:
            raise ValueError("duplicate signal date")
        dates.add(metadata["signal_date"])
        for name in ("source", "review", "unit_review"):
            if summary.get(name + "_path") and inputs.get(str(Path(summary[name + "_path"]).resolve())) != summary[name + "_sha256"]:
                raise ValueError("partition/ledger source mismatch")
        pins[path / "summary.json"] = sha
        for name, h in metadata["artifact_hashes"].items():
            pins[path / name] = h
        frames.append(frame)
        checks.append(dict(path=str(path), **validation))
    samples, links, report = route(pd.concat(frames, ignore_index=True), decisions)
    check()
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("label-event-audit-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out / "samples.json", samples)
    atomic_json(out / "links.json", links)
    atomic_json(out / "inputs.json", {str(p): h for p, h in pins.items()})
    report.update(directory=str(out), at=now(), partition_checks=checks,
                  code_sha256=digest(Path(__file__)),
                  artifacts={n: digest(out/n) for n in ("samples.json", "links.json", "inputs.json")})
    atomic_json(out / "summary.json", report)
    return report
