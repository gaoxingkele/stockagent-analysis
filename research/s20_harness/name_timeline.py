"""Compact announcement-gated name timelines; these are not official ST flags."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import pandas as pd

from .name_history import name_asof
from .runtime import atomic_json, digest, load_plan, now


def timeline(events, codes, start_date, end_date):
    start = pd.to_datetime(str(start_date), format="%Y%m%d")
    end = pd.to_datetime(str(end_date), format="%Y%m%d")
    if start > end:
        raise ValueError("reversed timeline interval")
    selected = events.loc[events.ts_code.isin(codes)].copy()
    effective = pd.to_datetime(selected.start_date.astype("string"), format="%Y%m%d", errors="coerce")
    announced = pd.to_datetime(selected.ann_date.astype("string"), format="%Y%m%d", errors="coerce")
    # No activation for unknown dates, including partially known records.
    valid = effective.notna() & announced.notna()
    activation = pd.concat([effective[valid], announced[valid] + pd.Timedelta(days=1)], axis=1).max(axis=1)
    boundaries = sorted({start, *activation.loc[activation.gt(start) & activation.le(end)].tolist()})
    rows = []
    for at in boundaries:
        state = name_asof(selected, codes, at.strftime("%Y%m%d"))
        state["valid_from"] = at.strftime("%Y%m%d")
        # Keep all activation boundaries even if the winning name stays the
        # same; late older records must not displace a newer effective name.
        rows.append(state)
    for i, row in enumerate(rows):
        row["valid_until_exclusive"] = rows[i + 1]["valid_from"] if i + 1 < len(rows) else (end + pd.Timedelta(days=1)).strftime("%Y%m%d")
    return rows


def build(source: Path, start_date="20240102", end_date="20260911"):
    from .name_history_audit import audit
    source = Path(source).resolve()
    checked = audit(source)
    if not checked["complete_receipt_coverage"]:
        raise ValueError("name acquisition incomplete; audit retained")
    audit_dir = Path(checked["directory"])
    audit_inputs = load_plan(audit_dir / "inputs.json")
    rows = []
    for item in audit_inputs["receipts"]:
        path = source / (item["code"] + ".parquet")
        if digest(path) != item["data_sha256"]:
            raise ValueError("audited history changed")
        events = pd.read_parquet(path)
        states = timeline(events, [item["code"]], start_date, end_date)
        if digest(path) != item["data_sha256"]:
            raise ValueError("history changed during timeline construction")
        for state in states:
            state["source_code"] = item["code"]
            state["source_sha256"] = item["data_sha256"]
        rows.extend(states)
    output = source / ("timeline-" + uuid.uuid4().hex)
    output.mkdir()
    pd.DataFrame(rows).to_parquet(output / "name_timeline.parquet", index=False)
    report = {"at": now(), "directory": str(output), "start_date": start_date, "end_date": end_date,
              "codes": len(audit_inputs["receipts"]), "timeline_rows": len(rows),
              "unknown_intervals": sum(r["status"] == "unknown" for r in rows),
              "audit_directory": str(audit_dir), "audit_inputs_sha256": digest(audit_dir / "inputs.json"),
              "timeline_sha256": digest(output / "name_timeline.parquet"),
              "code_sha256": digest(Path(__file__)),
              "asof_code_sha256": digest(Path(__file__).with_name("name_history.py")),
              "official_ST_status_proven": False, "historical_revision_availability_proven": False,
              "formal_training_eligible": False}
    atomic_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source), indent=2))
