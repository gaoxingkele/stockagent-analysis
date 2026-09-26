"""Receipt-verified name-history coverage; acquisition is not PIT proof."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import uuid

import pandas as pd

from .metadata_source import NAME_FIELDS
from .runtime import atomic_json, digest, load_plan, now


def audit(source: Path) -> dict:
    source = Path(source).resolve()
    plan_path = source / "collection_plan.json"
    plan_hash = digest(plan_path)
    plan = load_plan(plan_path)
    codes = plan["codes"]
    if len(codes) != len(set(codes)) or any(
        not isinstance(c, str) or not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", c) for c in codes
    ):
        raise ValueError("invalid or duplicate planned codes")
    # Snapshot only committed receipts. A live collector may append new ones;
    # this report never claims a transactional snapshot of its heartbeat.
    committed = [c for c in codes if (source / f"{c}.json").is_file()]
    diagnostics, inputs = [], []
    for code in committed:
        receipt_path = source / f"{code}.json"
        receipt_hash = digest(receipt_path)
        receipt = load_plan(receipt_path)
        path = source / f"{code}.parquet"
        if receipt.get("code") != code or not path.is_file() or digest(path) != receipt["sha256"]:
            raise ValueError("history receipt identity or hash mismatch")
        frame = pd.read_parquet(path)
        fields = NAME_FIELDS.split(",")
        if not set(fields).issubset(frame.columns) or not frame.ts_code.eq(code).all():
            raise ValueError("history schema or code mismatch")
        if len(frame) != receipt["rows"] or len(frame) >= 1000:
            raise ValueError("history receipt row count mismatch or truncation")
        if digest(path) != receipt["sha256"] or digest(receipt_path) != receipt_hash:
            raise ValueError("history input changed while reading")
        parsed = {key: pd.to_datetime(frame[key].astype("string"), format="%Y%m%d", errors="coerce")
                  for key in ("start_date", "end_date", "ann_date")}
        bad_required = parsed["start_date"].isna() | parsed["ann_date"].isna()
        supplied_end = frame.end_date.notna() & frame.end_date.astype("string").str.strip().ne("")
        bad_end = supplied_end & parsed["end_date"].isna()
        reversed_period = parsed["end_date"].lt(parsed["start_date"])
        blank_name = frame.name.isna() | frame.name.astype("string").str.strip().eq("")
        unique = frame[fields].drop_duplicates()
        conflicts = unique.groupby(["start_date", "ann_date"], dropna=False).name.nunique(dropna=False).gt(1)
        diagnostics.append({"ts_code": code, "rows": len(frame), "empty_unknown": frame.empty,
                            "exact_duplicate_rows": len(frame) - len(unique),
                            "invalid_required_date_rows": int(bad_required.sum()),
                            "invalid_end_date_rows": int(bad_end.sum()),
                            "reversed_period_rows": int(reversed_period.sum()),
                            "missing_name_rows": int(blank_name.sum()),
                            "conflicting_effective_announcement_groups": int(conflicts.sum())})
        inputs.append({"code": code, "data_sha256": receipt["sha256"], "receipt_sha256": receipt_hash})
    if digest(plan_path) != plan_hash:
        raise ValueError("collection plan changed")
    output = source / ("audit-" + uuid.uuid4().hex)
    output.mkdir()
    atomic_json(output / "inputs.json", {"plan_sha256": plan_hash, "receipts": inputs,
                                        "audit_code_sha256": digest(Path(__file__))})
    atomic_json(output / "code_diagnostics.json", diagnostics)
    total_fields = ("rows", "exact_duplicate_rows", "invalid_required_date_rows", "invalid_end_date_rows",
                    "reversed_period_rows", "missing_name_rows", "conflicting_effective_announcement_groups")
    report = {"at": now(), "directory": str(output), "source": str(source),
              "planned_codes": len(codes), "verified_codes": len(committed),
              "uncollected_codes": sorted(set(codes) - set(committed)),
              "empty_unknown_codes": [d["ts_code"] for d in diagnostics if d["empty_unknown"]],
              "totals": {k: sum(d[k] for d in diagnostics) for k in total_fields},
              "complete_receipt_coverage": len(committed) == len(codes),
              "official_ST_status_proven": False, "historical_revision_availability_proven": False,
              "formal_H01_gate_passed": False,
              "semantics": "Missing or empty is unknown. Complete receipts do not prove complete vendor history."}
    atomic_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source), ensure_ascii=False, indent=2))
