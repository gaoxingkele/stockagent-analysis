"""Verify committed daily suspension receipts; preserve partial-acquisition scope."""
from pathlib import Path
import re
import uuid

import pandas as pd

from .label_availability import _instant
from .runtime import atomic_json, digest, load_plan, now
from .suspension_daily_source import inspect


def audit(root, directory, *, expected_plan_sha256):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    plan_path = directory / "collection_plan.json"
    if digest(plan_path) != expected_plan_sha256:
        raise ValueError("collection plan pin mismatch")
    plan = load_plan(plan_path)
    dates = [item["date"] for item in plan["dates"]]
    if dates != sorted(set(dates)) or not dates or any(not re.fullmatch(r"\d{8}", d) for d in dates):
        raise ValueError("invalid collection dates")
    receipts, missing, tables = [], [], []
    # Capture the committed set once; later collector commits belong to a new
    # audit snapshot. No lock removal or writes into the live source directory.
    committed = {p.stem for p in directory.glob("????????.json")}
    if not committed.issubset(dates):
        raise ValueError("unexpected date receipts")
    for item in plan["dates"]:
        date = item["date"]
        if date not in committed:
            missing.append(date)
            continue
        receipt_path, data_path = directory / (date + ".json"), directory / (date + ".parquet")
        receipt_sha = digest(receipt_path)
        receipt = load_plan(receipt_path)
        if receipt["date"] != date or digest(data_path) != receipt["sha256"]:
            raise ValueError("receipt/data mismatch")
        if _instant(receipt["received_at"]) < _instant(receipt["requested_at"]):
            raise ValueError("reversed acquisition times")
        frame = pd.read_parquet(data_path)
        check = inspect(frame, date)
        if check != receipt["check"] or check["invalid_rows"] or check["duplicate_rows"] or check["possible_truncation"]:
            raise ValueError("schema/count/truncation check failed")
        reference = root / "output/tushare_cache/daily" / (date + ".parquet")
        if digest(reference) != item["sha256"]:
            raise ValueError("reference daily pin mismatch")
        quotes = set(pd.read_parquet(reference, columns=["ts_code"]).ts_code)
        frame["quote_present"] = frame.ts_code.isin(quotes)
        frame["receipt_sha256"] = receipt_sha
        frame["received_at"] = receipt["received_at"]
        # Null timing is only a provider full-day candidate, not approved truth.
        frame["full_day_candidate"] = frame.suspend_type.eq("S") & frame.suspend_timing.isna()
        if digest(data_path) != receipt["sha256"] or digest(receipt_path) != receipt_sha or digest(reference) != item["sha256"]:
            raise ValueError("input changed during audit")
        receipts.append({"date": date, "receipt_sha256": receipt_sha, "data_sha256": receipt["sha256"], "reference_sha256": item["sha256"]})
        tables.append(frame)
    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    output = root / "output/experiments/s20_safe_v4/sources" / ("suspension-daily-audit-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    if tables:
        combined.to_parquet(output / "rows.parquet", index=False)
    atomic_json(output / "inputs.json", receipts)
    report = {"directory": str(output), "source_directory": str(directory), "at": now(),
              "plan_sha256": expected_plan_sha256, "committed_dates": len(receipts), "total_dates": len(dates),
              "uncommitted_dates": missing, "complete_receipt_coverage": not missing,
              "rows": len(combined), "type_counts": combined.suspend_type.value_counts().to_dict() if tables else {},
              "full_day_candidates": int(combined.full_day_candidate.sum()) if tables else 0,
              "full_day_candidates_with_quotes": int((combined.full_day_candidate & combined.quote_present).sum()) if tables else 0,
              "rows_sha256": digest(output / "rows.parquet") if tables else None,
              "inputs_sha256": digest(output / "inputs.json"), "code_sha256": digest(Path(__file__)),
              "schema_validator_sha256": digest(Path(__file__).with_name("suspension_daily_source.py")),
              "formal_H01_gate_passed": False, "historical_availability_proven": False}
    atomic_json(output / "summary.json", report)
    return report
