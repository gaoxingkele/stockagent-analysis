"""Event-bound cash units without mutating raw values or reusing loose reviews."""
from __future__ import annotations

import math
from pathlib import Path
import uuid

import pandas as pd

from .distribution_adapter import row_fingerprint
from .runtime import atomic_json, digest, load_plan, now


def normalize_units(frame, reviews):
    if frame.normalized_event_id.isna().any() or frame.normalized_event_id.duplicated().any():
        raise ValueError("unique event identity required")
    if "gross_cash_per_quote_unit" in frame:
        raise ValueError("unit normalization already applied")
    output = frame.copy()
    values, states, provenance = [], [], []
    for _, row in frame.iterrows():
        event_id = str(row.normalized_event_id)
        review = reviews.get(event_id)
        value, status = None, "unit_not_reviewed"
        if review is not None:
            if review.get("row_sha256") != row_fingerprint(row):
                raise ValueError("unit review source row changed: " + event_id)
            if any(str(row[field]) != review.get(field) for field in ("ts_code", "record_date", "ex_date")):
                raise ValueError("unit review identity mismatch")
            if not review.get("source") or not review.get("source_unit") or not review.get("quote_unit"):
                raise ValueError("unit review evidence missing")
            ratio = review["quoted_units_per_source_unit"]
            value = review["gross_cash_per_quote_unit"]
            gross = review["source_gross_cash"]
            if not all(math.isfinite(x) and x > 0 for x in (ratio, value, gross)):
                raise ValueError("invalid reviewed unit quantities")
            if not math.isclose(float(row.cash_div_tax), gross, rel_tol=0, abs_tol=1e-12):
                raise ValueError("reviewed source amount mismatch")
            if not math.isclose(gross / ratio, value, rel_tol=0, abs_tol=1e-12):
                raise ValueError("reviewed conversion inconsistent")
            status = "event_specific_gross_unit_reviewed"
            provenance.append({"event_id": event_id, "original_row_sha256": review["row_sha256"],
                               "source_unit": review["source_unit"], "quote_unit": review["quote_unit"],
                               "source_gross_cash": gross, "gross_cash_per_quote_unit": value,
                               "source": review["source"], "formal_training_eligible": False})
        values.append(value)
        states.append(status)
    output["gross_cash_per_quote_unit"] = pd.array(values, dtype="Float64")
    output["cash_unit_status"] = states
    return output, provenance


def run(root, normalized_path, expected_hash):
    root, normalized_path = Path(root).resolve(), Path(normalized_path).resolve()
    review_path = root / "config/s20_v4_cash_unit_reviews.json"
    review_hash = digest(review_path)
    if digest(normalized_path) != expected_hash:
        raise ValueError("normalized source pin mismatch")
    frame = pd.read_parquet(normalized_path)
    output, lineage = normalize_units(frame, load_plan(review_path)["reviews"])
    pd.testing.assert_frame_equal(output[frame.columns], frame)
    if digest(normalized_path) != expected_hash or digest(review_path) != review_hash:
        raise ValueError("unit inputs changed during normalization")
    destination = root / "output/experiments/s20_safe_v4/sources" / ("cash-units-" + uuid.uuid4().hex)
    if not destination.resolve().is_relative_to(root):
        raise ValueError("output escapes repository")
    destination.mkdir(parents=True)
    output.to_parquet(destination / "unit_normalized.parquet", index=False)
    atomic_json(destination / "lineage.json", lineage)
    report = {"at": now(), "directory": str(destination), "rows": len(output),
              "reviewed_unit_rows": len(lineage), "original_columns_unchanged": True,
              "source_path": str(normalized_path), "source_sha256": expected_hash,
              "review_sha256": review_hash, "code_sha256": digest(Path(__file__)),
              "artifact_hashes": {name: digest(destination / name) for name in ("unit_normalized.parquet", "lineage.json")},
              "formal_training_authorized": False, "tax_policy_authorized": False,
              "consumer_requirement": "select reviewed quote-unit gross explicitly; never mix with unreviewed source units"}
    atomic_json(destination / "summary.json", report)
    return report
