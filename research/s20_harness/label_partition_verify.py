"""Pinned diagnostic partition consumer; never certifies formal training data."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import pandas as pd

from .runtime import digest, load_plan


def validate_rows(candidates, labels, summary):
    keys = ["sample_id", "entity_id", "signal_date"]
    if not set(keys).issubset(candidates) or not set(keys + ["p_class", "payload_json", "label_realized",
            "formal_training_eligible", "event_coverage_proven", "recommendation_kept", "label_status"]).issubset(labels):
        raise ValueError("missing partition schema")
    if candidates.sample_id.isna().any() or candidates.sample_id.duplicated().any():
        raise ValueError("invalid candidate identities")
    if candidates[keys].to_dict("records") != labels[keys].to_dict("records"):
        raise ValueError("candidate order/identity denominator mismatch")
    if not labels.formal_training_eligible.eq(False).all() or not labels.event_coverage_proven.eq(False).all():
        raise ValueError("diagnostic partition falsely claims formal eligibility/coverage")
    if not labels.recommendation_kept.eq(True).all():
        raise ValueError("recommendation not retained")
    for row in labels.itertuples(index=False):
        payload = json.loads(row.payload_json)
        cls = None if pd.isna(row.p_class) else row.p_class
        if payload.get("p_class") != cls or payload.get("label_realized", False) != row.label_realized:
            raise ValueError("payload/column label mismatch")
        if cls is None:
            if row.label_realized:
                raise ValueError("unknown class claims realized outcome")
        else:
            if cls not in ("A", "B", "C", "D") or row.label_realized != True:
                raise ValueError("invalid realized class")
            up, risk = payload.get("up_event"), payload.get("b5")
            if type(up) is not bool or type(risk) is not bool:
                raise ValueError("unknown event assigned class")
            expected = ("B" if risk else "A") if up else ("D" if risk else "C")
            net = payload.get("terminal_net")
            if cls != expected or net is None or not math.isfinite(net) or (net > 0) != up:
                raise ValueError("inconsistent economic class")
    counts = labels.p_class.fillna("UNKNOWN").value_counts().to_dict()
    if summary["candidate_rows"] != len(candidates) or summary["output_rows"] != len(labels):
        raise ValueError("summary denominator mismatch")
    if summary["class_counts"] != counts or summary["status_counts"] != labels.label_status.value_counts().to_dict():
        raise ValueError("summary label counts mismatch")
    if summary.get("tax_bounds_requested"):
        if not {"taxed_status", "taxed_p_class", "taxed_payload_json"}.issubset(labels):
            raise ValueError("missing tax-bound columns")
        if summary["taxed_status_counts"] != labels.taxed_status.value_counts().to_dict():
            raise ValueError("tax-bound counts mismatch")
        for row in labels.itertuples(index=False):
            payload = json.loads(row.taxed_payload_json)
            cls = None if pd.isna(row.taxed_p_class) else row.taxed_p_class
            if payload.get("status") != row.taxed_status or payload.get("p_class") != cls or payload.get("formal_training_eligible") is not False:
                raise ValueError("tax-bound payload mismatch")
            if row.taxed_status == "tax_bounds_diagnostic":
                lo, hi = payload["terminal_net_lower"], payload["terminal_net_upper"]
                if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                    raise ValueError("invalid return bounds")
                certain, possible = payload["b5_certain"], payload["b5_possible"]
                if type(certain) is not bool or type(possible) is not bool or (certain and not possible):
                    raise ValueError("invalid risk bounds")
                ups = [True] if lo > 0 else [False] if hi <= 0 else [False, True]
                risks = [True] if certain else [False] if not possible else [False, True]
                classes = sorted({("B" if r else "A") if u else ("D" if r else "C") for u in ups for r in risks})
                if payload["class_outer_set"] != classes or cls != (classes[0] if len(classes) == 1 else None):
                    raise ValueError("tax class set inconsistent with bounds")
            elif cls is not None:
                raise ValueError("unresolved tax result assigned class")


def verify(directory, expected_summary_sha256):
    directory = Path(directory).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_summary_sha256 or ""):
        raise ValueError("external expected summary hash required")
    if digest(directory / "summary.json") != expected_summary_sha256:
        raise ValueError("summary pin mismatch")
    summary = load_plan(directory / "summary.json")
    state = load_plan(directory / "state.json")
    if state.get("state") != "COMPLETED_DIAGNOSTIC" or state.get("summary_sha256") != expected_summary_sha256:
        raise ValueError("incomplete or inconsistent partition state")
    required = {"candidates.parquet", "labels.parquet", "inputs.json"}
    if set(summary["artifact_hashes"]) != required:
        raise ValueError("missing or unexpected partition artifacts")
    for name, expected in summary["artifact_hashes"].items():
        if digest(directory / name) != expected:
            raise ValueError("partition artifact changed: " + name)
    inputs = load_plan(directory / "inputs.json")
    if not inputs:
        raise ValueError("empty source provenance")
    code_count = 0
    for name, expected in inputs.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid input hash")
        path = Path(name)
        if path.suffix == ".py":
            path = directory / "code_blobs" / expected
            code_count += 1
        if digest(path) != expected:
            raise ValueError("source or recoverable code changed: " + name)
    if not code_count:
        raise ValueError("missing code snapshots")
    candidates = pd.read_parquet(directory / "candidates.parquet")
    labels = pd.read_parquet(directory / "labels.parquet")
    validate_rows(candidates, labels, summary)
    for name, expected in summary["artifact_hashes"].items():
        if digest(directory / name) != expected:
            raise ValueError("artifact changed during consumption")
    return labels, {"valid_diagnostic_partition": True, "rows": len(labels),
                    "verified_code_blobs": code_count, "input_references": len(inputs),
                    "summary_sha256": expected_summary_sha256, "formal_training_authorized": False,
                    "validation_scope": "integrity and label consistency; not independent price-path recomputation or PIT proof"}
