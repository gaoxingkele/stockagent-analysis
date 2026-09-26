"""Conservative announcement normalization with full source-row lineage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import pandas as pd

from .dividend_source import FIELDS
from .dividend_audit import audit_events
from .runtime import atomic_json, digest, now


def normalize(frame):
    columns = FIELDS.split(",")
    if not set(columns).issubset(frame.columns):
        raise ValueError("complete raw distribution schema required")
    raw = frame[columns].reset_index(drop=True).copy()
    # ann_date identifies proposal/decision announcements. Collapse only when
    # ALL other requested source fields agree, including report period and the
    # implementation announcement. Never coalesce null and zero or sum rates.
    term_columns = [column for column in columns if column != "ann_date"]
    term_key = raw[term_columns].apply(lambda r: r.to_json(force_ascii=True), axis=1)
    identity_columns = ["ts_code", "end_date", "record_date", "ex_date", "imp_ann_date"]
    identity_key = raw[identity_columns].apply(lambda r: r.to_json(force_ascii=True), axis=1)
    variants = pd.DataFrame({"identity": identity_key, "terms": term_key}).groupby("identity").terms.nunique()
    normalized, lineage = [], []
    for key, indices in term_key.groupby(term_key, sort=True).groups.items():
        source_ids = list(map(int, indices))
        group = raw.loc[source_ids]
        row = group.iloc[0].to_dict()
        announcements = sorted(set(group.ann_date.dropna().astype(str)))
        implementation = str(row["imp_ann_date"]) if pd.notna(row["imp_ann_date"]) else None
        # Conservative observation bound; not historical vendor availability.
        row["ann_date"] = max(announcements) if announcements else None
        row["terms_known_not_before_date"] = max(announcements + ([implementation] if implementation else []), default=None)
        event_id = hashlib.sha256(key.encode("utf-8")).hexdigest()
        row["normalized_event_id"] = event_id
        row["source_row_count"] = len(group)
        row["conflicting_variants_same_identity"] = int(variants[identity_key.loc[source_ids[0]]]) > 1
        row["normalization_rule"] = "identical_terms_except_ann_date" if len(group) > 1 else "unchanged"
        normalized.append(row)
        for source_row in source_ids:
            lineage.append({"source_row": source_row, "normalized_event_id": event_id,
                            "source_row_sha256": hashlib.sha256(raw.loc[source_row].to_json(force_ascii=True).encode()).hexdigest()})
    result = pd.DataFrame(normalized)
    checked, summary = audit_events(result)
    conflicting = checked.conflicting_variants_same_identity
    late = (pd.to_datetime(checked.terms_known_not_before_date, format="%Y%m%d", errors="coerce") >
            pd.to_datetime(checked.record_date, format="%Y%m%d", errors="coerce"))
    for mask, reason in [(conflicting, "conflicting_economic_variants"), (late, "conservative_terms_date_after_record")]:
        checked.loc[mask, "audit_reasons"] = checked.loc[mask, "audit_reasons"].map(lambda r: (r + ";" + reason).strip(";"))
        checked.loc[mask, "event_terms_usable_for_gross_reference_diagnostic"] = False
    summary.update({"raw_rows": len(raw), "normalized_rows": len(checked),
                    "announcement_repetitions_collapsed": len(raw) - len(checked),
                    "conflicting_variant_rows_retained": int(conflicting.sum()),
                    "final_term_checks_passed_rows": int(checked.event_terms_usable_for_gross_reference_diagnostic.sum()),
                    "final_reason_counts": checked.audit_reasons.str.split(";").explode().loc[lambda x: x.ne("")].value_counts().to_dict(),
                    "source_rows_preserved_in_lineage": len(lineage), "formal_H01_gate_passed": False})
    return checked, pd.DataFrame(lineage), summary


def run(root, audit_path):
    source = (root / audit_path).resolve()
    permitted = (root / "output/experiments/s20_safe_v4/sources").resolve()
    if not source.is_relative_to(permitted):
        raise ValueError("input must be an in-scope research source")
    source_hash = digest(source)
    frame = pd.read_parquet(source)
    if digest(source) != source_hash:
        raise ValueError("source changed during read")
    normalized, lineage, summary = normalize(frame)
    destination = source.parent / ("normalize-" + uuid.uuid4().hex)
    destination.mkdir()
    normalized.to_parquet(destination / "normalized_distributions.parquet", index=False)
    lineage.to_parquet(destination / "lineage.parquet", index=False)
    summary.update({"at": now(), "directory": str(destination), "source_path": str(source),
                    "source_sha256": source_hash, "code_hash": digest(Path(__file__)),
                    "output_sha256": digest(destination / "normalized_distributions.parquet"),
                    "lineage_sha256": digest(destination / "lineage.parquet")})
    atomic_json(destination / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-path", required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[2], args.audit_path), ensure_ascii=False, indent=2))
