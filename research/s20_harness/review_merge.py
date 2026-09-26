"""Conflict-preserving union of independently pinned event review bundles."""
from decimal import Decimal
import json
from pathlib import Path
import uuid

import pandas as pd

from .distribution_adapter import row_fingerprint
from .runtime import atomic_json, digest, now


CRITICAL = ("row_sha256", "ts_code", "beneficiary_scope", "record_date", "ex_date", "pay_date",
            "gross_cash_per_share", "gross_cash_per_quote_unit", "bonus_per_share", "quote_unit",
            "independent_entitlement_group", "tax_policy_id", "net_cash_per_share",
            "historical_feed_available_at")
NUMERIC = {"gross_cash_per_share", "gross_cash_per_quote_unit", "bonus_per_share", "net_cash_per_share"}


def merge(frame, bundles):
    if not frame.normalized_event_id.is_unique:
        raise ValueError("duplicate source events")
    indexed = frame.set_index("normalized_event_id", drop=False)
    grouped = {}
    for bundle_id, reviews in bundles:
        for event_id, review in reviews.items():
            grouped.setdefault(event_id, []).append(dict(bundle_id=bundle_id, review=review))
    combined, decisions = {}, []
    for event_id, lineage in sorted(grouped.items()):
        reasons = []
        if event_id not in indexed.index:
            reasons.append("source_event_missing")
        else:
            row = indexed.loc[event_id]
            fingerprint = row_fingerprint(row)
            for item in lineage:
                r = item["review"]
                if r.get("row_sha256") != fingerprint:
                    reasons.append("review_row_hash_mismatch")
                if not r.get("source") or r.get("beneficiary_scope") not in ("existing_shareholders_verified", "registered_cdr_holders_verified"):
                    reasons.append("unsupported_review_evidence")
                for key, field in [("ts_code", "ts_code"), ("record_date", "record_date"),
                                   ("ex_date", "ex_date"), ("pay_date", "pay_date"),
                                   ("gross_cash_per_share", "cash_div_tax"), ("bonus_per_share", "stk_div")]:
                    if key not in r:
                        continue
                    try:
                        equal = (Decimal(str(r[key])) == Decimal(str(row[field]))) if key in NUMERIC else str(r[key]) == str(row[field])
                    except (ArithmeticError, ValueError, TypeError):
                        equal = False
                    if not equal:
                        reasons.append("review_source_disagreement:" + key)
        for key in CRITICAL:
            values = [i["review"][key] for i in lineage if key in i["review"]]
            if key in NUMERIC:
                try:
                    values = [Decimal(str(v)) if v is not None else None for v in values]
                except (ArithmeticError, ValueError, TypeError):
                    reasons.append("invalid_numeric_review:" + key)
            if values and any(v != values[0] for v in values[1:]):
                reasons.append("review_conflict:" + key)
        if not reasons:
            # First bundle supplies descriptive metadata; all originals remain
            # in the lineage ledger. Only agreeing critical terms are filled.
            result = dict(lineage[0]["review"])
            for item in lineage[1:]:
                for key in CRITICAL:
                    if key in item["review"] and key not in result:
                        result[key] = item["review"][key]
            result.update(formal_training_eligible=False, merge_source_ids=[i["bundle_id"] for i in lineage])
            combined[event_id] = result
        decisions.append(dict(normalized_event_id=event_id, included=not reasons,
                              reasons=sorted(set(reasons)), lineage=lineage))
    return combined, decisions


def build(root, normalized_path, normalized_sha, sources):
    """Sources are ordered (path, externally expected SHA) pairs; no overwrite."""
    normalized_path = Path(normalized_path).resolve()
    pins = [(normalized_path, normalized_sha)] + [(Path(p).resolve(), sha) for p, sha in sources]
    if len({p for p, _ in pins}) != len(pins):
        raise ValueError("duplicate bundle/source paths")
    def verify():
        if any(digest(p) != sha for p, sha in pins):
            raise ValueError("merge input pin mismatch")
    verify()
    bundles = [(str(p), json.loads(p.read_text(encoding="utf-8"))["reviews"]) for p, _ in pins[1:]]
    combined, decisions = merge(pd.read_parquet(normalized_path), bundles)
    verify()
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("review-merge-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out / "reviews.json", dict(reviews=combined, formal_training_eligible=False))
    atomic_json(out / "decisions.json", decisions)
    atomic_json(out / "inputs.json", [dict(path=str(p), sha256=sha) for p, sha in pins])
    report = dict(directory=str(out), at=now(), input_review_rows=sum(len(r) for _, r in bundles),
                  unique_reviewed_events=len(decisions), included=len(combined), conflicts=len(decisions)-len(combined),
                  formal_training_eligible=False, code_sha256=digest(Path(__file__)),
                  artifacts={n:digest(out/n) for n in ["reviews.json", "decisions.json", "inputs.json"]})
    atomic_json(out / "summary.json", report)
    return report
