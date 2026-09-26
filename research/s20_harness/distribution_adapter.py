"""Reviewed normalized distributions to accounting events, with refusal ledger."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import uuid

import pandas as pd

from .corporate_actions import Distribution
from .runtime import atomic_json, digest, load_plan, now


def row_fingerprint(row):
    """Bind reviews to the full normalized row, not a loose stock/date pair."""
    return hashlib.sha256(pd.Series(row).sort_index().to_json(force_ascii=True).encode()).hexdigest()


def adapt(frame, reviews, *, rate_policy, unit_reviews=None):
    if rate_policy not in ("gross_reference_diagnostic", "reviewed_net_cash"):
        raise ValueError("explicit supported rate policy required")
    if frame.normalized_event_id.isna().any() or frame.normalized_event_id.duplicated().any():
        raise ValueError("unique normalized event IDs required")
    quoted_cash = {}
    if unit_reviews:
        from .cash_unit_adapter import normalize_units
        unit_frame, _ = normalize_units(frame, unit_reviews)
        quoted_cash = dict(zip(unit_frame.normalized_event_id, unit_frame.gross_cash_per_quote_unit))
    independently_reviewed = set()
    for _, group in frame.groupby(["ts_code", "record_date", "ex_date"], dropna=False):
        if len(group) < 2 or "audit_reasons" not in group:
            continue
        ids = sorted(group.normalized_event_id.astype(str))
        # Multiple rows may be independent dividends OR duplicated vendor
        # periods. Only a complete, source-bound group review resolves this.
        valid_group = group.audit_reasons.eq("multiple_events_same_stock_ex_date").all()
        for _, member in group.iterrows():
            evidence = reviews.get(str(member.normalized_event_id), {})
            valid_group = valid_group and (
                evidence.get("independent_entitlement_group") == ids
                and evidence.get("row_sha256") == row_fingerprint(member)
                and evidence.get("beneficiary_scope") == "existing_shareholders_verified"
                and bool(evidence.get("source"))
                and not pd.isna(member.conflicting_variants_same_identity)
                and not bool(member.conflicting_variants_same_identity))
        if valid_group:
            independently_reviewed.update(ids)
    events, decisions = [], []
    for _, row in frame.iterrows():
        event_id = str(row.normalized_event_id)
        reasons = []
        fingerprint = row_fingerprint(row)
        review = reviews.get(event_id)
        if (pd.isna(row.event_terms_usable_for_gross_reference_diagnostic) or row.event_terms_usable_for_gross_reference_diagnostic != True) and event_id not in independently_reviewed:
            reasons.append("source_terms_unresolved")
        if pd.isna(row.conflicting_variants_same_identity) or bool(row.conflicting_variants_same_identity):
            reasons.append("conflicting_or_unknown_economic_variants")
        if review is None:
            reasons.append("beneficiary_review_missing")
        else:
            if review.get("row_sha256") != fingerprint:
                reasons.append("review_row_hash_mismatch")
            if review.get("beneficiary_scope") not in ("existing_shareholders_verified", "registered_cdr_holders_verified") or not review.get("source"):
                reasons.append("beneficiary_evidence_unverified")
            if review.get("beneficiary_scope") == "registered_cdr_holders_verified":
                if (not unit_reviews or event_id not in unit_reviews or
                        unit_reviews[event_id].get("quote_unit") != "domestic_CDR" or
                        pd.isna(quoted_cash.get(event_id))):
                    reasons.append("cdr_quote_unit_review_required")
                if rate_policy != "gross_reference_diagnostic":
                    reasons.append("cdr_net_tax_policy_not_implemented")
            if rate_policy == "reviewed_net_cash" and (not review.get("tax_policy_id") or review.get("net_cash_per_share") is None):
                reasons.append("net_cash_policy_unverified")
        if not reasons:
            try:
                known = pd.to_datetime(str(row.terms_known_not_before_date), format="%Y%m%d", errors="raise")
                if pd.isna(known):
                    raise ValueError("unknown date")
                # Day-only source knowledge cannot be assumed known at 21:00
                # on the announcement date. Require the next calendar date.
                known_date = (known + pd.Timedelta(days=1)).strftime("%Y%m%d")
                cash = float(row.cash_div_tax) if rate_policy == "gross_reference_diagnostic" else float(review["net_cash_per_share"])
                quote_unit = "ordinary_share"
                if review["beneficiary_scope"] == "registered_cdr_holders_verified":
                    cash, quote_unit = float(quoted_cash[event_id]), "domestic_CDR"
                event = Distribution(event_id=event_id, record_date=str(row.record_date), ex_date=str(row.ex_date),
                                     known_date=known_date, cash_per_share=cash, bonus_per_share=float(row.stk_div),
                                     pay_date=None if pd.isna(row.pay_date) else str(row.pay_date),
                                     bonus_list_date=None if pd.isna(row.div_listdate) else str(row.div_listdate),
                                     beneficiary_scope=review["beneficiary_scope"], quote_unit=quote_unit)
                events.append(event)
            except (TypeError, ValueError):
                reasons.append("accounting_terms_or_conservative_timing_invalid")
        decisions.append({"normalized_event_id": event_id, "ts_code": str(row.ts_code),
                          "record_date": str(row.record_date), "ex_date": str(row.ex_date),
                          "row_sha256": fingerprint, "accepted_for_accounting": not reasons,
                          "reasons": reasons, "rate_policy": rate_policy,
                          "formal_training_eligible": False})
    return events, decisions


def window_events(frame, events, decisions, ts_codes, entry_date, end_date):
    """Reject unresolved relevant events instead of dropping them from a path.

    This checks supplied records only; absence of a record is not coverage proof.
    Entitlements arise from record-date holdings, including ex-dates after exit.
    """
    record = pd.to_datetime(frame.record_date.astype("string"), format="%Y%m%d", errors="coerce")
    start, end = pd.to_datetime(entry_date, format="%Y%m%d"), pd.to_datetime(end_date, format="%Y%m%d")
    if start > end:
        raise ValueError("reversed event window")
    # Unknown record date cannot be silently screened out. It could affect the
    # holding, even if a supplied ex-date falls outside the expected window.
    relevant = frame.loc[frame.ts_code.isin(ts_codes) & (record.isna() | record.between(start, end))]
    ids = set(relevant.normalized_event_id.astype(str))
    by_id = {decision["normalized_event_id"]: decision for decision in decisions}
    unresolved = sorted(event_id for event_id in ids if event_id not in by_id or not by_id[event_id]["accepted_for_accounting"])
    if unresolved:
        return {"events": [], "unresolved_event_ids": unresolved, "status": "unknown_event_terms",
                "event_coverage_proven": False}
    selected = [event for event in events if event.event_id in ids]
    if {event.event_id for event in selected} != ids:
        raise ValueError("accepted event payload missing")
    return {"events": selected, "unresolved_event_ids": [], "status": "supplied_events_resolved",
            "event_coverage_proven": False}


def refusal_summary(data, decisions):
    """Mutually exclusive routing plus nonexclusive reasons; no new approvals."""
    from collections import Counter
    if len(decisions) != len(data) or [d["normalized_event_id"] for d in decisions] != data.normalized_event_id.astype(str).tolist():
        raise ValueError("decision denominator/order mismatch")
    routes, reasons = Counter(), Counter()
    for row, decision in zip(data.itertuples(), decisions):
        why = set(decision["reasons"])
        if bool(decision["accepted_for_accounting"]) != (not why):
            raise ValueError("decision status/reasons mismatch")
        if not why:
            route = "reviewed_gross_accounting"
        elif why != {"beneficiary_review_missing"}:
            route = "terms_units_or_additional_evidence_needed"
        elif pd.isna(row.stk_div) or float(row.stk_div) != 0:
            route = "unreviewed_share_distribution"
        elif pd.isna(row.cash_div_tax) or float(row.cash_div_tax) <= 0:
            route = "unreviewed_zero_or_missing_cash"
        else:
            route = "cash_only_beneficiary_review_missing"
        routes[route] += 1
        reasons.update(why)
    return dict(routes=dict(routes), nonexclusive_reason_counts=dict(reasons),
                routing_is_not_acceptance=True, event_rows=len(data))


def run(root, normalized_path, review_path, *, unit_review_path=None):
    root, normalized_path, review_path = Path(root).resolve(), Path(normalized_path).resolve(), Path(review_path).resolve()
    input_hash, review_hash = digest(normalized_path), digest(review_path)
    normalization = load_plan(normalized_path.parent / "summary.json")
    if normalization["output_sha256"] != input_hash:
        raise ValueError("normalized source hash mismatch")
    data = pd.read_parquet(normalized_path)
    reviews = load_plan(review_path)["reviews"]
    unit_hash = digest(Path(unit_review_path)) if unit_review_path is not None else None
    units = load_plan(Path(unit_review_path))["reviews"] if unit_review_path is not None else None
    events, decisions = adapt(data, reviews, rate_policy="gross_reference_diagnostic", unit_reviews=units)
    routing = refusal_summary(data, decisions)
    if unit_review_path is not None and digest(Path(unit_review_path)) != unit_hash:
        raise ValueError("unit review changed")
    if digest(normalized_path) != input_hash or digest(review_path) != review_hash:
        raise ValueError("adaptation inputs changed")
    output = root / "output/experiments/s20_safe_v4/sources" / ("distribution-adapter-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    atomic_json(output / "events.json", [asdict(event) for event in events])
    atomic_json(output / "decisions.json", decisions)
    report = {"at": now(), "directory": str(output), "source_rows": len(data),
              "accepted_gross_accounting_events": len(events), "unresolved_events": len(data) - len(events),
              "source_path": str(normalized_path), "source_sha256": input_hash,
              "review_path": str(review_path), "review_sha256": review_hash,
              "unit_review_path": str(Path(unit_review_path).resolve()) if unit_review_path is not None else None,
              "unit_review_sha256": unit_hash, "refusal_routing": routing,
              "events_sha256": digest(output / "events.json"), "decisions_sha256": digest(output / "decisions.json"),
              "code_sha256": digest(Path(__file__)), "formal_training_eligible": False}
    atomic_json(output / "summary.json", report)
    return report
