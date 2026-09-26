"""Scoped suspension explanation for original gaps, not global H01 approval."""
from pathlib import Path
import uuid

import pandas as pd

from .label_availability import _instant
from .runtime import atomic_json, digest, now


def combine(gaps, semantics):
    keys = ["ts_code", "trade_date"]
    if gaps[keys].isna().any().any() or gaps.duplicated(keys).any():
        raise ValueError("unique gap keys required")
    required = {*keys, "review_state", "review_resolved_at", "receipt_sha256", "received_at",
                "historical_prediction_eligible", "executable_fill_proven"}
    if not required.issubset(semantics):
        raise ValueError("missing semantic evidence")
    if semantics[keys + ["review_state", "receipt_sha256", "received_at"]].isna().any().any():
        raise ValueError("missing semantic identity")
    for flag in ["historical_prediction_eligible", "executable_fill_proven"]:
        if not semantics[flag].map(lambda v: type(v) is bool and not v).all():
            raise ValueError("unsupported prediction/fill authorization")
    allowed = {"not_event_reviewed", "full_day_suspended", "resumption_announced_quote_observed"}
    if not semantics.review_state.isin(allowed).all():
        raise ValueError("unknown semantic state")
    groups = {}
    for key, group in semantics.groupby(keys, sort=False):
        if group.review_state.nunique() != 1:
            raise ValueError("inconsistent event review")
        groups[key] = group
    result = gaps.copy()
    supports, resolutions = [], []
    for r in gaps.itertuples():
        group = groups.get((r.ts_code, r.trade_date))
        state, resolved = "not_event_reviewed", None
        if group is None and r.daily_evidence_state not in {"no_provider_record_on_queried_date", "date_not_yet_queried"}:
            raise ValueError("missing semantic source group")
        if group is not None:
            if set(group.receipt_sha256) != set(r.daily_receipt_hashes) or set(group.received_at) != set(r.daily_received_at):
                raise ValueError("gap/semantic receipt lineage mismatch")
            state = group.review_state.iloc[0]
            if state != "not_event_reviewed":
                stamps = {_instant(v).isoformat() for v in group.review_resolved_at}
                if len(stamps) != 1:
                    raise ValueError("inconsistent review time")
                resolved = next(iter(stamps))
                if any(_instant(v) > _instant(resolved) for v in group.received_at):
                    raise ValueError("review precedes source receipt")
        exchange = r.evidence_state == "reported_suspension_consistent"
        if r.evidence_state == "conflicting_quote" or state == "resumption_announced_quote_observed" or r.daily_evidence_state == "quote_conflict_with_original_gap":
            support = "conflicting_evidence"
        elif exchange and state == "full_day_suspended":
            support = "exchange_and_event_review"
        elif exchange:
            support = "exchange_interval_supported"
        elif state == "full_day_suspended":
            support = "event_review_supported"
        elif r.daily_evidence_state == "provider_full_day_candidate":
            support = "provider_candidate_only"
        else:
            support = "unresolved"
        supports.append(support)
        resolutions.append(resolved)
    result["suspension_support"] = supports
    result["event_resolution_at"] = resolutions
    result["historical_prediction_eligible"] = False
    return result, dict(gap_stock_dates=len(gaps), support_counts=result.suspension_support.value_counts().to_dict(),
                        scope="original SH/SZ between-observation gaps only",
                        formal_training_authorized=False, historical_availability_proven=False,
                        remaining_requirements=["listing-edge and complete-universe coverage", "historical source availability",
                                                "identity and corporate-action validity", "execution feasibility", "BJ separate audit"])


def build(root, gaps_path, gaps_sha, semantics_path, semantics_sha):
    pins = [(Path(gaps_path), gaps_sha), (Path(semantics_path), semantics_sha)]
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("input pin mismatch")
    table, report = combine(*(pd.read_parquet(p) for p, _ in pins))
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("input changed during combination")
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("gap-validity-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    table.to_parquet(out / "gaps.parquet", index=False)
    atomic_json(out / "inputs.json", [{"path": str(p.resolve()), "sha256": sha} for p, sha in pins])
    report.update(directory=str(out), at=now(), code_sha256=digest(Path(__file__)),
                  table_sha256=digest(out / "gaps.parquet"), inputs_sha256=digest(out / "inputs.json"))
    atomic_json(out / "summary.json", report)
    return report
