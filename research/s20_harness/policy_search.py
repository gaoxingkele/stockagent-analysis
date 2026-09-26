"""Finite, selection-segment-only reference policy comparison."""
import hashlib
import json
import math

import pandas as pd

from .baseline_evaluation import evaluate
from .label_availability import _instant
from .recommendation_policy import apply
from .splits import assign_segments


def compare(samples, candidates, outcomes, boundaries, calendar, registry):
    """No outer outcomes accepted; all candidate rows stay in each trial.

    This fixed search uses a conservative event-rate lower bound under unknown
    outcomes, NOT a statistical confidence bound. Eligibility also requires
    prespecified coverage/count floors. It is a diagnostic selector, not G3.
    """
    if set(registry) != {"registry_id", "target_id", "registered_at", "policies",
                         "min_active_coverage", "min_selected", "min_mature_selected"}:
        raise ValueError("exact policy search registry required")
    for key in ("registry_id", "target_id"):
        if not isinstance(registry[key], str) or not registry[key].strip():
            raise ValueError("named registry/target required")
    policies = registry["policies"]
    if not isinstance(policies, list) or not 1 <= len(policies) <= 6:
        raise ValueError("one to six registered policies required")
    ids = [p["policy_id"] for p in policies]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate registered policy ID")
    coverage = registry["min_active_coverage"]
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError("positive finite coverage floor required")
    for key in ("min_selected", "min_mature_selected"):
        if type(registry[key]) is not int or registry[key] < 1:
            raise ValueError("positive integer count floors required")
    assignment, split = assign_segments(samples, boundaries)
    selection_start = _instant(boundaries[3]["start_at"])
    cutoff = _instant(boundaries[4]["start_at"])
    registered = _instant(registry["registered_at"])
    if registered >= selection_start:
        raise ValueError("registry must precede selection segment")
    for p in policies:
        if p["target_id"] != registry["target_id"]:
            raise ValueError("policy target mismatch")
        if _instant(p["frozen_at"]) > registered:
            raise ValueError("policy not frozen when registered")
    expected = assignment.loc[assignment.segment.eq("selection-policy"), "sample_id"]
    if set(candidates.sample_id) != set(expected):
        raise ValueError("exact complete selection candidate universe required")
    readiness = assignment.set_index("sample_id").feature_ready
    if any(pd.notna(row.score) and not readiness.loc[row.sample_id]
           for row in candidates.itertuples(index=False)):
        raise ValueError("unavailable features cannot yield selection scores")
    # Label-ready, not feature-ready: missing prediction cannot erase a candidate.
    ready_ids = assignment.loc[assignment.segment.eq("selection-policy") & assignment.label_ready, "sample_id"]
    if set(outcomes.sample_id) != set(ready_ids):
        raise ValueError("only exact mature selection outcomes accepted")
    meta = samples.set_index("sample_id")
    for row in outcomes.itertuples(index=False):
        if _instant(row.label_available_at) != _instant(meta.loc[row.sample_id, "label_available_at"]):
            raise ValueError("outcome maturity differs from sample manifest")
    for date in calendar:
        close = pd.to_datetime(date, format="%Y%m%d").tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        if not selection_start <= close < _instant(boundaries[3]["end_at"]):
            raise ValueError("calendar outside selection segment")
    trials, ledgers = [], {}
    for policy in policies:
        rows, selection = apply(candidates, policy, calendar)
        evaluated, metrics = evaluate(rows, samples, outcomes, evaluation_at=cutoff.isoformat(), calendar=calendar)
        bound = metrics["selected_candidates"]["event_bounds"]
        eligible = (selection["active_day_coverage"] >= coverage
                    and bound["denominator"] >= registry["min_selected"]
                    and bound["known"] >= registry["min_mature_selected"])
        trials.append({"policy_id": policy["policy_id"], "policy_sha256": selection["policy_sha256"],
                       "eligible": eligible, "coverage": selection["active_day_coverage"],
                       "selected_event_bounds": bound, "metrics": metrics})
        ledgers[policy["policy_id"]] = evaluated
    ranked = sorted((t for t in trials if t["eligible"]),
                    key=lambda t: (-t["selected_event_bounds"]["rate_lower"], -t["coverage"], t["policy_id"]))
    registry_hash = hashlib.sha256(json.dumps(registry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return ledgers, {"registry": registry, "registry_sha256": registry_hash, "split": split,
                     "trials": trials, "policy_evaluations": len(trials),
                     "selected_policy_id": ranked[0]["policy_id"] if ranked else None,
                     "status": "REFERENCE_POLICY_SELECTED" if ranked else "INSUFFICIENT_EVIDENCE",
                     "selection_rule": "max_unknown_lower_rate_then_coverage_then_policy_id",
                     "label_cutoff": cutoff.isoformat(), "outer_outcomes_consumed": False,
                     "criterion_is_confidence_bound": False, "historical_registry_receipt_proven": False,
                     "matched_coverage_gain_proven": False, "formal_H05_accepted": False}
