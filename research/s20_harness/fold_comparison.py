"""Target-bound evaluation pairing and nonoverlapping outer-fold summaries."""
import json
from pathlib import Path

import pandas as pd

from .paired_comparison import compare
from .policy_evaluation import verify_evaluation


def load_pair(safe_ref, risk_ref, *, safe_target_id, risk10_target_id):
    """Bind one strategy's two evaluated endpoints to the same saved policy.

    This adapter's reference B10 contract is explicit; B5 cannot be relabeled.
    Underlying market-label semantic validity is still a separate data gate.
    """
    if not isinstance(safe_target_id, str) or not safe_target_id.strip() or risk10_target_id != "P.B10.v4":
        raise ValueError("explicit safe target and reference P.B10.v4 risk endpoint required")
    loaded = []
    for ref, target in ((safe_ref, safe_target_id), (risk_ref, risk10_target_id)):
        if set(ref) != {"directory", "summary_sha256"}:
            raise ValueError("exact pinned evaluation reference required")
        directory = Path(ref["directory"]).resolve()
        verify_evaluation(directory, ref["summary_sha256"])
        report = json.loads((directory/"summary.json").read_text(encoding="utf-8"))
        if report["target_id"] != target:
            raise ValueError("paired evaluation target mismatch")
        binding = json.loads((directory/"bindings.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        rows = pd.read_parquet(directory/"evaluated_outer.parquet")
        loaded.append((rows, report, binding, metrics))
    safe, risk = loaded
    for field in ("policy_directory", "policy_summary_sha256"):
        if safe[2][field] != risk[2][field]:
            raise ValueError("endpoint policy binding mismatch")
    if safe[3]["evaluation_at"] != risk[3]["evaluation_at"] or safe[1]["evidence_mode"] != risk[1]["evidence_mode"]:
        raise ValueError("endpoint cutoff/evidence mismatch")
    columns = ["sample_id", "entity_id", "signal_date", "selected"]
    try:
        pd.testing.assert_frame_equal(safe[0][columns], risk[0][columns], check_exact=True)
    except AssertionError as exc:
        raise ValueError("endpoint candidate/selection mismatch") from exc
    result = safe[0][columns].copy()
    result["safe_profit"] = safe[0].evaluation_target.to_numpy()
    result["risk10"] = risk[0].evaluation_target.to_numpy()
    if (result.safe_profit.eq(True) & result.risk10.eq(True)).any():
        raise ValueError("safe-profit and full-window risk10 labels contradict")
    policy_input = json.loads((Path(safe[2]["policy_directory"])/"input.json").read_text(encoding="utf-8"))
    return result, {"calendar": policy_input["outer_calendar"], "evaluation_at": safe[3]["evaluation_at"],
                    "evidence_mode": safe[1]["evidence_mode"], "safe_target_id": safe_target_id,
                    "risk10_target_id": risk10_target_id, "references": [safe_ref, risk_ref],
                    "market_label_semantics_proven": False}


def aggregate(folds, *, draws=2000, seed=20):
    """No resampling across fold boundaries; no duplicate dates from seeds."""
    if not isinstance(folds, list) or not 1 <= len(folds) <= 10:
        raise ValueError("one to ten bounded outer folds required")
    seen_ids, seen_dates = set(), set()
    reports = []
    previous_end = None
    for fold in folds:
        if set(fold) != {"fold_id", "calendar", "candidate", "baseline"}:
            raise ValueError("exact fold schema required")
        identity, calendar = fold["fold_id"], fold["calendar"]
        if not isinstance(identity, str) or not identity.strip() or identity in seen_ids:
            raise ValueError("unique fold IDs required")
        if set(calendar) & seen_dates:
            raise ValueError("overlapping outer dates; seeds are not new evidence")
        if not calendar or (previous_end is not None and calendar[0] <= previous_end):
            raise ValueError("folds must be chronologically ordered")
        report = compare(fold["candidate"], fold["baseline"], calendar, draws=draws, seed=seed)
        reports.append({"fold_id": identity, "comparison": report})
        seen_ids.add(identity)
        seen_dates.update(calendar)
        previous_end = calendar[-1]
    total = sum(r["comparison"]["matched_active_days"] for r in reports)
    keys = reports[0]["comparison"]["equal_date_weight_unknown_delta_bounds"]
    weighted = {key: sum(r["comparison"]["matched_active_days"] *
                         r["comparison"]["equal_date_weight_unknown_delta_bounds"][key]
                         for r in reports if r["comparison"]["matched_active_days"] > 0) / total
                if total else None for key in keys}
    return {"folds": reports, "outer_folds": len(reports), "unique_calendar_days": len(seen_dates),
            "matched_active_days": total, "equal_date_weight_unknown_delta_bounds": weighted,
            "all_folds_count_matched": all(r["comparison"]["all_dates_same_recommendation_count"] for r in reports),
            "pooled_confidence_interval": None, "cross_fold_resampling_performed": False,
            "folds_are_independent_proven": False, "formal_G3_passed": False}


def compare_bound(candidate_refs, baseline_refs, *, safe_target_id, risk10_target_id, draws=2000, seed=20):
    left, lm = load_pair(*candidate_refs, safe_target_id=safe_target_id, risk10_target_id=risk10_target_id)
    right, rm = load_pair(*baseline_refs, safe_target_id=safe_target_id, risk10_target_id=risk10_target_id)
    for field in ("calendar", "evaluation_at", "evidence_mode", "safe_target_id", "risk10_target_id"):
        if lm[field] != rm[field]:
            raise ValueError("paired strategy scope mismatch: " + field)
    result = compare(left, right, lm["calendar"], draws=draws, seed=seed)
    # Revalidate bound artifacts after computation; no model/selection re-fitting.
    for ref in (*candidate_refs, *baseline_refs):
        verify_evaluation(ref["directory"], ref["summary_sha256"])
    return {"candidate_binding": lm, "baseline_binding": rm, "comparison": result,
            "formal_G3_passed": False}
