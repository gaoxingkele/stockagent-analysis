"""Paired moving-date-block diagnostics; never a standalone promotion gate."""
from __future__ import annotations

import numpy as np


def _counts(report, event):
    rows = report["daily"]
    dates = [row["date"] for row in rows]
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("unique chronological calendar required")
    counts = []
    for row in rows:
        item = row["events"][event]
        values = [item[key] for key in ("positive", "unknown", "denominator")]
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in values):
            raise ValueError("nonnegative integer counts required")
        if values[0] + values[1] > values[2] or row["recommendations"] != values[2]:
            raise ValueError("event counts do not conserve recommendation denominator")
        counts.append(values)
    return dates, np.asarray(counts, dtype=np.int64).reshape(-1, 3)


def paired_date_blocks(candidate, baseline, *, event, block_lengths=(20, 40, 60),
                       replicates=2000, seed=0):
    """Rate differences use pooled counts, with paired full-calendar sampling.

    Noncircular moving blocks preserve within-block order. Intervals are percentile
    diagnostics, not a proof of coverage under arbitrary market dependence. The
    two-block guard is a minimum computational guard, NOT a power requirement.
    Unknown outcomes return identification bounds only. Zero/all-event margins
    refuse bootstrap CIs, which cannot learn unobserved tail events.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 200:
        raise ValueError("at least 200 bootstrap replicates required")
    if not block_lengths or len(set(block_lengths)) != len(block_lengths):
        raise ValueError("distinct positive block lengths required")
    if any(isinstance(b, bool) or not isinstance(b, int) or b < 1 for b in block_lengths):
        raise ValueError("distinct positive block lengths required")
    dates, left = _counts(candidate, event)
    other_dates, right = _counts(baseline, event)
    if dates != other_dates:
        raise ValueError("identical complete evaluation calendars required")
    totals = [array.sum(axis=0) for array in (left, right)]
    p, q = totals
    bounds = None
    if p[2] and q[2]:
        bounds = [float(p[0] / p[2] - (q[0] + q[1]) / q[2]),
                  float((p[0] + p[1]) / p[2] - q[0] / q[2])]
    result = {"event": event, "difference_direction": "candidate_minus_baseline",
              "evaluation_days": len(dates), "seed": seed, "replicates": replicates,
              "candidate_counts": p.tolist(), "baseline_counts": q.tolist(),
              "same_daily_recommendation_counts": bool(np.array_equal(left[:, 2], right[:, 2])),
              "identification_bounds": bounds, "bounds_are_confidence_intervals": False,
              "blocks": [], "formal_promotion_authorized": False,
              "episode_dependence_checked": False}
    for length in block_lengths:
        row = {"length": length, "ci95": None, "status": "INSUFFICIENT_EVIDENCE"}
        result["blocks"].append(row)
        if not p[2] or not q[2]:
            row["reason"] = "empty_selection"
        elif p[1] or q[1]:
            row["reason"] = "unknown_outcomes_identification_only"
        elif any(t[0] in (0, t[2]) for t in totals):
            row["reason"] = "zero_or_all_events_no_tail_evidence"
        elif len(dates) < 2 * length:
            row["reason"] = "fewer_than_two_full_blocks_not_a_power_test"
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, length]))
            differences = []
            for _ in range(replicates):
                starts = rng.integers(0, len(dates) - length + 1,
                                     size=(len(dates) + length - 1) // length)
                indices = (starts[:, None] + np.arange(length)).ravel()[:len(dates)]
                a, b = left[indices].sum(axis=0), right[indices].sum(axis=0)
                if not a[2] or not b[2]:
                    break
                differences.append(float(a[0] / a[2] - b[0] / b[2]))
            if len(differences) != replicates:
                row["reason"] = "empty_bootstrap_denominator_no_silent_discard"
            elif np.ptp(differences) == 0:
                row["reason"] = "degenerate_resampling_distribution"
            else:
                row.update(status="DIAGNOSTIC_INTERVAL", ci95=np.quantile(differences, [.025, .975]).tolist(),
                           reason="requires_power_dependence_and_protocol_review")
    return result
