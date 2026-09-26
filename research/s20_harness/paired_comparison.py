"""Paired daily event comparison with unknown bounds and block sensitivity.

Dates, not stock rows, are resampled. Intervals are descriptive approximations;
this module alone cannot authorize G3 or claim an independent effective sample.
"""
import math

import numpy as np
import pandas as pd

from .metrics import binary_bounds


def compare(candidate, baseline, calendar, *, draws=2000, seed=20):
    required = {"sample_id", "entity_id", "signal_date", "selected", "safe_profit", "risk10"}
    for frame in (candidate, baseline):
        if frame.columns.duplicated().any() or set(frame.columns) != required:
            raise ValueError("exact paired candidate schema required")
        if frame[["sample_id", "entity_id", "signal_date"]].isna().any().any() or frame.sample_id.duplicated().any():
            raise ValueError("unique nonmissing sample identities required")
        if frame.duplicated(["entity_id", "signal_date"]).any():
            raise ValueError("duplicate stock-day")
        if not frame.selected.map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError("boolean selections required")
        for event in ("safe_profit", "risk10"):
            if not frame[event].dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
                raise ValueError("boolean or unknown events required")
    if not calendar or list(calendar) != sorted(set(calendar)):
        raise ValueError("nonempty unique chronological calendar required")
    for date in calendar:
        if not isinstance(date, str) or len(date) != 8:
            raise ValueError("YYYYMMDD calendar required")
        pd.to_datetime(date, format="%Y%m%d", errors="raise")
    if not set(candidate.signal_date).issubset(calendar) or set(candidate.sample_id) != set(baseline.sample_id):
        raise ValueError("paired universe/calendar mismatch")
    left = candidate.set_index("sample_id").sort_index()
    right = baseline.set_index("sample_id").loc[left.index]
    for column in ("entity_id", "signal_date", "safe_profit", "risk10"):
        a, b = left[column], right[column]
        equal = (a.eq(b) | (a.isna() & b.isna())).fillna(False)
        if not equal.all():
            raise ValueError("paired identity/outcome mismatch: " + column)
    if type(draws) is not int or not 200 <= draws <= 10000 or type(seed) is not int or seed < 0:
        raise ValueError("bounded draws and nonnegative seed required")
    daily = []
    for date in calendar:
        l = left.loc[left.signal_date.eq(date) & left.selected]
        r = right.loc[right.signal_date.eq(date) & right.selected]
        row = {"signal_date": date, "candidate_count": len(l), "baseline_count": len(r),
               "matched_active": len(l) == len(r) and len(l) > 0,
               "count_matched": len(l) == len(r)}
        for event in ("safe_profit", "risk10"):
            lb = binary_bounds([None if pd.isna(v) else bool(v) for v in l[event]])
            rb = binary_bounds([None if pd.isna(v) else bool(v) for v in r[event]])
            row[event] = {"candidate": lb, "baseline": rb}
            row[event + "_delta_lower"] = lb["rate_lower"] - rb["rate_upper"] if row["matched_active"] else None
            row[event + "_delta_upper"] = lb["rate_upper"] - rb["rate_lower"] if row["matched_active"] else None
        daily.append(row)
    active = [r for r in daily if r["matched_active"]]
    fields = [event + suffix for event in ("safe_profit", "risk10") for suffix in ("_delta_lower", "_delta_upper")]
    estimates = {f: float(np.mean([r[f] for r in active])) if active else None for f in fields}
    # Preserve inactive calendar positions in block sampling, not compressed dates.
    values = np.array([[np.nan if r[f] is None else r[f] for f in fields] for r in daily], dtype=float)
    rng = np.random.default_rng(seed)
    sensitivities = []
    all_matched = all(r["count_matched"] for r in daily)
    risky = [r[e][side]["positive"] for r in active for e in ("risk10",) for side in ("candidate", "baseline")]
    for length in (20, 40, 60):
        enough_blocks = len(active) >= 5 * length
        reason = ("coverage_mismatch" if not all_matched else "insufficient_date_blocks" if not enough_blocks
                  else "no_observed_risk_events" if not any(risky) else None)
        entry = {"block_length": length, "nominal_calendar_blocks": len(calendar) // length,
                 "independence_proven": False, "reason": reason, "intervals": None}
        if reason is None:
            draws_values = []
            blocks = math.ceil(len(calendar) / length)
            for _ in range(draws):
                starts = rng.integers(0, len(calendar), size=blocks)
                indices = ((starts[:, None] + np.arange(length)) % len(calendar)).ravel()[:len(calendar)]
                sampled = values[indices]
                known = sampled[~np.isnan(sampled[:, 0])]
                if len(known):
                    draws_values.append(known.mean(axis=0))
            if len(draws_values) < draws * .95:
                entry["reason"] = "too_many_empty_bootstrap_draws"
            else:
                quantiles = np.quantile(draws_values, [.025, .975], axis=0)
                entry["intervals"] = {f: {"lower": float(quantiles[0, i]), "upper": float(quantiles[1, i])}
                                      for i, f in enumerate(fields)}
                entry["nonempty_draws"] = len(draws_values)
        sensitivities.append(entry)
    return {"daily": daily, "calendar_days": len(calendar), "matched_active_days": len(active),
            "all_dates_same_recommendation_count": all_matched,
            "candidate_active_day_coverage": sum(r["candidate_count"] > 0 for r in daily) / len(daily),
            "baseline_active_day_coverage": sum(r["baseline_count"] > 0 for r in daily) / len(daily),
            "equal_date_weight_unknown_delta_bounds": estimates, "block_sensitivity": sensitivities,
            "resampling": "paired circular moving blocks of full calendar dates", "seed": seed, "draws": draws,
            "intervals_are_descriptive_not_guarantees": True, "rare_event_upper_bound_proven": False,
            "maturity_and_source_provenance_verified": False, "episode_sensitivity_completed": False,
            "same_capital_execution_compared": False, "formal_G3_passed": False}
