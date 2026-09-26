"""Recommendation-denominator metrics with missing-outcome bounds, not CIs."""
from __future__ import annotations

import pandas as pd


def binary_bounds(values):
    series = pd.Series(values, dtype=object)
    known = series.dropna()
    if not known.map(lambda v: isinstance(v, bool)).all():
        raise ValueError("outcomes must be bool or unknown, not scores or numeric labels")
    count, resolved = len(series), len(known)
    positive = int(known.eq(True).sum())
    unknown = count - resolved
    return {"denominator": count, "known": resolved, "positive": positive, "unknown": unknown,
            "rate_lower": positive / count if count else None,
            "rate_upper": (positive + unknown) / count if count else None,
            "known_only_rate": positive / resolved if resolved else None,
            "bounds_are_confidence_intervals": False}


def recommendation_report(recommendations, outcomes, calendar, *, event_columns):
    required = {"recommendation_id", "sample_id", "entity_id", "signal_date", "fill_status", "episode_id"}
    if not required.issubset(recommendations):
        raise ValueError("missing recommendation ledger fields")
    if recommendations.recommendation_id.isna().any() or recommendations.recommendation_id.duplicated().any():
        raise ValueError("duplicate/missing recommendation identity")
    if recommendations[["sample_id", "entity_id", "signal_date"]].isna().any().any():
        raise ValueError("missing recommendation sample/entity/date")
    if not {"sample_id", *event_columns}.issubset(outcomes) or not event_columns:
        raise ValueError("missing declared outcome events")
    if outcomes.sample_id.isna().any() or outcomes.sample_id.duplicated().any():
        raise ValueError("duplicate/missing outcome sample identity")
    if set(event_columns) & set(recommendations.columns):
        raise ValueError("outcome columns must not be embedded in recommendation ledger")
    if len(calendar) != len(set(calendar)) or list(calendar) != sorted(calendar):
        raise ValueError("unique chronological evaluation calendar required")
    if not set(recommendations.signal_date).issubset(calendar):
        raise ValueError("recommendations outside evaluation calendar")
    allowed = {"filled", "unfilled", "unknown", "pending"}
    if not recommendations.fill_status.isin(allowed).all():
        raise ValueError("unknown fill-state vocabulary")
    merged = recommendations.merge(outcomes[["sample_id", *event_columns]], on="sample_id", how="left",
                                   validate="many_to_one", sort=False)
    if merged.recommendation_id.tolist() != recommendations.recommendation_id.tolist():
        raise ValueError("recommendation order changed")
    metrics = {name: binary_bounds(merged[name].tolist()) for name in event_columns}
    filled = merged.loc[merged.fill_status.eq("filled")]
    daily = []
    for date in calendar:
        rows = merged.loc[merged.signal_date.eq(date)]
        daily.append({"date": date, "recommendations": len(rows),
                      "events": {name: binary_bounds(rows[name].tolist()) for name in event_columns}})
    return {"recommendations": len(merged), "unique_samples": merged.sample_id.nunique(),
            "unique_entities": merged.entity_id.nunique(), "active_signal_days": merged.signal_date.nunique(),
            "evaluation_days": len(calendar),
            "active_day_coverage": merged.signal_date.nunique() / len(calendar) if calendar else None,
            "episodes": merged.episode_id.nunique(), "missing_episode_rows": int(merged.episode_id.isna().sum()),
            "fill_counts": merged.fill_status.value_counts().to_dict(),
            "all_recommendation_events": metrics,
            "filled_only_events": {name: binary_bounds(filled[name].tolist()) for name in event_columns},
            "daily": daily, "confidence_intervals_computed": False,
            "portfolio_performance_computed": False, "formal_promotion_authorized": False,
            "interpretation": "known-only rates are conditional diagnostics; unknown bounds preserve original recommendations"}
