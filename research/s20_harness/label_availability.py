"""Actual label maturity, distinct from horizon date and prediction timestamp."""
from __future__ import annotations

import pandas as pd


def _instant(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("availability timestamps must be nonmissing and timezone-aware")
    return result.tz_convert("UTC")


def label_maturity(*, horizon_close_at, input_available_at, unresolved=False,
                   trading_exit_required=False, exit_confirmed_at=None):
    """Inputs are the complete dependency receipts for this specific label.

    Empty/missing availability cannot be replaced by the historical bar date.
    A retrospective acquisition receipt establishes availability now, not then.
    No early negative feedback shortcut: every class waits for horizon close.
    This function does not certify that the supplied dependency list is complete.
    """
    horizon = _instant(horizon_close_at)
    values = list(input_available_at)
    base = {"label_available_at": None, "mature": False,
            "dependency_coverage_proven": False}
    if unresolved:
        return dict(base, reason="unresolved_outcome")
    if not values or any(value is None for value in values):
        return dict(base, reason="missing_dependency_availability")
    times = [horizon] + [_instant(value) for value in values]
    if trading_exit_required:
        if exit_confirmed_at is None:
            return dict(base, reason="exit_pending")
        times.append(_instant(exit_confirmed_at))
    return dict(base, label_available_at=max(times).isoformat(), mature=True,
                reason="resolved_maturity_time_not_training_authorization")


def available_before(maturity, prediction_at):
    """Strict cutoff: equal timestamps do not establish ingestion before scoring."""
    prediction = _instant(prediction_at)
    if not maturity.get("mature") or not maturity.get("label_available_at"):
        return False
    return _instant(maturity["label_available_at"]) < prediction
