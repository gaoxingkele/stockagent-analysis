"""Chronological five-segment assignment with actual label-availability purge."""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from .label_availability import _instant

SEGMENTS = ("fit", "tune", "calibration", "selection-policy", "outer-test")


def _times(series, *, optional=False):
    # Validate timezone information before converting: utc=True alone would
    # silently interpret naive local timestamps as UTC.
    return pd.Series([pd.NaT if optional and pd.isna(value) else _instant(value)
                      for value in series], index=series.index, dtype="datetime64[ns, UTC]")


def assign_segments(samples, boundaries, *, evaluation_at=None):
    """Return assignments for every row, including unknown and purged rows.

    Each boundary is {name,start_at,end_at}, half-open [start,end). A segment's
    labels can be consumed only before the next segment starts. This contract
    must be used by every fit/calibration/policy consumer; it is not training
    authorization or evidence of feature/label provenance by itself.
    """
    required = {"sample_id", "prediction_at", "feature_available_at", "horizon_close_at", "label_available_at"}
    if not required.issubset(samples.columns):
        raise ValueError("missing split sample fields")
    if samples.sample_id.isna().any() or samples.sample_id.duplicated().any():
        raise ValueError("sample IDs must be unique and nonmissing")
    if tuple(b["name"] for b in boundaries) != SEGMENTS:
        raise ValueError("exact five ordered segments required")
    parsed = [(b["name"], _instant(b["start_at"]), _instant(b["end_at"])) for b in boundaries]
    for i, (_, start, end) in enumerate(parsed):
        if start >= end or (i and parsed[i-1][2] > start):
            raise ValueError("overlapping or reversed split intervals")
    predicted = _times(samples.prediction_at)
    feature = _times(samples.feature_available_at, optional=True)
    horizon = _times(samples.horizon_close_at)
    available = _times(samples.label_available_at, optional=True)
    if horizon.le(predicted).any() or (available.notna() & available.lt(horizon)).any():
        raise ValueError("label horizon or availability violates full-window contract")
    assignments = pd.DataFrame({"sample_id": samples.sample_id, "segment": "outside",
                                 "feature_ready": feature.lt(predicted), "label_ready": False,
                                 "supervised_eligible": False, "evaluation_eligible": False,
                                 "reason": "outside_registered_intervals"})
    counts = {}
    for i, (name, start, end) in enumerate(parsed):
        mask = predicted.ge(start) & predicted.lt(end)
        assignments.loc[mask, "segment"] = name
        if name == "outer-test":
            cutoff = _instant(evaluation_at) if evaluation_at is not None else None
            if cutoff is not None and cutoff < end:
                raise ValueError("evaluation cutoff precedes outer-test interval end")
        else:
            cutoff = parsed[i + 1][1]
        label_ready = available.lt(cutoff) if cutoff is not None else pd.Series(False, index=samples.index)
        ready = label_ready & assignments.feature_ready
        assignments.loc[mask, "label_ready"] = label_ready[mask]
        assignments.loc[mask, "reason"] = "ready"
        assignments.loc[mask & ~label_ready, "reason"] = "label_pending_or_boundary_purged"
        assignments.loc[mask & ~assignments.feature_ready, "reason"] = "feature_unknown_or_not_prior"
        field = "evaluation_eligible" if name == "outer-test" else "supervised_eligible"
        assignments.loc[mask, field] = ready[mask]
        counts[name] = {"assigned_rows": int(mask.sum()), "eligible_rows": int((mask & ready).sum()),
                        "label_not_ready_rows": int((mask & ~label_ready).sum()),
                        "feature_not_ready_rows": int((mask & ~assignments.feature_ready).sum())}
    protocol = {"segments": boundaries, "evaluation_at": evaluation_at,
                "purge": "actual_label_available_at_strictly_before_next_start",
                "feature_cutoff": "strictly_before_prediction", "outer_rows_never_training": True}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = {"split_protocol_sha256": protocol_hash, "rows": len(samples), "counts": counts,
              "outside_rows": int(assignments.segment.eq("outside").sum()),
              "all_input_rows_retained": len(assignments) == len(samples), "formal_training_authorized": False}
    return assignments, report


def training_ids(assignments, segment):
    """Never let an outer-test selection act as a training dataset."""
    if segment not in SEGMENTS[:-1]:
        raise ValueError("not a supervised development segment")
    return assignments.loc[assignments.segment.eq(segment) & assignments.supervised_eligible, "sample_id"].tolist()
