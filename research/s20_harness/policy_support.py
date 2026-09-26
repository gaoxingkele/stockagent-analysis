"""Shared helpers for turning per-config probabilities into frozen-policy output.

Wraps the frozen :mod:`recommendation_policy` gate so the ABCDS executors do not
re-implement risk-gating semantics. Every helper keeps the candidate schema the
frozen policy demands and never supplies outcomes to the policy layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .abcds_model import TARGET_ID
from .recommendation_policy import apply as apply_policy

CANDIDATE_COLUMNS = ("sample_id", "entity_id", "signal_date", "prediction_at", "score", "risk")
RISK_TARGET_ID = TARGET_ID + ".risk"


def gate_threshold(values, quantile: float) -> float:
    """Threshold taken from the calibration segment's own score distribution."""
    known = np.asarray([value for value in np.asarray(values, dtype=float)
                        if np.isfinite(value)], dtype=float)
    if known.size == 0:
        raise ValueError("no finite calibration values for a pre-registered threshold")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    return float(np.quantile(known, quantile))


def build_candidates(identity: pd.DataFrame, scores: pd.Series, risks: pd.Series) -> pd.DataFrame:
    """Exact candidate schema, aligned by sample id rather than by row position."""
    missing = [column for column in ("sample_id", "entity_id", "signal_date", "prediction_at")
               if column not in identity.columns]
    if missing:
        raise ValueError("identity table missing: " + ", ".join(missing))
    frame = identity[["sample_id", "entity_id", "signal_date", "prediction_at"]].copy()
    if frame.sample_id.duplicated().any():
        raise ValueError("duplicate candidate sample ids")
    frame["score"] = pd.Series(scores).reindex(frame.sample_id).to_numpy(dtype=float)
    frame["risk"] = pd.Series(risks).reindex(frame.sample_id).to_numpy(dtype=float)
    return frame[list(CANDIDATE_COLUMNS)]


def build_policy(policy_id: str, *, use_gate: bool, n_cap: int, frozen_at: str,
                 risk_cut: float | None = None, min_score: float = 0.0) -> dict:
    if n_cap not in (1, 3, 5, 10, 20):
        raise ValueError("invalid TopN cap")
    if use_gate and risk_cut is None:
        raise ValueError("a gated policy requires a pre-registered risk threshold")
    return dict(policy_id=policy_id, target_id=TARGET_ID,
                risk_target_id=RISK_TARGET_ID if use_gate else None,
                mode="risk_gated" if use_gate else "score_only_control",
                frozen_at=frozen_at, n_cap=n_cap, min_score=float(min_score),
                max_risk=None if not use_gate else float(risk_cut))


def evaluate_selection(identity: pd.DataFrame, scores: pd.Series, risks: pd.Series,
                       calendar: list[str], *, policy_id: str, use_gate: bool, n_cap: int,
                       frozen_at: str, risk_cut: float | None = None,
                       min_score: float = 0.0):
    """Return (selected sample ids, frozen policy report)."""
    policy = build_policy(policy_id, use_gate=use_gate, n_cap=n_cap, frozen_at=frozen_at,
                          risk_cut=risk_cut, min_score=min_score)
    candidates = build_candidates(identity, scores, risks)
    rows, report = apply_policy(candidates, policy, calendar)
    selected = rows.loc[rows.selected, "sample_id"].tolist()
    return selected, report
