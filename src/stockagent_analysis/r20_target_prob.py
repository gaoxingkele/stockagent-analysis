"""Utilities for the experimental R20-P v2 target-probability index.

This module is deliberately independent from the production V12/Pool-A path.
It contains only target construction and probability evaluation helpers so the
experiment cannot silently change the frozen v1 selection contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


GAIN_THRESHOLDS = (15, 20, 25)
DD_FLOOR = -15.0


@dataclass(frozen=True)
class WalkForwardFold:
    """Four-way chronological split for fitting, tuning, calibration and test."""

    name: str
    fit_end: str
    tune_start: str
    tune_end: str
    calibration_start: str
    calibration_end: str
    test_start: str
    test_end: str

    def masks(self, dates: pd.Series) -> dict[str, pd.Series]:
        values = dates.astype(str)
        return {
            "fit": values <= self.fit_end,
            "tune": values.between(self.tune_start, self.tune_end),
            "calibration": values.between(
                self.calibration_start, self.calibration_end
            ),
            "test": values.between(self.test_start, self.test_end),
        }


DEFAULT_FOLDS = (
    WalkForwardFold(
        "wf1",
        "20241031",
        "20241101",
        "20250131",
        "20250201",
        "20250228",
        "20250301",
        "20250430",
    ),
    WalkForwardFold(
        "wf2",
        "20250228",
        "20250301",
        "20250531",
        "20250601",
        "20250630",
        "20250701",
        "20250831",
    ),
    WalkForwardFold(
        "wf3",
        "20250630",
        "20250701",
        "20250930",
        "20251001",
        "20251031",
        "20251101",
        "20260126",
    ),
)


def build_target_frame(labels: pd.DataFrame) -> pd.DataFrame:
    """Validate forward labels and add nested safe-target outcomes.

    A safe target means that the stock reaches the requested gain at some point
    in the next 20 sessions while its lowest excursion from next-session entry
    open is no worse than ``DD_FLOOR``.  The output ordinal tier is:

    0: misses safe +15%; 1: reaches safe +15%; 2: reaches safe +20%;
    3: reaches safe +25%.
    """
    required = {
        "ts_code",
        "trade_date",
        "entry_open",
        "max_gain_20",
        "max_dd_20",
        "r20_close",
    }
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"missing label columns: {sorted(missing)}")

    out = labels.copy()
    out["ts_code"] = out["ts_code"].astype(str)
    out["trade_date"] = out["trade_date"].astype(str)
    numeric = ["entry_open", "max_gain_20", "max_dd_20", "r20_close"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    valid = np.isfinite(out[numeric]).all(axis=1) & (out["entry_open"] > 0)
    out = out.loc[valid].copy()

    tolerance = 1e-6
    if ((out["r20_close"] - out["max_gain_20"]) > tolerance).any():
        raise ValueError("r20_close cannot exceed max_gain_20")
    if ((out["max_dd_20"] - out["max_gain_20"]) > tolerance).any():
        raise ValueError("max_dd_20 cannot exceed max_gain_20")

    safe_risk = out["max_dd_20"] >= DD_FLOOR
    for threshold in GAIN_THRESHOLDS:
        out[f"target_p{threshold}_safe"] = (
            (out["max_gain_20"] >= threshold) & safe_risk
        ).astype("int8")
    out["target_close25_safe"] = (
        (out["r20_close"] >= 25.0) & safe_risk
    ).astype("int8")
    out["target_tier"] = (
        out["target_p15_safe"]
        + out["target_p20_safe"]
        + out["target_p25_safe"]
    ).astype("int8")
    return out


def ordered_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Project P15/P20/P25 onto the required non-increasing order."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(GAIN_THRESHOLDS):
        raise ValueError("expected an n x 3 P15/P20/P25 probability matrix")
    values = np.clip(values, 0.0, 1.0)
    return np.minimum.accumulate(values, axis=1)


def tier_probabilities(class_probabilities: np.ndarray) -> np.ndarray:
    """Convert four ordinal tier probabilities to nested P15/P20/P25."""
    values = np.asarray(class_probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("expected four class probabilities for tiers 0..3")
    return ordered_probabilities(
        np.column_stack(
            [values[:, 1:].sum(axis=1), values[:, 2:].sum(axis=1), values[:, 3]]
        )
    )


def expected_calibration_error(
    actual: Iterable[int], predicted: Iterable[float], bins: int = 10
) -> float:
    """Return equal-width expected calibration error."""
    y = np.asarray(list(actual), dtype=int)
    p = np.clip(np.asarray(list(predicted), dtype=float), 0.0, 1.0)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.minimum(np.digitize(p, edges[1:-1], right=True), bins - 1)
    error = 0.0
    for idx in range(bins):
        mask = bucket == idx
        if mask.any():
            error += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(error)


def probability_metrics(actual: Iterable[int], predicted: Iterable[float]) -> dict:
    """Metrics used by the experiment acceptance report."""
    y = np.asarray(list(actual), dtype=int)
    p = np.clip(np.asarray(list(predicted), dtype=float), 1e-7, 1 - 1e-7)
    base = float(y.mean())
    base_brier = float(brier_score_loss(y, np.full(len(y), base)))
    brier = float(brier_score_loss(y, p))
    result = {
        "n": int(len(y)),
        "event_rate": base,
        "mean_probability": float(p.mean()),
        "brier": brier,
        "brier_skill_vs_constant": (
            float(1.0 - brier / base_brier) if base_brier > 0 else None
        ),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": expected_calibration_error(y, p, bins=10),
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, p))
        result["pr_auc"] = float(average_precision_score(y, p))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    order = np.argsort(p)
    top_n = max(1, len(order) // 10)
    result["top_decile_event_rate"] = float(y[order[-top_n:]].mean())
    result["top_decile_mean_probability"] = float(p[order[-top_n:]].mean())
    return result
