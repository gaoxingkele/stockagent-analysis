"""Versioned five-class ABCDS estimator.

Deliberately separate from :mod:`joint_model`, whose ``CLASSES`` stay frozen at
the four-class P-track. This module never mutates that constant, never claims
formal acceptance, and reports raw uncalibrated probabilities.

Class order is fixed and must be validated, because a silent reordering would
turn ``p_A`` into a different class without any error.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .baseline_model import validate_seed
from .feature_pipeline import prepare

CLASS_ORDER = ("A", "B", "C", "D", "S")
TARGET_ID = "S20.hit15_risk10_silence8.v1"
MODEL_VERSION = "S20.abcds.multinomial.v1"

PROBABILITY_COLUMNS = tuple("p_" + name for name in CLASS_ORDER)
DERIVED_COLUMNS = ("p_hit15", "p_risk10", "p_silent")


def _derive(output: pd.DataFrame) -> pd.DataFrame:
    output["p_hit15"] = output["p_A"] + output["p_B"]
    output["p_risk10"] = output["p_B"] + output["p_D"]
    output["p_silent"] = output["p_S"]
    return output


def validate_class_order(observed) -> tuple:
    observed = tuple(observed)
    if observed != CLASS_ORDER:
        raise ValueError(f"unexpected ABCDS class order {observed}; expected {CLASS_ORDER}")
    return observed


def run(samples, features, fit_labels, boundaries, feature_contract, *,
        target_id=TARGET_ID, random_seed: int = 20, regularisation: float = 1.0):
    """Fit once on the eligible fit segment; withhold fit-segment predictions."""
    validate_seed(random_seed)
    if target_id != TARGET_ID:
        raise ValueError("explicit ABCDS target required")
    if isinstance(regularisation, bool) or not isinstance(regularisation, (int, float)) \
            or not np.isfinite(regularisation) or regularisation <= 0:
        raise ValueError("positive finite regularisation required")
    matrix, assignment, preprocessing = prepare(samples, features, boundaries, feature_contract)
    if fit_labels.columns.duplicated().any() or set(fit_labels.columns) != {"sample_id", "target"}:
        raise ValueError("exact ABCDS fit label schema required")
    if fit_labels.sample_id.isna().any() or fit_labels.sample_id.duplicated().any():
        raise ValueError("unique ABCDS fit labels required")
    ids = preprocessing["fit_sample_ids"]
    if set(fit_labels.sample_id) != set(ids):
        raise ValueError("only the exact eligible fit labels may be consumed")
    y = fit_labels.set_index("sample_id").loc[ids, "target"]
    if y.isna().any() or not y.isin(CLASS_ORDER).all():
        raise ValueError("resolved ABCDS fit labels required; no silent fallback")
    missing = sorted(set(CLASS_ORDER) - set(y))
    if missing:
        raise ValueError(f"fit segment does not contain every ABCDS class: {missing}")

    parameters = dict(C=float(regularisation), solver="lbfgs", max_iter=1000,
                      random_state=random_seed)
    model = LogisticRegression(**parameters)
    columns = preprocessing["feature_columns"]
    indexed = matrix.set_index("sample_id")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(indexed.loc[ids, columns], y)
    validate_class_order(model.classes_)

    output = assignment[["sample_id", "segment", "feature_ready"]].copy().reset_index(drop=True)
    for name in (*PROBABILITY_COLUMNS, *DERIVED_COLUMNS):
        output[name] = np.nan
    output["prediction_status"] = "not_forward_segment"
    forward = output.segment.isin(["tune", "calibration", "selection-policy", "outer-test"])
    output.loc[forward, "prediction_status"] = "feature_unavailable"
    ready = forward & output.feature_ready
    if ready.any():
        probabilities = model.predict_proba(indexed.loc[output.loc[ready, "sample_id"], columns])
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() \
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10):
            raise ValueError("invalid ABCDS probability simplex")
        output.loc[ready, list(PROBABILITY_COLUMNS)] = probabilities
        output.loc[ready, "prediction_status"] = "uncalibrated_abcds_reference"
    output = _derive(output)
    card = {
        "model_version": MODEL_VERSION,
        "target_id": target_id,
        "model": "fixed_multinomial_logistic",
        "classes": list(CLASS_ORDER),
        "class_meaning": {
            "A": "hit15 and no paired risk",
            "B": "hit15 with paired risk",
            "C": "no hit15, no risk, max_gain >= 8%",
            "D": "no hit15 with paired risk",
            "S": "max_gain < 8% and no paired risk",
        },
        "parameters": parameters,
        "sklearn_version": sklearn.__version__,
        "preprocessing": preprocessing,
        "randomness": {"requested_seed": random_seed, "seed_effect": "unused_by_lbfgs",
                       "independent_market_evidence": False},
        "fit_sample_ids": ids,
        "fit_class_counts": {name: int(y.eq(name).sum()) for name in CLASS_ORDER},
        "model_level_fits": 1,
        "underlying_fits": 1,
        "calibrator_fits": 0,
        "all_candidates_retained": True,
        "predicted_rows": int(ready.sum()),
        "calibration_performed": False,
        "absolute_probability_validated": False,
        "formal_H04_accepted": False,
        "formal_training_authorized": False,
        "production_eligible": False,
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
        "iterations": model.n_iter_.tolist(),
    }
    return output, card


def run_frequency(samples, features, fit_labels, boundaries, feature_contract, *,
                  target_id=TARGET_ID):
    """Constant fit-segment class frequencies; a zero-fit control."""
    if target_id != TARGET_ID:
        raise ValueError("explicit ABCDS target required")
    matrix, assignment, preprocessing = prepare(samples, features, boundaries, feature_contract)
    ids = preprocessing["fit_sample_ids"]
    if fit_labels.columns.duplicated().any() or set(fit_labels.columns) != {"sample_id", "target"}:
        raise ValueError("exact ABCDS fit label schema required")
    if set(fit_labels.sample_id) != set(ids):
        raise ValueError("only the exact eligible fit labels may be consumed")
    y = fit_labels.set_index("sample_id").loc[ids, "target"]
    if y.isna().any() or not y.isin(CLASS_ORDER).all():
        raise ValueError("resolved ABCDS fit labels required")
    frequency = {name: float(y.eq(name).mean()) for name in CLASS_ORDER}
    output = assignment[["sample_id", "segment", "feature_ready"]].copy().reset_index(drop=True)
    for name in (*PROBABILITY_COLUMNS, *DERIVED_COLUMNS):
        output[name] = np.nan
    output["prediction_status"] = "not_forward_segment"
    forward = output.segment.isin(["tune", "calibration", "selection-policy", "outer-test"])
    output.loc[forward, "prediction_status"] = "feature_unavailable"
    ready = forward & output.feature_ready
    for name in CLASS_ORDER:
        output.loc[ready, "p_" + name] = frequency[name]
    output.loc[ready, "prediction_status"] = "uncalibrated_frequency_reference"
    output = _derive(output)
    card = {"model_version": MODEL_VERSION, "target_id": target_id,
            "model": "mature_empirical_frequency", "classes": list(CLASS_ORDER),
            "fit_class_frequency": frequency, "fit_sample_ids": ids,
            "model_level_fits": 0, "underlying_fits": 0, "calibrator_fits": 0,
            "frequency_estimates": 1, "all_candidates_retained": True,
            "predicted_rows": int(ready.sum()), "calibration_performed": False,
            "absolute_probability_validated": False, "formal_H04_accepted": False,
            "formal_training_authorized": False, "production_eligible": False}
    return output, card
