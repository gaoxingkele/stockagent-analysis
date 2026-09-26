"""Bounded scalar temperature calibration for a versioned K-class simplex.

Generalises the frozen four-class calibrator by taking an explicit class order
instead of importing ``joint_model.CLASSES``. It consumes only mature
calibration-segment labels, applies the fitted temperature to later segments,
and never claims that a successful fit makes the probabilities trustworthy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from .abcds_model import CLASS_ORDER, TARGET_ID
from .splits import assign_segments, training_ids


def run(samples, predictions, labels, boundaries, model_card, *,
        target_id: str = TARGET_ID, class_order: tuple = CLASS_ORDER,
        method: str = "bounded_scalar_temperature"):
    if method not in ["bounded_scalar_temperature", "identity_raw"]:
        raise ValueError("unsupported calibration method")
    if target_id != TARGET_ID or model_card.get("target_id") != target_id:
        raise ValueError("ABCDS calibration target mismatch")
    if tuple(model_card.get("classes") or ()) != tuple(class_order):
        raise ValueError("calibration class order must match the fitted model card")
    columns = ["p_" + name for name in class_order]
    assignment, split = assign_segments(samples, boundaries)
    if model_card.get("preprocessing", {}).get("split", {}).get("split_protocol_sha256") \
            != split["split_protocol_sha256"]:
        raise ValueError("ABCDS calibration split mismatch")
    required = {"sample_id", "segment", "feature_ready", "prediction_status", *columns}
    if predictions.columns.duplicated().any() or not required.issubset(predictions.columns) \
            or predictions.sample_id.duplicated().any():
        raise ValueError("exact ABCDS predictions required")
    if set(predictions.sample_id) != set(samples.sample_id):
        raise ValueError("ABCDS prediction universe mismatch")
    frame = predictions.set_index("sample_id").loc[samples.sample_id].reset_index()
    if frame.segment.tolist() != assignment.segment.tolist() \
            or frame.feature_ready.tolist() != assignment.feature_ready.tolist():
        raise ValueError("ABCDS prediction assignment mismatch")
    values = frame[columns].to_numpy(dtype=float)
    ready = (assignment.segment.isin(["tune", "calibration", "selection-policy", "outer-test"])
             & assignment.feature_ready).to_numpy()
    if not np.isnan(values[~ready]).all() or not np.isfinite(values[ready]).all() \
            or (values[ready] < 0).any() \
            or not np.allclose(values[ready].sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("ABCDS probability availability/simplex mismatch")
    later = assignment.segment.isin(["selection-policy", "outer-test"]).to_numpy() & ready
    calibrated_names = ["cal_" + column for column in columns]
    output = frame.copy()
    for name in calibrated_names:
        output[name] = np.nan
    if method == "identity_raw":
        if labels.columns.duplicated().any() or set(labels.columns) != {"sample_id", "target"} \
                or len(labels):
            raise ValueError("identity calibration requires explicit empty labels")
        output.loc[later, calibrated_names] = values[later]
        state = dict(target_id=target_id, method=method, classes=list(class_order),
                     calibrator_fits=0, model_fits=0, calibration_sample_ids=[],
                     calibrated_rows=int(later.sum()), all_candidates_retained=True,
                     absolute_probability_validated=False, formal_H05_accepted=False,
                     production_eligible=False)
        return output, state
    ids = training_ids(assignment, "calibration")
    if labels.columns.duplicated().any() or set(labels.columns) != {"sample_id", "target"} \
            or labels.sample_id.duplicated().any() or set(labels.sample_id) != set(ids):
        raise ValueError("exact eligible calibration labels required")
    y = labels.set_index("sample_id").loc[ids, "target"]
    if not y.isin(class_order).all() or y.nunique() < 2:
        raise ValueError("resolved diverse calibration classes required")
    index_of = {name: position for position, name in enumerate(class_order)}
    # Map the calibration ids onto the aligned frame rows explicitly rather than
    # relying on any positional coincidence between two frames.
    position = {sample: row for row, sample in enumerate(samples.sample_id)}
    rows = [position[sample] for sample in ids]
    logits = np.log(np.clip(values[rows, :], 1e-12, 1))
    targets = np.array([index_of[value] for value in y])

    def loss(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(scaled)), targets]))

    fitted = minimize_scalar(loss, bounds=(np.log(0.1), np.log(10.0)), method="bounded",
                             options={"xatol": 1e-8, "maxiter": 500})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError("ABCDS temperature fit failed")
    temperature = float(np.exp(fitted.x))
    scaled = np.log(np.clip(values[later], 1e-12, 1)) / temperature
    calibrated = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
    output.loc[later, calibrated_names] = calibrated
    state = dict(target_id=target_id, method=method, classes=list(class_order),
                 temperature=temperature, temperature_bounds=[0.1, 10.0], epsilon=1e-12,
                 scipy_version=scipy.__version__, calibration_sample_ids=ids,
                 calibration_class_counts={name: int(y.eq(name).sum()) for name in class_order},
                 optimizer_evaluations=int(fitted.nfev), calibrator_fits=1, model_fits=0,
                 calibrated_rows=int(later.sum()), all_candidates_retained=True,
                 absolute_probability_validated=False, formal_H05_accepted=False,
                 production_eligible=False)
    return output, state
