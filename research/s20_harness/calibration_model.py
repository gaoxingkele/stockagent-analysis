"""Isolated calibration consumer with a separate mature-label boundary."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .splits import assign_segments, training_ids


def run(samples, predictions, calibration_labels, boundaries, baseline_card, *, target_id, anchor_context=None):
    """Fit fixed sigmoid-on-logit calibration, then predict only later segments.

    Supplied reference predictions are not independently authenticated here.
    Passing this consumer does not prove selected-subset calibration quality.
    """
    prior_anchor=baseline_card.get('preprocessing',{}).get('anchor_admission')
    if (prior_anchor is None)!=(anchor_context is None):
        raise ValueError('baseline and calibration anchor evidence must agree')
    if anchor_context is not None:
        from .anchor_features import effective_samples
        samples,anchor_report=effective_samples(samples,anchor_context)
        if anchor_report!=prior_anchor:
            raise ValueError('anchor admission changed before calibration')
    assignment, split = assign_segments(samples, boundaries)
    if (not isinstance(target_id, str) or not target_id.strip()
            or baseline_card.get("target_id") != target_id):
        raise ValueError("baseline/calibration target mismatch")
    if baseline_card.get("preprocessing", {}).get("split", {}).get("split_protocol_sha256") != split["split_protocol_sha256"]:
        raise ValueError("baseline/calibration split mismatch")
    required = {"sample_id", "segment", "feature_ready", "raw_probability", "prediction_status"}
    if predictions.columns.duplicated().any() or set(predictions.columns) != required:
        raise ValueError("exact baseline prediction schema required")
    if predictions.sample_id.isna().any() or predictions.sample_id.duplicated().any():
        raise ValueError("unique prediction IDs required")
    if set(predictions.sample_id) != set(samples.sample_id):
        raise ValueError("prediction universe mismatch")
    indexed = predictions.set_index("sample_id").loc[samples.sample_id]
    assignment = assignment.reset_index(drop=True)
    if indexed.segment.tolist() != assignment.segment.tolist():
        raise ValueError("prediction segment mismatch")
    if (not indexed.feature_ready.map(lambda x: isinstance(x, (bool, np.bool_))).all()
            or indexed.feature_ready.tolist() != assignment.feature_ready.tolist()):
        raise ValueError("prediction availability mismatch")
    probability = indexed.raw_probability
    if not pd.api.types.is_numeric_dtype(probability) or pd.api.types.is_bool_dtype(probability):
        raise ValueError("numeric probabilities required")
    valid = probability.notna()
    if not probability[valid].between(0, 1).all():
        raise ValueError("probability outside unit interval")
    expected = assignment.segment.isin(["tune", "calibration", "selection-policy", "outer-test"]) & assignment.feature_ready
    if valid.tolist() != expected.tolist():
        raise ValueError("unexpected missing or in-sample probability")
    expected_status = np.where(expected, "uncalibrated_reference_prediction",
                               np.where(assignment.segment.isin(["fit", "outside"]),
                                        "not_forward_segment", "feature_unavailable"))
    if indexed.prediction_status.tolist() != expected_status.tolist():
        raise ValueError("prediction status mismatch")
    ids = training_ids(assignment, "calibration")
    if not ids:
        raise ValueError("no eligible calibration samples")
    if (calibration_labels.columns.duplicated().any()
            or set(calibration_labels.columns) != {"sample_id", "target"}
            or calibration_labels.sample_id.isna().any()
            or calibration_labels.sample_id.duplicated().any()):
        raise ValueError("exact unique calibration label schema required")
    if set(calibration_labels.sample_id) != set(ids):
        raise ValueError("only exact eligible calibration labels may be consumed")
    y = calibration_labels.set_index("sample_id").loc[ids, "target"]
    if not y.map(lambda v: isinstance(v, (bool, np.bool_))).all() or y.nunique() != 2:
        raise ValueError("two resolved boolean calibration classes required")
    epsilon = 1e-6

    def logits(series):
        p = series.to_numpy(dtype=float).clip(epsilon, 1 - epsilon)
        return (np.log(p) - np.log1p(-p)).reshape(-1, 1)

    parameters = dict(C=1.0, solver="lbfgs", max_iter=1000, random_state=20)
    model = LogisticRegression(**parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(logits(probability.loc[ids]), y.astype(int))
    output = indexed.reset_index().copy()
    output["calibrated_probability"] = np.nan
    output["calibration_status"] = "not_after_calibration"
    later = assignment.segment.isin(["selection-policy", "outer-test"])
    output.loc[later, "calibration_status"] = "feature_unavailable"
    ready = later & assignment.feature_ready
    if ready.any():
        later_ids = output.loc[ready, "sample_id"]
        calibrated = model.predict_proba(logits(probability.loc[later_ids]))[:, 1]
        if not np.isfinite(calibrated).all():
            raise ValueError("nonfinite calibrated probability")
        output.loc[ready, "calibrated_probability"] = calibrated
        output.loc[ready, "calibration_status"] = "reference_calibrated_quality_unverified"
    return output, {
        "target_id": target_id, "split_protocol_sha256": split["split_protocol_sha256"],
        "method": "fixed_sigmoid_on_logit", "parameters": parameters, "epsilon": epsilon,
        "sklearn_version": sklearn.__version__, "calibrator_fits": 1,
        "calibration_sample_ids": ids, "positive_count": int(y.sum()),
        "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
        "iterations": model.n_iter_.tolist(), "rows": len(output),
        "calibrated_rows": int(ready.sum()), "all_candidates_retained": True,
        "selected_subset_reliability_verified": False, "input_predictions_authenticated": False,
        "historical_calibrator_availability_proven": False, "formal_H05_accepted": False,
        "production_eligible": False,
    }
