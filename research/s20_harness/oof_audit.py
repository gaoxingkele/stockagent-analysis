"""Audit chronological anchor predictions against model dependency membership."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .label_availability import _instant
from .runtime import digest


def membership_hash(ids):
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def audit(samples, predictions, models, dependencies):
    """dependencies includes ALL labeled fit/tune/calibration/policy inputs.

    Model information cutoff is a replay clock, not the artifact's wall-clock
    creation date. This verifies recorded provenance, not completeness of an
    arbitrary caller's membership list or validity of upstream data.
    """
    required = {"sample_id", "entity_id", "prediction_at", "feature_available_at", "horizon_close_at", "label_available_at"}
    if not required.issubset(samples.columns):
        raise ValueError("missing sample provenance fields")
    if samples.sample_id.isna().any() or samples.sample_id.duplicated().any():
        raise ValueError("duplicate/missing sample identity")
    if not {"sample_id", "model_id", "score"}.issubset(predictions.columns):
        raise ValueError("missing prediction fields")
    if predictions.sample_id.duplicated().any():
        raise ValueError("multiple anchor predictions for one sample")
    indexed = samples.set_index("sample_id")
    times = {sample_id: _instant(row.prediction_at) for sample_id, row in indexed.iterrows()}
    keys = {sample_id: (str(row.entity_id), times[sample_id]) for sample_id, row in indexed.iterrows()}
    prepared = {}
    for model_id in predictions.model_id.unique():
        errors = []
        if model_id not in models or model_id not in dependencies:
            prepared[model_id] = (["model_or_dependency_manifest_missing"], set(), None)
            continue
        manifest = models[model_id]
        ids = dependencies[model_id]
        if not ids or len(ids) != len(set(ids)):
            errors.append("empty_or_duplicate_dependencies")
        if membership_hash(ids) != manifest.get("dependency_sha256"):
            errors.append("dependency_hash_mismatch")
        path = Path(manifest["model_path"])
        if not path.is_file() or digest(path) != manifest.get("model_sha256"):
            errors.append("model_artifact_hash_mismatch")
        cutoff = _instant(manifest["information_cutoff_at"])
        training_keys = set()
        for sample_id in ids:
            if sample_id not in indexed.index:
                errors.append("unknown_dependency_sample")
                continue
            row = indexed.loc[sample_id]
            training_keys.add(keys[sample_id])
            if pd.isna(row.label_available_at) or pd.isna(row.feature_available_at):
                errors.append("dependency_availability_unknown")
                continue
            mature = _instant(row.label_available_at)
            horizon = _instant(row.horizon_close_at)
            if horizon <= times[sample_id] or mature < horizon or mature >= cutoff:
                errors.append("dependency_label_not_mature_before_cutoff")
            if _instant(row.feature_available_at) >= times[sample_id]:
                errors.append("dependency_feature_not_prior")
        prepared[model_id] = (errors, training_keys, cutoff)
    results = []
    for prediction in predictions.itertuples(index=False):
        reasons, training_keys, cutoff = prepared[prediction.model_id]
        reasons = list(reasons)
        sample_id = prediction.sample_id
        if sample_id not in indexed.index:
            reasons.append("unknown_prediction_sample")
        else:
            row = indexed.loc[sample_id]
            if keys[sample_id] in training_keys:
                reasons.append("prediction_entity_time_used_as_model_dependency")
            if cutoff is not None and cutoff >= times[sample_id]:
                reasons.append("model_information_not_prior")
            if pd.isna(row.feature_available_at) or _instant(row.feature_available_at) >= times[sample_id]:
                reasons.append("prediction_feature_not_prior")
        try:
            if not np.isfinite(float(prediction.score)):
                reasons.append("score_not_finite")
        except (ValueError, TypeError):
            reasons.append("score_not_numeric")
        results.append({"sample_id": sample_id, "model_id": prediction.model_id,
                        "recorded_oof_provenance_valid": not reasons, "reasons": sorted(set(reasons))})
    return pd.DataFrame(results), {"predictions": len(results),
        "valid_recorded_provenance": sum(r["recorded_oof_provenance_valid"] for r in results),
        "all_prediction_rows_retained": True, "dependency_completeness_proven": False,
        "formal_training_authorized": False}
