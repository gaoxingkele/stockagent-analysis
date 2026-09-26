"""Train-segment-only preprocessing for the chronological research pipeline.

This consumes explicit feature lineage, not arbitrary numeric label columns.
Caller-supplied timestamps and declarations are checked, not independently
certified. It never authorizes formal training or silently removes candidates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .splits import assign_segments, training_ids


def prepare(samples, values, boundaries, feature_contract, *, evaluation_at=None,
            dependency_contract=None, dependency_receipts=None, anchor_context=None):
    """Fit medians/scales on eligible fit rows; retain every sample at transform.

    values has sample_id plus the exact allowlisted features. Missing numeric
    observations may be imputed; unknown/late feature availability may not.
    Constant and entirely missing fit columns are retained as zero with flags.
    No tune, calibration, selection or outer labels are accepted by this API.
    """
    if feature_contract.get("role") != "prediction_features":
        raise ValueError("prediction feature contract required")
    features = feature_contract.get("columns")
    if (not isinstance(features, list) or not features
            or any(not isinstance(c, str) or not c.strip() for c in features)
            or len(set(features)) != len(features) or "sample_id" in features):
        raise ValueError("unique named feature columns required")
    if feature_contract.get("source_provenance_verified") is not True:
        raise ValueError("explicit feature provenance declaration required")
    anchor_report=None
    unavailable_anchor=set()
    if anchor_context is not None:
        from .anchor_features import attach
        if set(anchor_context)!={'samples','predictions','models','dependencies','feature_name'}:
            raise ValueError('exact anchor context required')
        name=anchor_context['feature_name']
        if name not in features:
            raise ValueError('anchor feature must be allowlisted')
        provenance=anchor_context['samples'].set_index('sample_id')
        columns=['entity_id','prediction_at','feature_available_at','horizon_close_at','label_available_at']
        # Anchor admission may have extra earlier dependency samples, but cannot
        # relabel this run's own sample provenance to make an anchor look valid.
        pd.testing.assert_frame_equal(samples.set_index('sample_id')[columns],
            provenance.loc[samples.sample_id,columns],check_dtype=False,check_names=False)
        attached,anchor_report=attach(values,anchor_context['samples'],anchor_context['predictions'],
            anchor_context['models'],anchor_context['dependencies'],feature_name=name)
        unavailable_anchor=set(attached.loc[~attached.anchor_provenance_valid,'sample_id'])
        values=attached.drop(columns=['anchor_provenance_valid','anchor_provenance_reasons'])
    if values.columns.duplicated().any() or set(values.columns) != {"sample_id", *features}:
        raise ValueError("exact feature allowlist required")
    if values.sample_id.isna().any() or values.sample_id.duplicated().any():
        raise ValueError("unique feature sample IDs required")
    dependency_report = None
    if (dependency_contract is None) != (dependency_receipts is None):
        raise ValueError("dependency contract and receipts must be supplied together")
    if dependency_contract is not None:
        from .dependency_availability import resolve
        samples, _, dependency_report = resolve(samples, dependency_contract, dependency_receipts)
    if unavailable_anchor:
        samples=samples.copy()
        samples.loc[samples.sample_id.isin(unavailable_anchor),'feature_available_at']=None
    assignments, split_report = assign_segments(samples, boundaries, evaluation_at=evaluation_at)
    if set(values.sample_id) != set(samples.sample_id):
        raise ValueError("feature and sample universes differ")
    frame = values.set_index("sample_id").loc[samples.sample_id, features].copy()
    for column in features:
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError("features must be numeric, not boolean or text")
    array = frame.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(array).any():
        raise ValueError("infinite feature value")
    ids = training_ids(assignments, "fit")
    if not ids:
        raise ValueError("no eligible fit samples")
    fit = frame.loc[ids].astype(float)
    median = fit.median().fillna(0.0)
    filled = fit.fillna(median)
    center = filled.mean()
    raw_scale = filled.std(ddof=0)
    scale = raw_scale.mask(raw_scale.eq(0), 1.0)
    if not np.isfinite(np.concatenate([median.values, center.values, scale.values])).all():
        raise ValueError("nonfinite fitted feature statistics")
    transformed = (frame.astype(float).fillna(median) - center) / scale
    if not np.isfinite(transformed.to_numpy()).all():
        raise ValueError("nonfinite transformed features")
    # Availability is eligibility, not an ordinary missing numeric observation.
    ready = assignments.feature_ready.to_numpy()
    transformed.iloc[np.flatnonzero(~ready), :] = np.nan
    transformed = transformed.reset_index()
    report = {
        "split": split_report, "fit_sample_ids": ids, "feature_columns": features,
        "median": median.to_dict(), "center": center.to_dict(), "scale": scale.to_dict(),
        "all_missing_fit_columns": fit.columns[fit.isna().all()].tolist(),
        "constant_fit_columns": raw_scale.index[raw_scale.eq(0)].tolist(),
        "feature_unavailable_rows": int((~ready).sum()),
        "all_candidates_retained": len(transformed) == len(samples),
        "lineage_independently_verified": False, "formal_training_authorized": False,
    }
    if dependency_report is not None:
        report["dependency_availability"] = dependency_report
    if anchor_report is not None:
        report['anchor_admission']=anchor_report
    return transformed, assignments, report
