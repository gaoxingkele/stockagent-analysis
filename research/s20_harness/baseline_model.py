"""Isolated fixed logistic baseline; not a formal H03 stage executor."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .feature_pipeline import prepare


def validate_seed(seed):
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('random_seed must be an integer in [0, 2**32-1]')
    return seed


def pipeline_costs(plan):
    validate_seed(plan.get('random_seed', 20))
    family=plan.get('model_family','logistic')
    if family not in {'logistic','shallow_tree','mature_frequency'}: raise ValueError('unsupported fixed baseline family')
    fits=0 if family=='mature_frequency' else 1
    return dict(model_fits=fits,underlying_fits=fits+1,calibrator_fits=1,policy_evaluations=1)


def run(samples, features, fit_labels, boundaries, feature_contract, *, target_id, model_family='logistic', anchor_context=None, random_seed=20):
    """Fit once, without access to tune/calibration/selection/outer outcomes.

    Labels must contain precisely the eligible fit IDs and a boolean target.
    Predictions on fit/outside rows are withheld, not passed off as OOF.
    Later unknown labels do not suppress predictions. Output is uncalibrated.
    Segment times simulate a research fold, not real historical model receipts.
    """
    validate_seed(random_seed)
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("explicit versioned target identity required")
    if model_family not in {'logistic', 'shallow_tree','mature_frequency'}:
        raise ValueError('unsupported fixed baseline family')
    matrix, assignment, preprocessing = prepare(samples, features, boundaries, feature_contract,
        anchor_context=anchor_context)
    if fit_labels.columns.duplicated().any() or set(fit_labels.columns) != {"sample_id", "target"}:
        raise ValueError("exact fit label schema required")
    if fit_labels.sample_id.isna().any() or fit_labels.sample_id.duplicated().any():
        raise ValueError("unique fit label IDs required")
    ids = preprocessing["fit_sample_ids"]
    if set(fit_labels.sample_id) != set(ids):
        raise ValueError("only the exact eligible fit labels may be consumed")
    y = fit_labels.set_index("sample_id").loc[ids, "target"]
    if not y.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("fit target must be resolved boolean")
    if not len(y): raise ValueError('nonempty mature fit labels required')
    if y.nunique() != 2 and model_family!='mature_frequency':
        raise ValueError("both fit target classes required; no silent constant fallback")
    columns = preprocessing["feature_columns"]
    indexed = matrix.set_index("sample_id")
    if model_family == 'logistic':
        parameters = dict(C=1.0, solver="lbfgs", max_iter=1000, random_state=random_seed)
        model = LogisticRegression(**parameters)
    elif model_family=='shallow_tree':
        parameters = dict(max_depth=3, min_samples_leaf=5, random_state=random_seed)
        model = DecisionTreeClassifier(**parameters)
    else:
        parameters=dict(smoothing=0);model=None
    if model is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(indexed.loc[ids, columns], y.astype(int))
    output = assignment[["sample_id", "segment", "feature_ready"]].copy().reset_index(drop=True)
    output["raw_probability"] = np.nan
    output["prediction_status"] = "not_forward_segment"
    forward = output.segment.isin(["tune", "calibration", "selection-policy", "outer-test"])
    output.loc[forward, "prediction_status"] = "feature_unavailable"
    ready = forward & output.feature_ready
    if ready.any():
        prediction_ids = output.loc[ready, "sample_id"]
        probability = (np.full(len(prediction_ids),float(y.mean())) if model is None
                       else model.predict_proba(indexed.loc[prediction_ids, columns])[:, 1])
        if not np.isfinite(probability).all():
            raise ValueError("nonfinite baseline probability")
        output.loc[ready, "raw_probability"] = probability
        output.loc[ready, "prediction_status"] = "uncalibrated_reference_prediction"
    card = {
        "target_id": target_id, "model": {'logistic':'fixed_logistic_reference','shallow_tree':'fixed_shallow_tree_reference','mature_frequency':'mature_empirical_frequency'}[model_family], "parameters": parameters,
        "sklearn_version": sklearn.__version__, "model_level_fits": 0 if model is None else 1,
        "randomness": {
            "requested_seed": random_seed,
            "seed_effect": {'logistic': 'unused_by_lbfgs', 'shallow_tree': 'feature_permutation_and_split_ties',
                            'mature_frequency': 'deterministic_frequency'}[model_family],
            "independent_market_evidence": False,
        },
        "fit_sample_ids": ids, "fit_positive_count": int(y.sum()),
        "classes": [0,1] if model is None else model.classes_.tolist(),
        "preprocessing": preprocessing, "rows": len(output),
        "predicted_rows": int(ready.sum()), "all_candidates_retained": True,
        "fit_predictions_are_oof": False, "calibration_performed": False,
        "historical_model_availability_proven": False, "formal_training_authorized": False,
        "formal_H03_accepted": False, "production_eligible": False,
    }
    if model_family == 'logistic':
        card.update(coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),iterations=model.n_iter_.tolist())
    elif model_family=='shallow_tree':
        tree=model.tree_
        card['tree_state']=dict(children_left=tree.children_left.tolist(),children_right=tree.children_right.tolist(),
            feature=tree.feature.tolist(),threshold=tree.threshold.tolist(),value=tree.value.tolist(),
            node_samples=tree.n_node_samples.tolist())
    else:
        card.update(probability=float(y.mean()),label_frequency_estimates=1)
    return output, card
