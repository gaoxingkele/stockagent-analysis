"""Does pruning Stage-1 Top100 with a learned failure score make it safer?

The honest answer depends on how well failures separate from hits *within* the
list. Failure probabilities are produced out-of-fold, then the list is pruned at
several retention levels. Two controls are reported: no pruning, and pruning by
the selection score itself (drop the lowest stage1_probability names).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .unrecommend import factor_columns, load_joined

ROOT = Path(__file__).resolve().parents[2]
TOP_K = 100
FOLDS = 5
SEED = 20260926
RETAIN = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def out_of_fold_failure_probability(top: pd.DataFrame, features: list[str]) -> pd.Series:
    known = top.dropna(subset=features)
    matrix = known[features].to_numpy(dtype=float)
    labels = known.fail.to_numpy(dtype=int)
    probability = pd.Series(np.nan, index=known.index, dtype=float)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for train_index, test_index in skf.split(matrix, labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3))
        model.fit(matrix[train_index], labels[train_index])
        probability.iloc[test_index] = model.predict_proba(matrix[test_index])[:, 1]
    return probability.reindex(top.index)


def prune(top: pd.DataFrame, score: pd.Series, retain: float) -> dict:
    """Drop the most likely failures (or the weakest picks), keep ``retain``."""
    threshold = float(score.quantile(retain))
    kept = top[score.le(threshold)] if retain < 1.0 else top
    if len(kept) == 0:
        return {"retain": retain, "kept": 0, "hit_rate": None, "lift": None,
                "failure_rate": None, "composition": {}}
    base = float(top.positive20.mean())
    composition = kept.loc[kept.positive20.eq(0), "class20"].value_counts().to_dict()
    return {"retain": retain, "kept": int(len(kept)),
            "hit_rate": float(kept.positive20.mean()),
            "lift": float(kept.positive20.mean() / base),
            "failure_rate": float(kept.fail.mean()),
            "failure_composition": {str(k): int(v) for k, v in composition.items()}}


def run(output_dir: Path) -> dict:
    frame = load_joined()
    features = factor_columns(frame)
    top = frame.sort_values(["trade_date", "stage1_probability"],
                            ascending=[True, False]) \
        .groupby("trade_date", sort=False).head(TOP_K).copy()
    top["fail"] = top.positive20.eq(0)
    failure_score = out_of_fold_failure_probability(top, features)

    rows = []
    for retain in RETAIN:
        rows.append({"strategy": "no_prune" if retain == 1.0 else "learned_failure_prune",
                     **prune(top, failure_score, retain)})
    stage1_score = top.stage1_probability
    for retain in (0.9, 0.8, 0.7, 0.6, 0.5):
        rows.append({"strategy": "drop_lowest_stage1",
                     **prune(top, -stage1_score, retain)})
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "prune_curve.csv", index=False)
    summary = {
        "substrate": "s20_20r_confirmation (2026-01-27..08-05)",
        "list": "stage1_probability daily Top100",
        "rows": int(len(top)), "days": int(top.trade_date.nunique()),
        "base_hit_rate": float(top.positive20.mean()),
        "failure_classifier_cv_auc": _cv_auc(top, features),
        "class20_meaning": {"0": "positive20",
                            "1/2/3": "the three negative tiers (N1/N2/N3), "
                                     "order unverified in this artifact"},
        "curve": rows,
        "conclusion_guard": "window already consumed; exploration only",
    }
    return summary


def _cv_auc(top: pd.DataFrame, features: list[str]) -> float:
    known = top.dropna(subset=features)
    matrix = known[features].to_numpy(dtype=float)
    labels = known.fail.to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    for train_index, test_index in skf.split(matrix, labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3))
        model.fit(matrix[train_index], labels[train_index])
        probability = model.predict_proba(matrix[test_index])[:, 1]
        aucs.append(_auc(probability, labels[test_index]))
    return float(np.mean(aucs))


def _auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive, negative = ranks[labels], ranks[~labels]
    return float((positive.sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))
