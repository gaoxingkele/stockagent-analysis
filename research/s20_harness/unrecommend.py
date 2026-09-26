"""Failure-rate analysis for the daily recommendation lists (unR20 / unS20).

For each frozen signal we build the daily Top20 list, split the selected names by
the realized ``positive20`` outcome over the next 20 sessions, and ask:

1. What is the failure rate of the list itself?
2. Do the failures look different from the hits on the stored factors?
3. Can a small classifier estimate the failure probability among recommended
   names from those factors (a reusable negative-sample training set)?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "output/experiments/s20_20r_confirmation/run_v1/predictions.parquet"
FACTOR_DIR = ROOT / "output/experiments/s20_20r_confirmation/factor_groups"
META = ("ts_code", "trade_date", "industry")
SCORES = ("s20_20r_rank", "stage1_probability", "lambdarank_score", "r20_p20_reference")
OUTCOMES = ("positive20", "class20")

# list_name -> (signal, ascending) ; higher signal = stronger recommendation
LISTS = (
    ("unR20", "r20_p20_reference"),
    ("unS20", "s20_20r_rank"),
    ("unS20_stage1", "stage1_probability"),
)
TOP_K = 20
CV_FOLDS = 5
SEED = 20260926


def load_joined() -> pd.DataFrame:
    predictions = pd.read_parquet(PREDICTIONS)
    groups = pd.concat([pd.read_parquet(path) for path in sorted(FACTOR_DIR.glob("group_*.parquet"))],
                       ignore_index=True)
    return predictions.merge(groups, on=["ts_code", "trade_date"], how="inner")


def factor_columns(frame: pd.DataFrame) -> list[str]:
    used = set(META) | set(SCORES) | set(OUTCOMES)
    return [column for column in frame.columns if column not in used
            and pd.api.types.is_numeric_dtype(frame[column])]


def daily_topk(frame: pd.DataFrame, signal: str, k: int = TOP_K) -> pd.DataFrame:
    ranked = frame.sort_values(["trade_date", signal], ascending=[True, False])
    top = ranked.groupby("trade_date", sort=False).head(k).copy()
    top["fail"] = top.positive20.eq(0)
    return top.reset_index(drop=True)


def feature_separation(top: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Standardised difference between failures and hits for every factor."""
    rows = []
    fail = top.fail
    for name in features:
        values = top[name]
        hit = values[~fail].dropna()
        miss = values[fail].dropna()
        if len(hit) < 20 or len(miss) < 20:
            continue
        pooled = float(np.sqrt((hit.var(ddof=1) + miss.var(ddof=1)) / 2))
        if not np.isfinite(pooled) or pooled == 0:
            continue
        rows.append({"feature": name,
                     "mean_hit": float(hit.mean()), "mean_fail": float(miss.mean()),
                     "std_diff": float((miss.mean() - hit.mean()) / pooled),
                     "n_hit": int(len(hit)), "n_fail": int(len(miss))})
    table = pd.DataFrame(rows).sort_values("std_diff", key=lambda s: s.abs(), ascending=False)
    return table.reset_index(drop=True)


def failure_classifier_auc(top: pd.DataFrame, features: list[str]) -> dict:
    known = top[["fail", *features]].dropna()
    matrix = known[features].to_numpy(dtype=float)
    labels = known.fail.to_numpy(dtype=int)
    if len(set(labels)) < 2:
        return {"cv_auc": None, "fold_aucs": [], "n": int(len(known)), "reason": "one class"}
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    fold_aucs = []
    for train_index, test_index in skf.split(matrix, labels):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=0.3))
        model.fit(matrix[train_index], labels[train_index])
        probability = model.predict_proba(matrix[test_index])[:, 1]
        fold_aucs.append(_auc(probability, labels[test_index]))
    return {"cv_auc": float(np.mean(fold_aucs)), "fold_aucs": [float(v) for v in fold_aucs],
            "n": int(len(known))}


def _auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive, negative = ranks[labels], ranks[~labels]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    return float((positive.sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def run(output_dir: Path) -> dict:
    frame = load_joined()
    features = factor_columns(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"rows": int(len(frame)), "features": len(features), "top_k": TOP_K,
               "lists": {}}
    lists = []
    for list_name, signal in LISTS:
        top = daily_topk(frame, signal)
        separation = feature_separation(top, features)
        classifier = failure_classifier_auc(top, features)
        summary["lists"][list_name] = {
            "signal": signal,
            "days": int(top.trade_date.nunique()),
            "selected": int(len(top)),
            "failures": int(top.fail.sum()),
            "failure_rate": float(top.fail.mean()),
            "hit_rate": float(top.positive20.mean()),
            "failure_classifier_cv_auc": classifier["cv_auc"],
            "top_separating_features": separation.head(12).to_dict("records"),
        }
        top.to_parquet(output_dir / (list_name + "_list.parquet"), index=False)
        top[top.fail].to_parquet(output_dir / (list_name + "_failures.parquet"), index=False)
        lists.append(top.assign(list_name=list_name))
    pd.concat(lists, ignore_index=True).to_parquet(output_dir / "lists_combined.parquet",
                                                   index=False)
    summary["note"] = ("failure = positive20 == 0 within each daily Top20 list; "
                       "window 2026-01-27..08-05, already consumed by the 09-07 report")
    return summary
