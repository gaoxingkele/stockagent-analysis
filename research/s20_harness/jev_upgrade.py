"""Does a Jev judgment add discrimination beyond the R20/S20 scores?

Substrate: the frozen 2026-01-27..08-05 confirmation matrix (613,500 rows, 126
days) with the realized ``positive20`` outcome and four model signals. The state
given to Jev is numeric only -- no instrument code and no date -- so there is no
identity/period channel for the model to leak through. Any skill it shows is
genuine composition of the numbers already available to the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
MYLIB = Path("D:/aicoding/mylib")
CONFIRMATION = ROOT / "output/experiments/s20_20r_confirmation/run_v1/predictions.parquet"
SIGNALS = ("s20_20r_rank", "stage1_probability", "lambdarank_score", "r20_p20_reference")
SAMPLE_PER_CLASS = 180
SEED = 20260926

INSTRUCTIONS = (
    "Over the next 20 A-share trading sessions, with tomorrow's opening price as "
    "the entry: will the price first touch +20% above the entry before it first "
    "breaks -10% below the entry, and after touching +20% stay above the entry "
    "through session 20? Judge from the model signals only."
)
CRITERIA = {
    "true": "the +20% target is reached first and the post-target path stays above entry",
    "false": "the -10% line is broken first, or the target is never reached, or the post-target path falls back below entry",
}


def auc(scores, labels) -> float | None:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if positive.size == 0 or negative.size == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def sample_matrix(frame: pd.DataFrame, *, per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for value in (1, 0):
        pool = frame[frame.positive20.eq(value)]
        take = min(per_class, len(pool))
        parts.append(pool.sample(n=take, random_state=int(rng.integers(2 ** 31))))
    return pd.concat(parts, ignore_index=True)


def run(frame: pd.DataFrame, *, output_dir: Path, per_class: int = SAMPLE_PER_CLASS,
        seed: int = SEED, progress=None) -> dict:
    sys.path.insert(0, str(MYLIB))
    from jev import get_client, noul

    sample = sample_matrix(frame, per_class=per_class, seed=seed)
    client = get_client()
    jev_scores, failures = [], 0
    for position, row in enumerate(sample.itertuples(index=False)):
        state = {"model_signals": {name: round(float(getattr(row, name)), 6)
                                   for name in SIGNALS}}
        try:
            value = float(noul(state, INSTRUCTIONS, CRITERIA, client=client))
            jev_scores.append(value)
        except Exception:
            jev_scores.append(np.nan)
            failures += 1
        if progress and (position + 1) % 30 == 0:
            progress(position + 1, len(sample))

    result = sample.copy()
    result["jev_noul"] = jev_scores
    result["positive"] = result.positive20.astype(bool)
    rows = []
    for name in (*SIGNALS, "jev_noul"):
        known = result[[name, "positive"]].dropna()
        rows.append({"signal": name, "scored": int(len(known)),
                     "auc_positive20": auc(known[name], known.positive),
                     "mean": float(known[name].mean()) if len(known) else None})
    baseline = {name: auc(result[name], result.positive) for name in SIGNALS}
    known = result.dropna(subset=["jev_noul"])
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    combination_aucs = {"r20_reference_plus_jev": [],
                        "s20_rank_plus_jev": [],
                        "all_signals_plus_jev": [],
                        "all_signals_only": []}
    matrix = known[list(SIGNALS) + ["jev_noul"]].to_numpy(dtype=float)
    labels = known.positive.to_numpy(dtype=int)
    for train_index, test_index in skf.split(matrix, labels):
        for key, columns in (
            ("r20_reference_plus_jev", [SIGNALS.index("r20_p20_reference"), len(SIGNALS)]),
            ("s20_rank_plus_jev", [SIGNALS.index("s20_20r_rank"), len(SIGNALS)]),
            ("all_signals_plus_jev", list(range(len(SIGNALS) + 1))),
            ("all_signals_only", list(range(len(SIGNALS)))),
        ):
            model = LogisticRegression(max_iter=2000).fit(matrix[train_index][:, columns],
                                                          labels[train_index])
            probability = model.predict_proba(matrix[test_index][:, columns])[:, 1]
            combination_aucs[key].append(auc(probability, labels[test_index]))
    combination = {key: {"mean": float(np.mean(values)),
                         "fold_aucs": [float(value) for value in values]}
                   for key, values in combination_aucs.items()}
    correlation = known[list(SIGNALS) + ["jev_noul", "positive"]].corr()["jev_noul"].round(4).to_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    known.to_csv(output_dir / "rows.csv", index=False)
    summary = {
        "substrate": "s20_20r_confirmation/run_v1/predictions.parquet",
        "event": "positive20 (S20-20 strict path: +20% first vs -10% first, "
                 "post-target path stays above entry)",
        "state": "numeric model signals only; no instrument code or date",
        "sample": {"rows": int(len(result)), "positive_rate": float(result.positive.mean()),
                   "jev_scored": int(len(known)), "jev_failures": int(failures)},
        "baseline_auc": baseline,
        "jev_auc": auc(known.jev_noul, known.positive),
        "jev_mean_score": float(known.jev_noul.mean()),
        "jev_auc_gain_vs_best_baseline": float(
            auc(known.jev_noul, known.positive) - max(baseline.values())),
        "combination_cv_auc": combination,
        "jev_correlation": correlation,
        "model": "jev-latest",
        "leakage_control": "identity and date withheld from the state; "
                           "numeric-only judgment cannot localize the outcome",
        "conclusion_guard": "confirmation window was already consumed by the 2026-09-07 "
                            "report; this is an exploration, not independent validation",
    }
    return summary
