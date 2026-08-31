#!/usr/bin/env python
"""Evaluate S20 candidates by daily high-confidence stock-selection metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.s20 import daily_topk_metrics  # noqa: E402


CANDIDATES = {
    "b1_raw": "b1_raw_upside",
    "b1_platt": "s20_probability",
    "b0_retargeted": "b0_probability",
    "b1_raw_b0_50_posthoc": "b1_raw_b0_50",
    "b1_platt_b0_50_posthoc": "b1_platt_b0_50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT
        / "output/experiments/s20_b1_competing_risk_50pct/walk_forward_predictions.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "config/s20_high_confidence_evaluation.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.predictions)
    # Older B1 artifacts predate explicit blend columns; reconstruct them exactly.
    frame["b1_raw_b0_50"] = (
        frame["b1_raw_upside"] + frame["b0_probability"]
    ) / 2
    frame["b1_platt_b0_50"] = (
        frame["s20_probability"] + frame["b0_probability"]
    ) / 2
    evaluations = []
    for fold_name, fold in [("aggregate", frame), *frame.groupby("fold")]:
        for candidate, column in CANDIDATES.items():
            for k in (10, 20, 50):
                evaluations.append(
                    {
                        "fold": fold_name,
                        "candidate": candidate,
                        **daily_topk_metrics(
                            fold,
                            probability_col=column,
                            target_col="target25_up_first",
                            k=k,
                        ),
                    }
                )
    payload = {
        "contract_version": "s20-high-confidence-evaluation-v1",
        "objective": "find a small daily set with high probability of hitting +25% before -15% within 20 sessions",
        "status": "retrospective_on_50pct_sample",
        "warning": "posthoc blend candidates are diagnostic only; daily top-k must be confirmed on full-universe untouched dates",
        "evaluations": evaluations,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = pd.DataFrame(evaluations)
    print(
        summary[summary["fold"] == "aggregate"][
            ["candidate", "k", "dates", "precision", "lift", "positive_pick_days_rate"]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
