#!/usr/bin/env python
"""Evaluate the frozen S20-v2 promotion gate from training artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_FOLDS = ("wf1", "wf2", "wf3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_v2_multitarget",
    )
    parser.add_argument("--target", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    topk = pd.read_csv(args.input_dir / "daily_topk_metrics.csv")
    probability = pd.read_csv(args.input_dir / "probability_metrics.csv")
    ranking = topk[
        (topk["target_pct"] == args.target)
        & (topk["k"] == 20)
        & (topk["candidate"] == "raw_seed_ensemble")
        & topk["fold"].isin(TEST_FOLDS)
    ].copy()
    baseline = topk[
        (topk["target_pct"] == args.target)
        & (topk["k"] == 20)
        & (topk["candidate"] == "b0_retargeted")
        & topk["fold"].isin(TEST_FOLDS)
    ][["fold", "precision"]].rename(columns={"precision": "b0_precision"})
    ranking = ranking.merge(baseline, on="fold", validate="one_to_one")
    ranking["precision_delta_vs_b0"] = ranking["precision"] - ranking["b0_precision"]

    calibrated = probability[
        (probability["target_pct"] == args.target)
        & (probability["candidate"] == "platt_seed_ensemble")
        & probability["fold"].isin(TEST_FOLDS)
    ].copy()
    calibrated["selected_tail_error"] = (
        calibrated["top_decile_event_rate"]
        - calibrated["top_decile_mean_probability"]
    ).abs()

    checks = {
        "daily_top20_lift_each_fold_gt_1_5": bool((ranking["lift"] > 1.5).all()),
        "single_fold_precision_regression_vs_b0_lte_2pp": bool(
            (ranking["precision_delta_vs_b0"] >= -0.02).all()
        ),
        "brier_skill_positive_each_fold": bool(
            (calibrated["brier_skill_vs_constant"] > 0).all()
        ),
        "global_ece_each_fold_lte_0_05": bool((calibrated["ece_10"] <= 0.05).all()),
        "selected_tail_error_each_fold_lte_0_05": bool(
            (calibrated["selected_tail_error"] <= 0.05).all()
        ),
    }
    eligible = all(checks.values())
    aggregate = topk[
        (topk["target_pct"] == args.target)
        & (topk["k"] == 20)
        & (topk["candidate"] == "raw_seed_ensemble")
        & (topk["fold"] == "aggregate")
    ].iloc[0]
    result = {
        "contract_version": "s20-v2-five-target-20260903",
        "target_pct": args.target,
        "status": "eligible_for_full_universe_confirmation"
        if eligible
        else "offline_completed_not_shadow_eligible",
        "promotion_eligible": eligible,
        "checks": checks,
        "aggregate_top20": {
            "dates": int(aggregate["dates"]),
            "base_rate": float(aggregate["base_rate"]),
            "precision": float(aggregate["precision"]),
            "lift": float(aggregate["lift"]),
            "positive_pick_days_rate": float(aggregate["positive_pick_days_rate"]),
        },
        "fold_ranking": ranking[
            [
                "fold",
                "base_rate",
                "precision",
                "lift",
                "positive_pick_days_rate",
                "b0_precision",
                "precision_delta_vs_b0",
            ]
        ].to_dict(orient="records"),
        "fold_calibration": calibrated[
            [
                "fold",
                "brier_skill_vs_constant",
                "ece_10",
                "selected_tail_error",
                "roc_auc",
                "pr_auc",
            ]
        ].to_dict(orient="records"),
        "full_universe_confirmation_run": False,
        "shadow_run_started": False,
        "decision": "Stop before full-universe and shadow stages; retain R20 as production index.",
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    output = args.output or args.input_dir / "gate_report.json"
    output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
