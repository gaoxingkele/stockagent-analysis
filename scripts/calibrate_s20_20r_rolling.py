#!/usr/bin/env python
"""Simulate matured-label rolling calibration for the S20-20R ranker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.r20_target_prob import probability_metrics  # noqa: E402
from stockagent_analysis.s20 import purged_walk_forward_masks  # noqa: E402
from train_s20_20r_ranker import _add_rank_features, _daily_percentile  # noqa: E402
from train_s20_20r_residual import (  # noqa: E402
    PORTABLE_EXCLUDED_FEATURES,
    _logit,
    _prepare,
    _residual_probability,
)
from train_s20_v2_multitarget import S20_V2_FOLDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-bps", type=int, default=5000)
    parser.add_argument(
        "--feature-mode", choices=("portable", "all"), default="portable"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "output/experiments/s20_v2_labels/labels.parquet",
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_residual_v11",
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_ranker",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_calibration",
    )
    parser.add_argument("--update-sessions", type=int, default=5)
    parser.add_argument("--rolling-sessions", type=int, default=250)
    return parser.parse_args()


def _calibration_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(frame)),
            _logit(frame["stage1_probability"].to_numpy()),
            _logit(frame["hybrid_rank"].to_numpy()),
        ]
    )


def _fit_bounded_head(frame: pd.DataFrame, initial=None) -> np.ndarray:
    x = _calibration_matrix(frame)
    y = frame["positive20"].to_numpy(dtype=float)
    start = np.asarray(initial if initial is not None else [0.0, 1.0, 0.5])

    def objective(parameters):
        logits = np.clip(x @ parameters, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        loss = -np.mean(
            y * np.log(np.clip(probability, 1e-12, 1))
            + (1 - y) * np.log(np.clip(1 - probability, 1e-12, 1))
        )
        penalty = 0.001 * ((parameters[1] - 1.0) ** 2 + parameters[2] ** 2)
        return float(loss + penalty)

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (0.0, 3.0), (0.0, 3.0)],
    )
    if not result.success:
        raise RuntimeError(f"bounded calibration failed: {result.message}")
    return result.x


def _apply_head(frame: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
    logits = np.clip(_calibration_matrix(frame) @ parameters, -30, 30)
    return 1 / (1 + np.exp(-logits))


def _rolling_calibrate(calibration, test, update_sessions, rolling_sessions):
    test_dates = sorted(test["trade_date"].astype(str).unique())
    predictions = []
    parameters = None
    parameter_rows = []
    for date_index, trade_date in enumerate(test_dates):
        matured = test[
            (test["trade_date"] < trade_date)
            & (test["horizon_end_date"] < trade_date)
        ]
        eligible = pd.concat([calibration, matured], ignore_index=True)
        eligible_dates = sorted(eligible["trade_date"].astype(str).unique())
        if len(eligible_dates) > rolling_sessions:
            eligible = eligible[
                eligible["trade_date"] >= eligible_dates[-rolling_sessions]
            ]
        if parameters is None or date_index % update_sessions == 0:
            parameters = _fit_bounded_head(eligible, parameters)
            parameter_rows.append(
                {
                    "asof_trade_date": trade_date,
                    "training_rows": len(eligible),
                    "training_dates": len(eligible_dates),
                    "intercept": float(parameters[0]),
                    "anchor_coefficient": float(parameters[1]),
                    "rank_coefficient": float(parameters[2]),
                }
            )
        current = test[test["trade_date"] == trade_date].copy()
        current["s20_20r_probability"] = _apply_head(current, parameters)
        predictions.append(current)
    return pd.concat(predictions, ignore_index=True), parameter_rows


def _build_scores(frame, features, selected, anchor, residual_models, rankers):
    result = frame.copy()
    result["r20_anchor"] = anchor.predict(result[features])
    result["stage1_probability"], _ = _residual_probability(
        residual_models, result, features
    )
    result, rank_columns = _add_rank_features(result, selected)
    result["r20_anchor_feature"] = result["r20_anchor"].astype("float32")
    rank_features = [*features, "r20_anchor_feature", *rank_columns]
    result["lambdarank_score"] = np.mean(
        [model.predict(result[rank_features]) for model in rankers], axis=0
    )
    result["hybrid_rank"] = 0.5 * _daily_percentile(
        result, "stage1_probability"
    ) + 0.5 * _daily_percentile(result, "lambdarank_score")
    return result


def _selected_top20_error(frame):
    selected = (
        frame.sort_values(
            ["trade_date", "hybrid_rank", "ts_code"],
            ascending=[True, False, True],
        )
        .groupby("trade_date", sort=False)
        .head(20)
    )
    return {
        "selected_rows": len(selected),
        "event_rate": float(selected["positive20"].mean()),
        "mean_probability": float(selected["s20_20r_probability"].mean()),
        "absolute_error": float(
            abs(
                selected["positive20"].mean()
                - selected["s20_20r_probability"].mean()
            )
        ),
    }


def main() -> int:
    args = parse_args()
    data, features, audit = _prepare(args)
    if args.feature_mode == "portable":
        features = [f for f in features if f not in PORTABLE_EXCLUDED_FEATURES]
    audit["feature_mode"] = args.feature_mode
    audit["model_feature_count"] = len(features)
    rank_report = json.loads(
        (args.stage2_dir / "report.json").read_text(encoding="utf-8")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_parts = []
    fold_rows = []
    parameter_rows = []
    parity_rows = []
    saved_stage2 = pd.read_parquet(args.stage2_dir / "predictions.parquet")
    for fold in S20_V2_FOLDS:
        masks = purged_walk_forward_masks(
            data["trade_date"], data["horizon_end_date"], fold
        )
        valid = data["positive20"] >= 0
        calibration = data.loc[masks["calibration"] & valid].copy()
        test = data.loc[masks["test"] & valid].copy()
        anchor = lgb.Booster(
            model_file=str(args.stage1_dir / fold.name / "r20_anchor_refit.txt")
        )
        residual_models = [
            lgb.Booster(model_file=str(path))
            for path in sorted((args.stage1_dir / fold.name).glob("s20_residual_seed*.txt"))
        ]
        rankers = [
            lgb.Booster(model_file=str(path))
            for path in sorted((args.stage2_dir / fold.name).glob("ranker_seed*.txt"))
        ]
        selected = next(
            row["selected_rank_features"]
            for row in rank_report["models"]
            if row["fold"] == fold.name
        )
        calibration_scores = _build_scores(
            calibration, features, selected, anchor, residual_models, rankers
        )
        test_scores = _build_scores(
            test, features, selected, anchor, residual_models, rankers
        )
        reference = saved_stage2[saved_stage2["fold"] == fold.name][
            ["ts_code", "trade_date", "hybrid_rank"]
        ].rename(columns={"hybrid_rank": "saved_hybrid_rank"})
        parity = test_scores.merge(
            reference, on=["ts_code", "trade_date"], validate="one_to_one"
        )
        parity_rows.append(
            {
                "fold": fold.name,
                "max_hybrid_rank_difference": float(
                    (parity["hybrid_rank"] - parity["saved_hybrid_rank"])
                    .abs()
                    .max()
                ),
            }
        )
        calibrated, parameters = _rolling_calibrate(
            calibration_scores,
            test_scores,
            args.update_sessions,
            args.rolling_sessions,
        )
        calibrated["fold"] = fold.name
        metrics = probability_metrics(
            calibrated["positive20"], calibrated["s20_20r_probability"]
        )
        selected_metrics = _selected_top20_error(calibrated)
        fold_rows.append(
            {"fold": fold.name, **metrics, "selected_top20": selected_metrics}
        )
        parameter_rows.extend(
            [{"fold": fold.name, **row} for row in parameters]
        )
        prediction_parts.append(calibrated)
        print(fold.name, metrics, selected_metrics, flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    aggregate = probability_metrics(
        predictions["positive20"], predictions["s20_20r_probability"]
    )
    aggregate_selected = _selected_top20_error(predictions)
    checks = {
        "brier_skill_positive_each_fold": all(
            row["brier_skill_vs_constant"] > 0 for row in fold_rows
        ),
        "ece_each_fold_lte_0_05": all(row["ece_10"] <= 0.05 for row in fold_rows),
        "selected_top20_probability_error_each_fold_lte_0_05": all(
            row["selected_top20"]["absolute_error"] <= 0.05
            for row in fold_rows
        ),
    }
    report = {
        "contract_version": "s20-20r-r20-anchor-v1.4-20260907",
        "status": "development_complete_ready_to_freeze_confirmation"
        if all(checks.values())
        else "calibration_failed_not_ready_for_confirmation",
        "data_audit": audit,
        "rolling_calibration": {
            "update_sessions": args.update_sessions,
            "rolling_sessions": args.rolling_sessions,
            "coefficient_bounds": [0.0, 3.0],
            "intercept_bounds": [-5.0, 5.0],
        },
        "parity_audit": parity_rows,
        "fold_metrics": fold_rows,
        "aggregate_metrics": aggregate,
        "aggregate_selected_top20": aggregate_selected,
        "checks": checks,
        "passed": all(checks.values()),
        "new_confirmation_opened": False,
    }
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        args.output_dir / "calibration_parameters.csv", index=False
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
