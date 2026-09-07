#!/usr/bin/env python
"""Train the leakage-safe R20-anchored S20-20 residual baseline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.r20_target_prob import probability_metrics  # noqa: E402
from stockagent_analysis.s20 import (  # noqa: E402
    anchored_residual_probability,
    daily_topk_metrics,
    purged_walk_forward_masks,
)
from train_s20_v2_multitarget import (  # noqa: E402
    COMPARABLE_SAMPLE_SEED,
    S20_V2_FOLDS,
    _prepare,
)


DEFAULT_SEEDS = (20260907, 20260921, 20261005)
MAX_DELTA_LOGIT = 1.0
PORTABLE_EXCLUDED_FEATURES = {
    "winner_rate",
    "holder_pct",
    "total_mv",
    "pe",
    "pe_ttm",
    "pb",
    "market_score_adj",
    "mf_divergence",
    "mf_strength",
    "mf_consecutive",
}


@dataclass(frozen=True)
class NestedAnchorPlan:
    fold: str
    anchor_fit_end: str
    anchor_tune_start: str
    anchor_tune_end: str
    residual_fit_start: str


NESTED_PLANS = {
    "wf1": NestedAnchorPlan("wf1", "20240430", "20240501", "20240630", "20240701"),
    "wf2": NestedAnchorPlan("wf2", "20240831", "20240901", "20241031", "20241101"),
    "wf3": NestedAnchorPlan("wf3", "20241231", "20250101", "20250228", "20250301"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-bps", type=int, default=5000)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--feature-mode", choices=("portable", "all"), default="portable"
    )
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "output/experiments/s20_v2_labels/labels.parquet",
    )
    parser.add_argument(
        "--b0-predictions",
        type=Path,
        default=ROOT / "output/experiments/r20_target_prob_v2/walk_forward_predictions.parquet",
    )
    parser.add_argument(
        "--s20-v2-predictions",
        type=Path,
        default=ROOT / "output/experiments/s20_v2_multitarget/predictions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_residual",
    )
    return parser.parse_args()


def _dataset(
    frame: pd.DataFrame,
    features: list[str],
    label: str,
    *,
    init_score: np.ndarray | None = None,
) -> lgb.Dataset:
    return lgb.Dataset(
        frame[features],
        label=frame[label],
        init_score=init_score,
        feature_name=features,
        free_raw_data=False,
    )


def _anchor_params(seed, threads):
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
        "num_threads": threads,
    }


def _fit_anchor(fit, tune, features, seed, threads):
    return lgb.train(
        _anchor_params(seed, threads),
        _dataset(fit, features, "target_p20_safe"),
        num_boost_round=350,
        valid_sets=[_dataset(tune, features, "target_p20_safe")],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )


def _refit_anchor(frame, features, seed, threads, iterations):
    return lgb.train(
        _anchor_params(seed, threads),
        _dataset(frame, features, "target_p20_safe"),
        num_boost_round=max(1, int(iterations)),
        callbacks=[lgb.log_evaluation(0)],
    )


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _fit_residual(fit, tune, features, seed, threads):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.025,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": seed,
        "num_threads": threads,
    }
    return lgb.train(
        params,
        _dataset(
            fit,
            features,
            "positive20",
            init_score=_logit(fit["r20_anchor"].to_numpy()),
        ),
        num_boost_round=250,
        valid_sets=[
            _dataset(
                tune,
                features,
                "positive20",
                init_score=_logit(tune["r20_anchor"].to_numpy()),
            )
        ],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )


def _residual_probability(models, frame, features):
    delta = np.mean(
        [
            model.predict(
                frame[features], num_iteration=model.best_iteration, raw_score=True
            )
            for model in models
        ],
        axis=0,
    )
    return anchored_residual_probability(
        frame["r20_anchor"], delta, max_absolute_residual=MAX_DELTA_LOGIT
    ), delta


def _fit_platt(probability, actual):
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=DEFAULT_SEEDS[0])
    model.fit(_logit(probability).reshape(-1, 1), actual)
    return model


def _apply_platt(model, probability):
    return model.predict_proba(_logit(probability).reshape(-1, 1))[:, 1]


def _evaluation_rows(frame, fold, candidate, probability_column):
    probability = probability_metrics(frame["positive20"], frame[probability_column])
    topk = []
    for k in (10, 20, 50):
        topk.append(
            {
                "fold": fold,
                "candidate": candidate,
                **daily_topk_metrics(
                    frame,
                    probability_col=probability_column,
                    target_col="positive20",
                    k=k,
                ),
            }
        )
    return {"fold": fold, "candidate": candidate, **probability}, topk


def main() -> int:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one residual seed is required")
    data, features, audit = _prepare(args)
    if args.feature_mode == "portable":
        features = [f for f in features if f not in PORTABLE_EXCLUDED_FEATURES]
    audit["feature_mode"] = args.feature_mode
    audit["model_feature_count"] = len(features)
    audit["excluded_features"] = (
        sorted(PORTABLE_EXCLUDED_FEATURES)
        if args.feature_mode == "portable"
        else []
    )
    if args.sample_bps != 5000:
        print("warning: published B0/S20 comparisons only fully align at 5000 bps", flush=True)

    b0 = pd.read_parquet(
        args.b0_predictions,
        columns=["ts_code", "trade_date", "fold", "ensemble_p20"],
    )
    old_s20 = pd.read_parquet(
        args.s20_v2_predictions,
        columns=["ts_code", "trade_date", "fold", "s20_20_raw"],
    )
    for frame in (b0, old_s20):
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["trade_date"] = frame["trade_date"].astype(str)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability_rows = []
    topk_rows = []
    prediction_parts = []
    split_rows = []
    model_rows = []
    for fold in S20_V2_FOLDS:
        plan = NESTED_PLANS[fold.name]
        dates = data["trade_date"].astype(str)
        ends = data["horizon_end_date"].astype(str)
        outer = purged_walk_forward_masks(dates, ends, fold)
        valid = data["positive20"] >= 0
        anchor_fit_mask = (dates <= plan.anchor_fit_end) & (
            ends < plan.anchor_tune_start
        )
        anchor_tune_mask = dates.between(
            plan.anchor_tune_start, plan.anchor_tune_end
        ) & (ends < plan.residual_fit_start)
        residual_fit_mask = dates.between(plan.residual_fit_start, fold.fit_end) & (
            ends < fold.tune_start
        )
        anchor_fit = data.loc[anchor_fit_mask]
        anchor_tune = data.loc[anchor_tune_mask]
        outer_fit = data.loc[outer["fit"]].copy()
        residual_fit = data.loc[residual_fit_mask & valid].copy()
        residual_tune = data.loc[outer["tune"] & valid].copy()
        calibration = data.loc[outer["calibration"] & valid].copy()
        test = data.loc[outer["test"] & valid].copy()
        sizes = {
            "fold": fold.name,
            "anchor_fit": len(anchor_fit),
            "anchor_tune": len(anchor_tune),
            "outer_fit_refit": len(outer_fit),
            "residual_fit": len(residual_fit),
            "residual_tune": len(residual_tune),
            "calibration": len(calibration),
            "test": len(test),
        }
        split_rows.append(sizes)
        print(sizes, flush=True)

        fold_dir = args.output_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        early_anchor = _fit_anchor(
            anchor_fit, anchor_tune, features, seeds[0] - 503, args.num_threads
        )
        early_anchor.save_model(str(fold_dir / "r20_anchor_oof_early.txt"))
        residual_fit["r20_anchor"] = early_anchor.predict(
            residual_fit[features], num_iteration=early_anchor.best_iteration
        )
        refit_anchor = _refit_anchor(
            outer_fit,
            features,
            seeds[0] - 503,
            args.num_threads,
            early_anchor.best_iteration,
        )
        refit_anchor.save_model(str(fold_dir / "r20_anchor_refit.txt"))
        for frame in (residual_tune, calibration, test):
            frame["r20_anchor"] = refit_anchor.predict(frame[features])

        residual_models = []
        for seed in seeds:
            model = _fit_residual(
                residual_fit, residual_tune, features, seed, args.num_threads
            )
            model.save_model(str(fold_dir / f"s20_residual_seed{seed}.txt"))
            residual_models.append(model)
            model_rows.append(
                {
                    "fold": fold.name,
                    "kind": "residual",
                    "seed": seed,
                    "best_iteration": model.best_iteration,
                }
            )
        raw_cal, _ = _residual_probability(residual_models, calibration, features)
        raw_test, delta_test = _residual_probability(residual_models, test, features)
        platt = _fit_platt(raw_cal, calibration["positive20"].to_numpy())
        calibrated_test = _apply_platt(platt, raw_test)

        result = test[["ts_code", "trade_date", "positive20"]].copy()
        result["fold"] = fold.name
        result["r20_nested_anchor"] = test["r20_anchor"].to_numpy()
        result["s20_20r_raw"] = raw_test
        result["s20_20r_platt"] = calibrated_test
        result["s20_20r_delta_logit"] = delta_test
        result = result.merge(
            b0[b0["fold"] == fold.name],
            on=["ts_code", "trade_date", "fold"],
            how="left",
            validate="one_to_one",
        ).merge(
            old_s20[old_s20["fold"] == fold.name],
            on=["ts_code", "trade_date", "fold"],
            how="left",
            validate="one_to_one",
        )
        candidates = {
            "r20_nested_anchor": "r20_nested_anchor",
            "s20_20r_raw": "s20_20r_raw",
            "s20_20r_platt": "s20_20r_platt",
            "r20_b0_published": "ensemble_p20",
            "s20_v2": "s20_20_raw",
        }
        for candidate, column in candidates.items():
            subset = result[result[column].notna()]
            probability_row, candidate_topk = _evaluation_rows(
                subset, fold.name, candidate, column
            )
            probability_rows.append(probability_row)
            topk_rows.extend(candidate_topk)
        prediction_parts.append(result)
        model_rows.append(
            {
                "fold": fold.name,
                "kind": "anchor",
                "seed": seeds[0] - 503,
                "best_iteration": early_anchor.best_iteration,
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    candidates = {
        "r20_nested_anchor": "r20_nested_anchor",
        "s20_20r_raw": "s20_20r_raw",
        "s20_20r_platt": "s20_20r_platt",
        "r20_b0_published": "ensemble_p20",
        "s20_v2": "s20_20_raw",
    }
    for candidate, column in candidates.items():
        subset = predictions[predictions[column].notna()]
        probability_row, candidate_topk = _evaluation_rows(
            subset, "aggregate", candidate, column
        )
        probability_rows.append(probability_row)
        topk_rows.extend(candidate_topk)

    topk_frame = pd.DataFrame(topk_rows)
    primary = topk_frame[topk_frame["k"] == 20]
    pivot = primary.pivot(index="fold", columns="candidate", values="precision")
    fold_delta = pivot["s20_20r_raw"] - pivot["r20_b0_published"]
    aggregate_lift = float(
        primary[
            (primary["fold"] == "aggregate")
            & (primary["candidate"] == "s20_20r_raw")
        ]["lift"].iloc[0]
    )
    checks = {
        "aggregate_top20_precision_gt_r20_b0": bool(
            pivot.loc["aggregate", "s20_20r_raw"]
            > pivot.loc["aggregate", "r20_b0_published"]
        ),
        "each_fold_precision_regression_vs_r20_b0_lte_1pp": bool(
            (fold_delta.drop(index="aggregate") >= -0.01).all()
        ),
        "aggregate_top20_lift_gte_1_3": aggregate_lift >= 1.3,
    }
    stage_1_pass = all(checks.values())
    report = {
        "contract_version": "s20-20r-r20-anchor-v1.4-20260907",
        "status": "stage_1_passed_freeze_stage_2"
        if stage_1_pass
        else "stage_1_failed_stop_before_lambdarank",
        "sample_bps": args.sample_bps,
        "sample_seed": COMPARABLE_SAMPLE_SEED,
        "seeds": list(seeds),
        "feature_count": len(features),
        "data_audit": audit,
        "splits": split_rows,
        "models": model_rows,
        "stage_1_checks": checks,
        "stage_1_pass": stage_1_pass,
        "probability_metrics": probability_rows,
        "daily_topk_metrics": topk_rows,
        "new_confirmation_opened": False,
    }
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    pd.DataFrame(probability_rows).to_csv(
        args.output_dir / "probability_metrics.csv", index=False
    )
    topk_frame.to_csv(args.output_dir / "daily_topk_metrics.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        primary[
            primary["fold"].isin(["wf1", "wf2", "wf3", "aggregate"])
        ][["fold", "candidate", "precision", "lift", "positive_pick_days_rate"]]
        .sort_values(["fold", "candidate"])
        .to_string(index=False),
        flush=True,
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
