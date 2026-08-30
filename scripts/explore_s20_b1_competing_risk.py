#!/usr/bin/env python
"""Run S20-B1 piecewise discrete-time competing-risk walk-forward tests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explore_r20_target_prob_v2 import _load_dataset  # noqa: E402
from stockagent_analysis.r20_target_prob import (  # noqa: E402
    DEFAULT_FOLDS,
    probability_metrics,
)
from stockagent_analysis.s20 import cumulative_incidence  # noqa: E402


INTERVAL_ENDS = (1, 3, 5, 10, 15, 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-bps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT
        / "output/experiments/s20_first_passage_v1/first_passage_labels.parquet",
    )
    parser.add_argument(
        "--b0-predictions",
        type=Path,
        default=ROOT
        / "output/experiments/r20_target_prob_v2/walk_forward_predictions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_b1_competing_risk",
    )
    return parser.parse_args()


def _prepare(sample_bps: int, seed: int, label_path: Path):
    data, features, source_audit = _load_dataset(sample_bps, seed)
    labels = pd.read_parquet(
        label_path,
        columns=[
            "ts_code",
            "trade_date",
            "down_day",
            "up25_day",
            "event25",
            "target25_up_first",
            "target25_window_safe",
        ],
    )
    labels["ts_code"] = labels["ts_code"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    overlap = data.merge(
        labels,
        on=["ts_code", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    before = len(overlap)
    overlap = overlap[overlap["event25"] != "ambiguous"].copy()
    overlap["first_event_day"] = np.where(
        overlap["event25"] == "up_first",
        overlap["up25_day"],
        np.where(overlap["event25"] == "down_first", overlap["down_day"], 0),
    ).astype(np.int16)
    audit = {
        "source": source_audit,
        "rows_joined": before,
        "rows_after_ambiguous_exclusion": len(overlap),
        "ambiguous_excluded": before - len(overlap),
        "event_rates": overlap["event25"].value_counts(normalize=True).to_dict(),
    }
    return overlap, features, audit


def _interval_frame(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    risk = (frame["first_event_day"] == 0) | (frame["first_event_day"] > start)
    result = frame.loc[risk].copy()
    result["interval_target"] = 0
    up = (
        (result["event25"] == "up_first")
        & result["up25_day"].between(start + 1, end)
    )
    down = (
        (result["event25"] == "down_first")
        & result["down_day"].between(start + 1, end)
    )
    result.loc[up, "interval_target"] = 1
    result.loc[down, "interval_target"] = 2
    return result


def _fit_interval_model(
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    features: list[str],
    start: int,
    end: int,
    seed: int,
    threads: int,
) -> lgb.Booster:
    train = _interval_frame(fit, start, end)
    valid = _interval_frame(tune, start, end)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed + end,
        "num_threads": threads,
    }
    datasets = []
    for frame in (train, valid):
        datasets.append(
            lgb.Dataset(
                frame[features],
                label=frame["interval_target"],
                feature_name=features,
                free_raw_data=False,
            )
        )
    return lgb.train(
        params,
        datasets[0],
        num_boost_round=250,
        valid_sets=[datasets[1]],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )


def _predict_cif(models: list[lgb.Booster], frame: pd.DataFrame, features: list[str]):
    interval_probabilities = [model.predict(frame[features]) for model in models]
    return cumulative_incidence(interval_probabilities)


def _fit_platt(raw_probability: np.ndarray, actual: np.ndarray):
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    score = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=20260830)
    model.fit(score, actual)
    return model


def _apply_platt(model: LogisticRegression, raw_probability: np.ndarray):
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    score = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(score)[:, 1]


def _metric_row(fold: str, candidate: str, actual, predicted) -> dict:
    return {"fold": fold, "candidate": candidate, **probability_metrics(actual, predicted)}


def main() -> int:
    args = parse_args()
    data, features, audit = _prepare(args.sample_bps, args.seed, args.labels)
    b0 = pd.read_parquet(
        args.b0_predictions,
        columns=["ts_code", "trade_date", "fold", "ensemble_p25"],
    ).rename(columns={"ensemble_p25": "b0_probability"})
    b0["ts_code"] = b0["ts_code"].astype(str)
    b0["trade_date"] = b0["trade_date"].astype(str)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    predictions = []
    model_summary = []
    for fold in DEFAULT_FOLDS:
        masks = fold.masks(data["trade_date"])
        fit = data.loc[masks["fit"]]
        tune = data.loc[masks["tune"]]
        calibration = data.loc[masks["calibration"]]
        test = data.loc[masks["test"]]
        print(
            f"{fold.name}: fit={len(fit):,} tune={len(tune):,} "
            f"cal={len(calibration):,} test={len(test):,}",
            flush=True,
        )
        models = []
        start = 0
        fold_dir = args.output_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        for end in INTERVAL_ENDS:
            model = _fit_interval_model(
                fit, tune, features, start, end, args.seed, args.num_threads
            )
            model.save_model(str(fold_dir / f"hazard_{start + 1:02d}_{end:02d}.txt"))
            models.append(model)
            model_summary.append(
                {
                    "fold": fold.name,
                    "start": start + 1,
                    "end": end,
                    "best_iteration": model.best_iteration,
                }
            )
            print(
                f"  interval {start + 1:02d}-{end:02d}: "
                f"best_iteration={model.best_iteration}",
                flush=True,
            )
            start = end

        calibration_cif = _predict_cif(models, calibration, features)
        test_cif = _predict_cif(models, test, features)
        platt = _fit_platt(
            calibration_cif["upside"],
            calibration["target25_up_first"].to_numpy(),
        )
        calibrated = _apply_platt(platt, test_cif["upside"])
        actual = test["target25_up_first"].to_numpy()
        metrics.append(_metric_row(fold.name, "b1_raw_cif", actual, test_cif["upside"]))
        metrics.append(_metric_row(fold.name, "b1_platt_cif", actual, calibrated))

        result = test[
            ["ts_code", "trade_date", "event25", "target25_up_first"]
        ].copy()
        result["fold"] = fold.name
        result["b1_raw_upside"] = test_cif["upside"]
        result["b1_raw_downside"] = test_cif["downside"]
        result["b1_raw_survival"] = test_cif["survival"]
        result["s20_probability"] = calibrated
        result = result.merge(
            b0[b0["fold"] == fold.name],
            on=["ts_code", "trade_date", "fold"],
            how="left",
            validate="one_to_one",
        )
        has_b0 = result["b0_probability"].notna()
        metrics.append(
            _metric_row(
                fold.name,
                "b0_ensemble_retargeted",
                result.loc[has_b0, "target25_up_first"],
                result.loc[has_b0, "b0_probability"],
            )
        )
        result["b1_raw_b0_50"] = (
            result["b1_raw_upside"] + result["b0_probability"]
        ) / 2
        result["b1_platt_b0_50"] = (
            result["s20_probability"] + result["b0_probability"]
        ) / 2
        for candidate, column in [
            ("b1_raw_b0_50_exploratory", "b1_raw_b0_50"),
            ("b1_platt_b0_50_exploratory", "b1_platt_b0_50"),
        ]:
            metrics.append(
                _metric_row(
                    fold.name,
                    candidate,
                    result.loc[has_b0, "target25_up_first"],
                    result.loc[has_b0, column],
                )
            )
        predictions.append(result)

    predictions_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics)
    aggregate = []
    for candidate, column in [
        ("b1_raw_cif", "b1_raw_upside"),
        ("b1_platt_cif", "s20_probability"),
        ("b0_ensemble_retargeted", "b0_probability"),
        ("b1_raw_b0_50_exploratory", "b1_raw_b0_50"),
        ("b1_platt_b0_50_exploratory", "b1_platt_b0_50"),
    ]:
        valid = predictions_frame[column].notna()
        aggregate.append(
            _metric_row(
                "aggregate",
                candidate,
                predictions_frame.loc[valid, "target25_up_first"],
                predictions_frame.loc[valid, column],
            )
        )
    metrics_frame = pd.concat([metrics_frame, pd.DataFrame(aggregate)], ignore_index=True)
    report = {
        "contract_version": "s20-b1-competing-risk-v1",
        "status": "offline_experiment",
        "selection_warning": "fixed blends are post-hoc diagnostics and cannot be promoted from this run",
        "sample_bps": args.sample_bps,
        "interval_ends": list(INTERVAL_ENDS),
        "feature_count": len(features),
        "data_audit": audit,
        "model_summary": model_summary,
        "metrics": metrics_frame.to_dict(orient="records"),
    }
    predictions_frame.to_parquet(
        args.output_dir / "walk_forward_predictions.parquet", index=False
    )
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics_frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
