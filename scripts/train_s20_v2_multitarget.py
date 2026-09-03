#!/usr/bin/env python
"""Train S20-v2 five-target probability and negative-reason models."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from explore_r20_target_prob_v2 import _load_dataset  # noqa: E402
from stockagent_analysis.r20_target_prob import (  # noqa: E402
    DEFAULT_FOLDS,
    probability_metrics,
)
from stockagent_analysis.s20 import (  # noqa: E402
    S20_V2_TARGET_DRAWDOWN,
    daily_topk_metrics,
    purged_walk_forward_masks,
)


TARGETS = tuple(int(value) for value in S20_V2_TARGET_DRAWDOWN)
DEFAULT_SEEDS = (20260903, 20260917, 20261001)
COMPARABLE_SAMPLE_SEED = 20260830
S20_V2_FOLDS = (
    replace(
        DEFAULT_FOLDS[0], tune_end="20241231", calibration_start="20250101"
    ),
    replace(
        DEFAULT_FOLDS[1], tune_end="20250430", calibration_start="20250501"
    ),
    replace(
        DEFAULT_FOLDS[2], tune_end="20250831", calibration_start="20250901"
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-bps", type=int, default=5000)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "output/experiments/s20_v2_labels/labels.parquet",
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
        default=ROOT / "output/experiments/s20_v2_multitarget",
    )
    return parser.parse_args()


def _prepare(args: argparse.Namespace):
    # Keep the row sample identical to the published B0 experiment.  Model seeds
    # remain independent so seed ensembling does not silently change the test set.
    data, features, source_audit = _load_dataset(
        args.sample_bps, COMPARABLE_SAMPLE_SEED
    )
    label_columns = ["ts_code", "trade_date", "horizon_end_date"]
    for target in TARGETS:
        label_columns.extend([f"positive{target}", f"class{target}", f"reason{target}"])
    labels = pd.read_parquet(args.labels, columns=label_columns)
    labels["ts_code"] = labels["ts_code"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    data = data.merge(
        labels,
        on=["ts_code", "trade_date"],
        how="inner",
        validate="one_to_one",
    )
    return data, features, {
        "source": source_audit,
        "rows": len(data),
        "symbols": int(data["ts_code"].nunique()),
        "date_min": data["trade_date"].min(),
        "date_max": data["trade_date"].max(),
    }


def _lgb_dataset(frame: pd.DataFrame, features: list[str], target: str):
    return lgb.Dataset(
        frame[features],
        label=frame[target],
        feature_name=features,
        free_raw_data=False,
    )


def _fit_binary(
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    features: list[str],
    target_column: str,
    seed: int,
    threads: int,
) -> lgb.Booster:
    params = {
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
    return lgb.train(
        params,
        _lgb_dataset(fit, features, target_column),
        num_boost_round=350,
        valid_sets=[_lgb_dataset(tune, features, target_column)],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )


def _fit_reason(
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    features: list[str],
    target_column: str,
    seed: int,
    threads: int,
) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed + 101,
        "num_threads": threads,
    }
    return lgb.train(
        params,
        _lgb_dataset(fit, features, target_column),
        num_boost_round=350,
        valid_sets=[_lgb_dataset(tune, features, target_column)],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)],
    )


def _fit_platt(raw_probability: np.ndarray, actual: np.ndarray):
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=DEFAULT_SEEDS[0])
    model.fit(logits, actual)
    return model


def _apply_platt(model: LogisticRegression, raw_probability: np.ndarray):
    clipped = np.clip(raw_probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def _probability_row(fold: str, target: int, candidate: str, actual, predicted):
    return {
        "fold": fold,
        "target_pct": target,
        "candidate": candidate,
        **probability_metrics(actual, predicted),
    }


def _topk_rows(fold_name: str, target: int, candidate: str, frame, column):
    rows = []
    for k in (10, 20, 50):
        rows.append(
            {
                "fold": fold_name,
                "target_pct": target,
                "candidate": candidate,
                **daily_topk_metrics(
                    frame,
                    probability_col=column,
                    target_col=f"positive{target}",
                    k=k,
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    data, features, audit = _prepare(args)
    b0_columns = ["ts_code", "trade_date", "fold"] + [
        f"ensemble_p{target}" for target in TARGETS if target <= 25
    ]
    b0 = pd.read_parquet(args.b0_predictions, columns=b0_columns)
    b0["ts_code"] = b0["ts_code"].astype(str)
    b0["trade_date"] = b0["trade_date"].astype(str)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probability_rows = []
    topk_rows = []
    reason_rows = []
    prediction_parts = []
    model_rows = []
    split_rows = []
    for fold in S20_V2_FOLDS:
        masks = purged_walk_forward_masks(
            data["trade_date"], data["horizon_end_date"], fold
        )
        split_rows.append(
            {"fold": fold.name, **{name: int(mask.sum()) for name, mask in masks.items()}}
        )
        fold_dir = args.output_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_predictions = data.loc[masks["test"], ["ts_code", "trade_date"]].copy()
        fold_predictions["fold"] = fold.name
        for target in TARGETS:
            binary_column = f"positive{target}"
            reason_column = f"class{target}"
            valid = data[binary_column] >= 0
            fit = data.loc[masks["fit"] & valid]
            tune = data.loc[masks["tune"] & valid]
            calibration = data.loc[masks["calibration"] & valid]
            test = data.loc[masks["test"] & valid]
            print(
                f"{fold.name} T{target}: fit={len(fit):,} tune={len(tune):,} "
                f"cal={len(calibration):,} test={len(test):,}",
                flush=True,
            )
            seed_models = []
            for seed in seeds:
                model = _fit_binary(
                    fit, tune, features, binary_column, seed, args.num_threads
                )
                model.save_model(
                    str(fold_dir / f"binary_t{target}_seed{seed}.txt")
                )
                seed_models.append(model)
                model_rows.append(
                    {
                        "fold": fold.name,
                        "target_pct": target,
                        "kind": "binary",
                        "seed": seed,
                        "best_iteration": model.best_iteration,
                    }
                )
            raw_cal = np.mean(
                [model.predict(calibration[features]) for model in seed_models], axis=0
            )
            raw_test = np.mean(
                [model.predict(test[features]) for model in seed_models], axis=0
            )
            platt = _fit_platt(raw_cal, calibration[binary_column].to_numpy())
            calibrated_test = _apply_platt(platt, raw_test)

            reason_model = _fit_reason(
                fit, tune, features, reason_column, seeds[0], args.num_threads
            )
            reason_model.save_model(str(fold_dir / f"reason_t{target}.txt"))
            reason_probability = reason_model.predict(test[features])
            reason_actual = test[reason_column].to_numpy()
            reason_rows.append(
                {
                    "fold": fold.name,
                    "target_pct": target,
                    "n": len(test),
                    "multi_log_loss": float(
                        log_loss(reason_actual, reason_probability, labels=[0, 1, 2, 3])
                    ),
                    "accuracy": float(
                        accuracy_score(reason_actual, reason_probability.argmax(axis=1))
                    ),
                    "macro_f1": float(
                        f1_score(
                            reason_actual,
                            reason_probability.argmax(axis=1),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
            model_rows.append(
                {
                    "fold": fold.name,
                    "target_pct": target,
                    "kind": "reason",
                    "seed": seeds[0],
                    "best_iteration": reason_model.best_iteration,
                }
            )

            target_prediction = test[
                ["ts_code", "trade_date", binary_column, reason_column]
            ].copy()
            raw_name = f"s20_{target}_raw"
            calibrated_name = f"s20_{target}_calibrated"
            target_prediction[raw_name] = raw_test
            target_prediction[calibrated_name] = calibrated_test
            for reason in range(4):
                target_prediction[f"s20_{target}_reason_p{reason}"] = reason_probability[
                    :, reason
                ]
            fold_predictions = fold_predictions.merge(
                target_prediction,
                on=["ts_code", "trade_date"],
                how="left",
                validate="one_to_one",
            )
            actual = test[binary_column].to_numpy()
            probability_rows.append(
                _probability_row(fold.name, target, "raw_seed_ensemble", actual, raw_test)
            )
            probability_rows.append(
                _probability_row(fold.name, target, "platt_seed_ensemble", actual, calibrated_test)
            )
            topk_rows.extend(
                _topk_rows(fold.name, target, "raw_seed_ensemble", target_prediction, raw_name)
            )
            topk_rows.extend(
                _topk_rows(
                    fold.name,
                    target,
                    "platt_seed_ensemble",
                    target_prediction,
                    calibrated_name,
                )
            )
            if target <= 25:
                b0_column = f"ensemble_p{target}"
                comparison = target_prediction.merge(
                    b0[b0["fold"] == fold.name][
                        ["ts_code", "trade_date", b0_column]
                    ],
                    on=["ts_code", "trade_date"],
                    how="inner",
                    validate="one_to_one",
                )
                probability_rows.append(
                    _probability_row(
                        fold.name,
                        target,
                        "b0_retargeted",
                        comparison[binary_column],
                        comparison[b0_column],
                    )
                )
                topk_rows.extend(
                    _topk_rows(
                        fold.name,
                        target,
                        "b0_retargeted",
                        comparison,
                        b0_column,
                    )
                )
        prediction_parts.append(fold_predictions)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    for target in TARGETS:
        valid = predictions[f"positive{target}"].notna()
        for candidate, column in [
            ("raw_seed_ensemble", f"s20_{target}_raw"),
            ("platt_seed_ensemble", f"s20_{target}_calibrated"),
        ]:
            subset = predictions.loc[valid]
            probability_rows.append(
                _probability_row(
                    "aggregate", target, candidate, subset[f"positive{target}"], subset[column]
                )
            )
            topk_rows.extend(_topk_rows("aggregate", target, candidate, subset, column))

    report = {
        "contract_version": "s20-v2-five-target-model-v1",
        "status": "offline_experiment",
        "sample_bps": args.sample_bps,
        "sample_seed": COMPARABLE_SAMPLE_SEED,
        "seeds": list(seeds),
        "feature_count": len(features),
        "data_audit": audit,
        "purged_splits": split_rows,
        "models": model_rows,
        "probability_metrics": probability_rows,
        "daily_topk_metrics": topk_rows,
        "reason_metrics": reason_rows,
    }
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    pd.DataFrame(probability_rows).to_csv(
        args.output_dir / "probability_metrics.csv", index=False
    )
    pd.DataFrame(topk_rows).to_csv(args.output_dir / "daily_topk_metrics.csv", index=False)
    pd.DataFrame(reason_rows).to_csv(args.output_dir / "reason_metrics.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aggregate = pd.DataFrame(topk_rows)
    print(
        aggregate[(aggregate["fold"] == "aggregate") & (aggregate["k"] == 20)][
            ["target_pct", "candidate", "precision", "lift", "positive_pick_days_rate"]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
