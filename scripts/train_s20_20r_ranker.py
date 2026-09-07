#!/usr/bin/env python
"""Train the frozen daily LambdaRank head for S20-20R stage 2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.s20 import daily_topk_metrics, purged_walk_forward_masks  # noqa: E402
from train_s20_20r_residual import (  # noqa: E402
    DEFAULT_SEEDS,
    NESTED_PLANS,
    PORTABLE_EXCLUDED_FEATURES,
    _fit_anchor,
    _prepare,
    _refit_anchor,
)
from train_s20_v2_multitarget import S20_V2_FOLDS  # noqa: E402


TOP_RANK_FEATURES = 32


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
        "--stage1-predictions",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_residual_v11/predictions.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_ranker",
    )
    return parser.parse_args()


def _add_rank_features(frame, selected):
    result = frame.copy()
    rank_columns = []
    grouped = result.groupby("trade_date", sort=False)
    for feature in selected:
        column = f"xs_rank__{feature}"
        result[column] = grouped[feature].rank(pct=True, method="average").astype(
            "float32"
        )
        rank_columns.append(column)
    return result, rank_columns


def _ranking_frame(frame, features, selected):
    result, rank_columns = _add_rank_features(frame, selected)
    result["r20_anchor_feature"] = result["r20_anchor"].astype("float32")
    result["relevance20"] = np.select(
        [result["class20"] == 0, result["class20"] == 1],
        [3, 1],
        default=0,
    ).astype("int8")
    result = result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    groups = result.groupby("trade_date", sort=False).size().to_numpy()
    return result, [*features, "r20_anchor_feature", *rank_columns], groups


def _fit_ranker(fit, tune, rank_features, fit_groups, tune_groups, seed, threads):
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20, 50],
        "label_gain": [0, 1, 2, 3],
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 250,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": seed,
        "num_threads": threads,
    }
    train_set = lgb.Dataset(
        fit[rank_features],
        label=fit["relevance20"],
        group=fit_groups,
        feature_name=rank_features,
        free_raw_data=False,
    )
    tune_set = lgb.Dataset(
        tune[rank_features],
        label=tune["relevance20"],
        group=tune_groups,
        feature_name=rank_features,
        free_raw_data=False,
    )
    return lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[tune_set],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )


def _daily_percentile(frame, column):
    return frame.groupby("trade_date", sort=False)[column].rank(
        pct=True, method="average"
    )


def _topk_rows(frame, fold, candidate, column):
    return [
        {
            "fold": fold,
            "candidate": candidate,
            **daily_topk_metrics(
                frame, probability_col=column, target_col="positive20", k=k
            ),
        }
        for k in (10, 20, 50)
    ]


def main() -> int:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    data, features, audit = _prepare(args)
    if args.feature_mode == "portable":
        features = [f for f in features if f not in PORTABLE_EXCLUDED_FEATURES]
    audit["feature_mode"] = args.feature_mode
    audit["model_feature_count"] = len(features)
    stage1 = pd.read_parquet(args.stage1_predictions)
    stage1["ts_code"] = stage1["ts_code"].astype(str)
    stage1["trade_date"] = stage1["trade_date"].astype(str)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topk_rows = []
    prediction_parts = []
    fold_rows = []
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
        outer_fit = data.loc[outer["fit"]]
        rank_fit = data.loc[residual_fit_mask & valid].copy()
        rank_tune = data.loc[outer["tune"] & valid].copy()
        test = data.loc[outer["test"] & valid].copy()

        early_anchor = _fit_anchor(
            anchor_fit, anchor_tune, features, seeds[0] - 503, args.num_threads
        )
        selected = (
            pd.DataFrame(
                {
                    "feature": features,
                    "gain": early_anchor.feature_importance(importance_type="gain"),
                }
            )
            .sort_values(["gain", "feature"], ascending=[False, True])
            .head(TOP_RANK_FEATURES)["feature"]
            .tolist()
        )
        rank_fit["r20_anchor"] = early_anchor.predict(
            rank_fit[features], num_iteration=early_anchor.best_iteration
        )
        refit_anchor = _refit_anchor(
            outer_fit,
            features,
            seeds[0] - 503,
            args.num_threads,
            early_anchor.best_iteration,
        )
        for frame in (rank_tune, test):
            frame["r20_anchor"] = refit_anchor.predict(frame[features])

        fit_rank, rank_features, fit_groups = _ranking_frame(
            rank_fit, features, selected
        )
        tune_rank, tune_features, tune_groups = _ranking_frame(
            rank_tune, features, selected
        )
        test_rank, test_features, _ = _ranking_frame(test, features, selected)
        if rank_features != tune_features or rank_features != test_features:
            raise AssertionError("ranking feature schema changed between segments")

        fold_dir = args.output_dir / fold.name
        fold_dir.mkdir(parents=True, exist_ok=True)
        models = []
        for seed in seeds:
            model = _fit_ranker(
                fit_rank,
                tune_rank,
                rank_features,
                fit_groups,
                tune_groups,
                seed,
                args.num_threads,
            )
            model.save_model(str(fold_dir / f"ranker_seed{seed}.txt"))
            models.append(model)
            fold_rows.append(
                {
                    "fold": fold.name,
                    "seed": seed,
                    "best_iteration": model.best_iteration,
                    "selected_rank_features": selected,
                }
            )
        test_rank["lambdarank_score"] = np.mean(
            [
                model.predict(
                    test_rank[rank_features], num_iteration=model.best_iteration
                )
                for model in models
            ],
            axis=0,
        )
        compare = stage1[stage1["fold"] == fold.name][
            [
                "ts_code",
                "trade_date",
                "s20_20r_raw",
                "ensemble_p20",
                "s20_20_raw",
            ]
        ]
        result = test_rank[
            ["ts_code", "trade_date", "positive20", "lambdarank_score"]
        ].merge(compare, on=["ts_code", "trade_date"], validate="one_to_one")
        result["fold"] = fold.name
        result["hybrid_rank"] = 0.5 * _daily_percentile(
            result, "s20_20r_raw"
        ) + 0.5 * _daily_percentile(result, "lambdarank_score")
        candidates = {
            "lambdarank": "lambdarank_score",
            "hybrid_rank_fixed_50_50": "hybrid_rank",
            "stage1_residual": "s20_20r_raw",
            "r20_b0_published": "ensemble_p20",
            "s20_v2": "s20_20_raw",
        }
        for candidate, column in candidates.items():
            topk_rows.extend(_topk_rows(result, fold.name, candidate, column))
        prediction_parts.append(result)
        print(
            fold.name,
            {"fit": len(fit_rank), "tune": len(tune_rank), "test": len(result)},
            flush=True,
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    candidates = {
        "lambdarank": "lambdarank_score",
        "hybrid_rank_fixed_50_50": "hybrid_rank",
        "stage1_residual": "s20_20r_raw",
        "r20_b0_published": "ensemble_p20",
        "s20_v2": "s20_20_raw",
    }
    for candidate, column in candidates.items():
        topk_rows.extend(_topk_rows(predictions, "aggregate", candidate, column))
    metrics = pd.DataFrame(topk_rows)
    primary = metrics[metrics["k"] == 20]
    pivot = primary.pivot(index="fold", columns="candidate", values="precision")
    fold_delta = (
        pivot["hybrid_rank_fixed_50_50"] - pivot["r20_b0_published"]
    ).drop(index="aggregate")
    checks = {
        "aggregate_top20_precision_gt_stage_1": bool(
            pivot.loc["aggregate", "hybrid_rank_fixed_50_50"]
            > pivot.loc["aggregate", "stage1_residual"]
        ),
        "aggregate_top20_precision_gt_s20_v2": bool(
            pivot.loc["aggregate", "hybrid_rank_fixed_50_50"]
            > pivot.loc["aggregate", "s20_v2"]
        ),
        "each_fold_precision_regression_vs_r20_b0_lte_1pp": bool(
            (fold_delta >= -0.01).all()
        ),
    }
    report = {
        "contract_version": "s20-20r-r20-anchor-v1.4-20260907",
        "status": "stage_2_passed_freeze_calibration"
        if all(checks.values())
        else "stage_2_failed_keep_stage_1",
        "sample_bps": args.sample_bps,
        "feature_count": len(features),
        "data_audit": audit,
        "models": fold_rows,
        "stage_2_checks": checks,
        "stage_2_pass": all(checks.values()),
        "daily_topk_metrics": topk_rows,
        "new_confirmation_opened": False,
    }
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    metrics.to_csv(args.output_dir / "daily_topk_metrics.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        primary[["fold", "candidate", "precision", "lift"]]
        .sort_values(["fold", "candidate"])
        .to_string(index=False),
        flush=True,
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
