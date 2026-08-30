#!/usr/bin/env python
"""Run the isolated R20-P v2 walk-forward probability experiment.

The script writes only beneath ``output/experiments/r20_target_prob_v2`` by
default.  It does not import or modify the Pool-A builder, production aliases,
web payload, or the daily scheduler.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.r20_target_prob import (  # noqa: E402
    DEFAULT_FOLDS,
    GAIN_THRESHOLDS,
    build_target_frame,
    ordered_probabilities,
    probability_metrics,
    tier_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-bps",
        type=int,
        default=2000,
        help="Deterministic per-row sample in basis points; 10000 uses all rows.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "experiments" / "r20_target_prob_v2",
    )
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _load_dataset(sample_bps: int, seed: int) -> tuple[pd.DataFrame, list[str], dict]:
    if not 1 <= sample_bps <= 10000:
        raise ValueError("sample-bps must be in 1..10000")
    model_meta = json.loads(
        (ROOT / "output/lgbm_maxgain/feature_meta.json").read_text(encoding="utf-8")
    )
    feature_cols = list(model_meta["feature_cols"])
    regime_features = [
        "regime_id",
        "mkt_ret_5d",
        "mkt_ret_20d",
        "mkt_ret_60d",
        "mkt_rsi14",
        "mkt_vol_ratio",
        "regime_days_in",
        "regime_intensity",
        "hs300_ret60_z60",
        "cyb_rel_strength",
        "zz500_rel_strength",
    ]
    feature_cols.extend(column for column in regime_features if column not in feature_cols)
    raw_features = [column for column in feature_cols if column != "industry_id"]

    labels = build_target_frame(
        pd.read_parquet(ROOT / "output/labels/max_gain_labels.parquet")
    )
    labels = labels[
        labels["ts_code"].str.endswith((".SH", ".SZ"))
    ].copy()
    label_cols = [
        "ts_code",
        "trade_date",
        "r20_close",
        "max_gain_20",
        "max_dd_20",
        "target_p15_safe",
        "target_p20_safe",
        "target_p25_safe",
        "target_close25_safe",
        "target_tier",
    ]
    labels = labels[label_cols].set_index(["ts_code", "trade_date"]).sort_index()

    parts: list[pd.DataFrame] = []
    group_paths = sorted((ROOT / "output/factor_lab_3y/factor_groups").glob("group_*.parquet"))
    if not group_paths:
        raise FileNotFoundError("factor-group parquet files are missing")
    for number, path in enumerate(group_paths, 1):
        available = set(pq.read_schema(path).names)
        columns = [
            column
            for column in ["ts_code", "trade_date", "industry", *raw_features]
            if column in available
        ]
        factors = pd.read_parquet(path, columns=columns)
        factors["ts_code"] = factors["ts_code"].astype(str)
        factors["trade_date"] = factors["trade_date"].astype(str)
        factors = factors[
            factors["ts_code"].str.endswith((".SH", ".SZ"))
            & factors["trade_date"].between("20240101", "20260126")
        ]
        joined = factors.join(labels, on=["ts_code", "trade_date"], how="inner")
        if sample_bps < 10000 and not joined.empty:
            hashes = pd.util.hash_pandas_object(
                joined[["ts_code", "trade_date"]], index=False, hash_key="0123456789abcdef"
            ).to_numpy(dtype="uint64")
            joined = joined[(hashes + np.uint64(seed)) % 10000 < sample_bps]
        if not joined.empty:
            parts.append(joined)
        print(f"loaded {number:02d}/{len(group_paths)} {path.name}: {len(joined):,}", flush=True)

    data = pd.concat(parts, ignore_index=True)
    before_st = len(data)
    basic_path = ROOT / "output/tushare_cache/stock_basic.parquet"
    if basic_path.exists():
        basic = pd.read_parquet(basic_path, columns=["ts_code", "name"])
        st_codes = set(
            basic.loc[
                basic["name"].fillna("").str.contains("ST", regex=False), "ts_code"
            ].astype(str)
        )
        data = data[~data["ts_code"].isin(st_codes)].copy()

    regime = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet")
    regime["trade_date"] = regime["trade_date"].astype(str)
    regime = regime.rename(
        columns={
            "ret_5d": "mkt_ret_5d",
            "ret_20d": "mkt_ret_20d",
            "ret_60d": "mkt_ret_60d",
            "rsi14": "mkt_rsi14",
            "vol_ratio": "mkt_vol_ratio",
        }
    )
    regime_extra = pd.read_parquet(ROOT / "output/regime_extra/regime_extra.parquet")
    regime_extra["trade_date"] = regime_extra["trade_date"].astype(str)
    regime = regime.merge(regime_extra, on="trade_date", how="left")
    available_regime = ["trade_date", *[c for c in regime_features if c in regime]]
    data = data.merge(regime[available_regime], on="trade_date", how="left")
    industry_map = model_meta.get("industry_map", {})
    data["industry_id"] = data["industry"].fillna("unknown").astype(str).map(
        lambda value: industry_map.get(value, -1)
    )
    for column in feature_cols:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    audit = {
        "sample_bps": sample_bps,
        "rows": len(data),
        "symbols": int(data["ts_code"].nunique()),
        "date_min": data["trade_date"].min(),
        "date_max": data["trade_date"].max(),
        "duplicate_keys": int(data.duplicated(["ts_code", "trade_date"]).sum()),
        "current_st_rows_excluded": int(before_st - len(data)),
        "feature_count": len(feature_cols),
        "event_rates": {
            f"p{threshold}_safe": float(data[f"target_p{threshold}_safe"].mean())
            for threshold in GAIN_THRESHOLDS
        },
        "close25_safe_rate": float(data["target_close25_safe"].mean()),
        "tier_counts": {
            str(key): int(value)
            for key, value in data["target_tier"].value_counts().sort_index().items()
        },
    }
    return data, feature_cols, audit


def _dataset(frame: pd.DataFrame, features: list[str], label: str) -> lgb.Dataset:
    return lgb.Dataset(
        frame[features],
        label=frame[label],
        feature_name=features,
        free_raw_data=False,
    )


def _fit_ordinal(
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    features: list[str],
    seed: int,
    threads: int,
) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
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
        _dataset(fit, features, "target_tier"),
        num_boost_round=500,
        valid_sets=[_dataset(tune, features, "target_tier")],
        callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(0)],
    )


def _fit_r20_regressor(
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    features: list[str],
    seed: int,
    threads: int,
) -> lgb.Booster:
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed + 17,
        "num_threads": threads,
    }
    return lgb.train(
        params,
        _dataset(fit, features, "r20_close"),
        num_boost_round=500,
        valid_sets=[_dataset(tune, features, "r20_close")],
        callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(0)],
    )


def _fit_binary_models(
    fit: pd.DataFrame,
    tune: pd.DataFrame,
    features: list[str],
    seed: int,
    threads: int,
) -> list[lgb.Booster]:
    models = []
    for offset, threshold in enumerate(GAIN_THRESHOLDS):
        target = f"target_p{threshold}_safe"
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
            "seed": seed + offset * 31,
            "num_threads": threads,
        }
        models.append(
            lgb.train(
                params,
                _dataset(fit, features, target),
                num_boost_round=500,
                valid_sets=[_dataset(tune, features, target)],
                callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(0)],
            )
        )
    return models


def _predict_binary_models(
    models: list[lgb.Booster], frame: pd.DataFrame, features: list[str]
) -> np.ndarray:
    values = [
        model.predict(frame[features], num_iteration=model.best_iteration)
        for model in models
    ]
    return ordered_probabilities(np.column_stack(values))


def _fit_calibrators(
    scores: np.ndarray, calibration: pd.DataFrame
) -> list[IsotonicRegression]:
    calibrators = []
    for index, threshold in enumerate(GAIN_THRESHOLDS):
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        source = scores[:, index] if scores.ndim == 2 else scores
        model.fit(source, calibration[f"target_p{threshold}_safe"].to_numpy())
        calibrators.append(model)
    return calibrators


def _apply_calibrators(
    calibrators: list[IsotonicRegression], scores: np.ndarray
) -> np.ndarray:
    calibrated = []
    for index, model in enumerate(calibrators):
        source = scores[:, index] if scores.ndim == 2 else scores
        calibrated.append(model.predict(source))
    return ordered_probabilities(np.column_stack(calibrated))


def _calibrator_payload(models: list[IsotonicRegression]) -> dict:
    return {
        f"p{threshold}_safe": {
            "x": model.X_thresholds_.tolist(),
            "y": model.y_thresholds_.tolist(),
        }
        for threshold, model in zip(GAIN_THRESHOLDS, models)
    }


def _score_column(scores: np.ndarray, index: int, use_logit: bool) -> np.ndarray:
    source = scores[:, index] if scores.ndim == 2 else scores
    source = np.asarray(source, dtype=float)
    if use_logit:
        clipped = np.clip(source, 1e-6, 1 - 1e-6)
        source = np.log(clipped / (1.0 - clipped))
    return source.reshape(-1, 1)


def _fit_platt_calibrators(
    scores: np.ndarray, calibration: pd.DataFrame, use_logit: bool
) -> list[LogisticRegression]:
    models = []
    for index, threshold in enumerate(GAIN_THRESHOLDS):
        model = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
        model.fit(
            _score_column(scores, index, use_logit),
            calibration[f"target_p{threshold}_safe"].to_numpy(),
        )
        models.append(model)
    return models


def _apply_platt_calibrators(
    models: list[LogisticRegression], scores: np.ndarray, use_logit: bool
) -> np.ndarray:
    values = [
        model.predict_proba(_score_column(scores, index, use_logit))[:, 1]
        for index, model in enumerate(models)
    ]
    return ordered_probabilities(np.column_stack(values))


def _platt_payload(models: list[LogisticRegression]) -> dict:
    return {
        f"p{threshold}_safe": {
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
        }
        for threshold, model in zip(GAIN_THRESHOLDS, models)
    }


def _candidate_metrics(
    frame: pd.DataFrame, probabilities: np.ndarray, candidate: str, fold: str
) -> list[dict]:
    rows = []
    for index, threshold in enumerate(GAIN_THRESHOLDS):
        rows.append(
            {
                "fold": fold,
                "candidate": candidate,
                "target": f"p{threshold}_safe",
                **probability_metrics(
                    frame[f"target_p{threshold}_safe"], probabilities[:, index]
                ),
            }
        )
    return rows


def _subgroup_metrics(predictions: pd.DataFrame, minimum_n: int = 500) -> list[dict]:
    rows = []
    work = predictions.copy()
    work["exchange"] = work["ts_code"].str[-2:]
    if "total_mv" in work:
        work["market_cap_bucket"] = pd.qcut(
            pd.to_numeric(work["total_mv"], errors="coerce"),
            5,
            labels=["mv1", "mv2", "mv3", "mv4", "mv5"],
            duplicates="drop",
        ).astype(str)
    dimensions = ["fold", "exchange", "market_cap_bucket", "regime_id"]
    for dimension in dimensions:
        if dimension not in work:
            continue
        for value, group in work.groupby(dimension, dropna=False):
            if len(group) < minimum_n:
                continue
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    **probability_metrics(group["target_p25_safe"], group["p25_safe"]),
                }
            )
    top_industries = work["industry"].value_counts().head(20).index
    for industry, group in work[work["industry"].isin(top_industries)].groupby("industry"):
        if len(group) >= minimum_n:
            rows.append(
                {
                    "dimension": "industry",
                    "value": str(industry),
                    **probability_metrics(group["target_p25_safe"], group["p25_safe"]),
                }
            )
    return rows


def _decile_table(actual: pd.Series, predicted: pd.Series) -> list[dict]:
    frame = pd.DataFrame({"actual": actual.astype(int), "predicted": predicted.astype(float)})
    frame["decile"] = pd.qcut(
        frame["predicted"].rank(method="first"), 10, labels=False
    )
    table = frame.groupby("decile").agg(
        n=("actual", "size"),
        mean_probability=("predicted", "mean"),
        event_rate=("actual", "mean"),
    )
    return [
        {"decile": int(index), **row.to_dict()}
        for index, row in table.iterrows()
    ]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, features, audit = _load_dataset(args.sample_bps, args.seed)
    if audit["duplicate_keys"]:
        raise ValueError("factor/label join produced duplicate stock-date keys")
    (args.output_dir / "label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=_jsonable),
        encoding="utf-8",
    )

    metrics: list[dict] = []
    prediction_parts: list[pd.DataFrame] = []
    fold_report = {}
    for fold_no, fold in enumerate(DEFAULT_FOLDS, 1):
        masks = fold.masks(data["trade_date"])
        fit = data.loc[masks["fit"]]
        tune = data.loc[masks["tune"]]
        calibration = data.loc[masks["calibration"]]
        test = data.loc[masks["test"]]
        sizes = {name: int(mask.sum()) for name, mask in masks.items()}
        if min(sizes.values()) == 0:
            raise ValueError(f"{fold.name} has an empty chronological segment: {sizes}")
        print(f"{fold.name} sizes={sizes}: training ordinal model", flush=True)

        ordinal = _fit_ordinal(fit, tune, features, args.seed + fold_no, args.num_threads)
        ordinal_cal_scores = tier_probabilities(
            ordinal.predict(calibration[features], num_iteration=ordinal.best_iteration)
        )
        ordinal_isotonic = _fit_calibrators(ordinal_cal_scores, calibration)
        ordinal_platt = _fit_platt_calibrators(
            ordinal_cal_scores, calibration, use_logit=True
        )
        ordinal_raw = tier_probabilities(
            ordinal.predict(test[features], num_iteration=ordinal.best_iteration)
        )
        ordinal_isotonic_prob = _apply_calibrators(ordinal_isotonic, ordinal_raw)
        ordinal_platt_prob = _apply_platt_calibrators(
            ordinal_platt, ordinal_raw, use_logit=True
        )
        metrics.extend(_candidate_metrics(test, ordinal_raw, "ordinal_raw", fold.name))
        metrics.extend(
            _candidate_metrics(
                test, ordinal_isotonic_prob, "ordinal_isotonic", fold.name
            )
        )
        metrics.extend(
            _candidate_metrics(test, ordinal_platt_prob, "ordinal_platt", fold.name)
        )

        print(f"{fold.name}: training direct P15/P20/P25 binary models", flush=True)
        binary_models = _fit_binary_models(
            fit, tune, features, args.seed + 1000 + fold_no, args.num_threads
        )
        binary_cal_scores = _predict_binary_models(binary_models, calibration, features)
        binary_isotonic = _fit_calibrators(binary_cal_scores, calibration)
        binary_platt = _fit_platt_calibrators(
            binary_cal_scores, calibration, use_logit=True
        )
        binary_raw = _predict_binary_models(binary_models, test, features)
        binary_isotonic_prob = _apply_calibrators(binary_isotonic, binary_raw)
        binary_platt_prob = _apply_platt_calibrators(
            binary_platt, binary_raw, use_logit=True
        )
        metrics.extend(_candidate_metrics(test, binary_raw, "binary_raw", fold.name))
        metrics.extend(
            _candidate_metrics(
                test, binary_isotonic_prob, "binary_isotonic", fold.name
            )
        )
        metrics.extend(
            _candidate_metrics(test, binary_platt_prob, "binary_platt", fold.name)
        )
        ensemble_prob = ordered_probabilities(
            0.5 * ordinal_isotonic_prob + 0.5 * binary_platt_prob
        )
        metrics.extend(
            _candidate_metrics(test, ensemble_prob, "ensemble_50_50", fold.name)
        )

        print(f"{fold.name}: training legacy-style R20 regression baseline", flush=True)
        r20 = _fit_r20_regressor(fit, tune, features, args.seed + fold_no, args.num_threads)
        r20_cal_score = r20.predict(calibration[features], num_iteration=r20.best_iteration)
        r20_isotonic = _fit_calibrators(r20_cal_score, calibration)
        r20_platt = _fit_platt_calibrators(
            r20_cal_score, calibration, use_logit=False
        )
        r20_test_score = r20.predict(test[features], num_iteration=r20.best_iteration)
        r20_isotonic_prob = _apply_calibrators(r20_isotonic, r20_test_score)
        r20_platt_prob = _apply_platt_calibrators(
            r20_platt, r20_test_score, use_logit=False
        )
        metrics.extend(
            _candidate_metrics(
                test, r20_isotonic_prob, "r20_regression_isotonic", fold.name
            )
        )
        metrics.extend(
            _candidate_metrics(test, r20_platt_prob, "r20_regression_platt", fold.name)
        )

        close_calibrator = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
        close_calibrator.fit(
            np.asarray(r20_cal_score).reshape(-1, 1),
            calibration["target_close25_safe"],
        )
        close_probability = close_calibrator.predict_proba(
            np.asarray(r20_test_score).reshape(-1, 1)
        )[:, 1]

        fold_dir = args.output_dir / fold.name
        fold_dir.mkdir(exist_ok=True)
        ordinal.save_model(str(fold_dir / "ordinal_multiclass.txt"))
        for threshold, model in zip(GAIN_THRESHOLDS, binary_models):
            model.save_model(str(fold_dir / f"binary_p{threshold}_safe.txt"))
        r20.save_model(str(fold_dir / "r20_regressor.txt"))
        calibrator_payload = {
            "ordinal_isotonic": _calibrator_payload(ordinal_isotonic),
            "ordinal_platt": _platt_payload(ordinal_platt),
            "binary_isotonic": _calibrator_payload(binary_isotonic),
            "binary_platt": _platt_payload(binary_platt),
            "r20_regression_isotonic": _calibrator_payload(r20_isotonic),
            "r20_regression_platt": _platt_payload(r20_platt),
            "close25_safe": {
                "coefficient": float(close_calibrator.coef_[0, 0]),
                "intercept": float(close_calibrator.intercept_[0]),
            },
        }
        (fold_dir / "calibrators.json").write_text(
            json.dumps(calibrator_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        importance = pd.DataFrame(
            {
                "feature": ordinal.feature_name(),
                "gain": ordinal.feature_importance(importance_type="gain"),
                "split": ordinal.feature_importance(importance_type="split"),
            }
        ).sort_values("gain", ascending=False)
        importance.to_csv(fold_dir / "feature_importance.csv", index=False)

        columns = [
            "ts_code",
            "trade_date",
            "industry",
            "total_mv",
            "regime_id",
            "target_p15_safe",
            "target_p20_safe",
            "target_p25_safe",
            "target_close25_safe",
        ]
        result = test[[column for column in columns if column in test]].copy()
        result["fold"] = fold.name
        result[["raw_p15_safe", "raw_p20_safe", "raw_p25_safe"]] = ordinal_raw
        result[["iso_p15_safe", "iso_p20_safe", "iso_p25_safe"]] = ordinal_isotonic_prob
        result[["p15_safe", "p20_safe", "p25_safe"]] = ordinal_platt_prob
        result[["binary_raw_p15", "binary_raw_p20", "binary_raw_p25"]] = binary_raw
        result[["binary_iso_p15", "binary_iso_p20", "binary_iso_p25"]] = binary_isotonic_prob
        result[["binary_platt_p15", "binary_platt_p20", "binary_platt_p25"]] = binary_platt_prob
        result[["ensemble_p15", "ensemble_p20", "ensemble_p25"]] = ensemble_prob
        result[["r20_iso_p15", "r20_iso_p20", "r20_iso_p25"]] = r20_isotonic_prob
        result[["r20_platt_p15", "r20_platt_p20", "r20_platt_p25"]] = r20_platt_prob
        result["p_close25_safe"] = close_probability
        result["r20_pred_walkforward"] = r20_test_score
        prediction_parts.append(result)
        fold_report[fold.name] = {
            "sizes": sizes,
            "ordinal_best_iteration": ordinal.best_iteration,
            "binary_best_iterations": {
                str(threshold): model.best_iteration
                for threshold, model in zip(GAIN_THRESHOLDS, binary_models)
            },
            "r20_best_iteration": r20.best_iteration,
            "date_boundaries": fold.__dict__,
        }

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(args.output_dir / "walk_forward_metrics.csv", index=False)

    aggregate = []
    persisted_candidates = {
        "ordinal_raw": ["raw_p15_safe", "raw_p20_safe", "raw_p25_safe"],
        "ordinal_isotonic": ["iso_p15_safe", "iso_p20_safe", "iso_p25_safe"],
        "ordinal_platt": ["p15_safe", "p20_safe", "p25_safe"],
        "binary_raw": ["binary_raw_p15", "binary_raw_p20", "binary_raw_p25"],
        "binary_isotonic": ["binary_iso_p15", "binary_iso_p20", "binary_iso_p25"],
        "binary_platt": ["binary_platt_p15", "binary_platt_p20", "binary_platt_p25"],
        "ensemble_50_50": ["ensemble_p15", "ensemble_p20", "ensemble_p25"],
        "r20_regression_isotonic": ["r20_iso_p15", "r20_iso_p20", "r20_iso_p25"],
        "r20_regression_platt": ["r20_platt_p15", "r20_platt_p20", "r20_platt_p25"],
    }
    for candidate, columns in persisted_candidates.items():
        aggregate.extend(
            _candidate_metrics(
                predictions, predictions[columns].to_numpy(), candidate, "all_tests"
            )
        )
    aggregate_df = pd.DataFrame(aggregate)
    aggregate_df.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)

    p25_candidates = aggregate_df[aggregate_df["target"] == "p25_safe"].copy()
    fold_p25 = metrics_df[metrics_df["target"] == "p25_safe"]
    minimum_fold_skill = fold_p25.groupby("candidate")[
        "brier_skill_vs_constant"
    ].min()
    p25_candidates["minimum_fold_brier_skill"] = p25_candidates["candidate"].map(
        minimum_fold_skill
    )
    qualified = p25_candidates[
        (p25_candidates["ece_10"] <= 0.05)
        & (p25_candidates["brier_skill_vs_constant"] > 0)
    ].copy()
    if qualified.empty:
        qualified = p25_candidates.copy()
    qualified["all_folds_positive"] = qualified["minimum_fold_brier_skill"] > 0
    selected = qualified.sort_values(
        ["all_folds_positive", "brier_skill_vs_constant", "pr_auc"],
        ascending=False,
    ).iloc[0]
    selected_candidate = str(selected["candidate"])
    selected_columns = persisted_candidates[selected_candidate]
    predictions[["p15_safe", "p20_safe", "p25_safe"]] = predictions[
        selected_columns
    ].to_numpy()
    predictions.to_parquet(args.output_dir / "walk_forward_predictions.parquet", index=False)

    subgroup_df = pd.DataFrame(_subgroup_metrics(predictions))
    subgroup_df.to_csv(args.output_dir / "p25_subgroup_metrics.csv", index=False)

    p25 = aggregate_df[
        (aggregate_df["candidate"] == selected_candidate)
        & (aggregate_df["target"] == "p25_safe")
    ].iloc[0]
    deciles = _decile_table(predictions["target_p25_safe"], predictions["p25_safe"])
    decile_rates = np.asarray([row["event_rate"] for row in deciles], dtype=float)
    deciles_monotonic = bool((np.diff(decile_rates) >= -0.01).all())
    fold_decile_monotonic = {}
    for fold_name, group in predictions.groupby("fold"):
        table = _decile_table(group["target_p25_safe"], group["p25_safe"])
        rates = np.asarray([row["event_rate"] for row in table], dtype=float)
        fold_decile_monotonic[str(fold_name)] = bool((np.diff(rates) >= -0.01).all())
    all_folds_positive = bool(selected["minimum_fold_brier_skill"] > 0)
    acceptance = {
        "ece_10_lte_0_05": bool(p25["ece_10"] <= 0.05),
        "positive_brier_skill": bool(p25["brier_skill_vs_constant"] > 0),
        "positive_brier_skill_in_every_fold": all_folds_positive,
        "decile_event_rates_monotonic_with_1pp_tolerance": deciles_monotonic,
        "decile_monotonic_in_every_fold": bool(all(fold_decile_monotonic.values())),
        "ordered_probabilities": bool(
            (
                predictions["p15_safe"] >= predictions["p20_safe"]
            ).all()
            and (predictions["p20_safe"] >= predictions["p25_safe"]).all()
        ),
        "eligible_for_shadow_review": False,
        "note": "Offline metrics are necessary but explicit review is required before shadow integration.",
    }
    report = {
        "experiment": "r20-target-prob-v2",
        "status": "offline_exploration",
        "production_impact": "none",
        "selected_candidate": selected_candidate,
        "label_audit": audit,
        "folds": fold_report,
        "p25_primary_metrics": p25.to_dict(),
        "p25_deciles": deciles,
        "p25_fold_decile_monotonic": fold_decile_monotonic,
        "acceptance": acceptance,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_jsonable),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
