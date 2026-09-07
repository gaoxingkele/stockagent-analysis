#!/usr/bin/env python
"""Run the one-time full-universe S20-20R portable confirmation."""
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
sys.path.insert(0, str(ROOT / "scripts"))

from explore_r20_target_prob_v2 import (  # noqa: E402
    _apply_calibrators,
    _apply_platt_calibrators,
    _fit_binary_models,
    _fit_calibrators,
    _fit_ordinal,
    _fit_platt_calibrators,
    _predict_binary_models,
)
from stockagent_analysis.r20_target_prob import tier_probabilities  # noqa: E402
from stockagent_analysis.s20 import daily_topk_metrics  # noqa: E402
from train_s20_20r_ranker import (  # noqa: E402
    TOP_RANK_FEATURES,
    _daily_percentile,
    _fit_ranker,
    _ranking_frame,
)
from train_s20_20r_residual import (  # noqa: E402
    DEFAULT_SEEDS,
    PORTABLE_EXCLUDED_FEATURES,
    _fit_anchor,
    _prepare,
    _refit_anchor,
    _residual_probability,
)


CONFIRMATION_START = "20260127"
CONFIRMATION_END = "20260805"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-bps", type=int, default=5000)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "output/experiments/s20_v2_labels/labels.parquet",
    )
    parser.add_argument(
        "--factor-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_confirmation/factor_groups",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_20r_confirmation/run_v1",
    )
    return parser.parse_args()


def _load_confirmation(args, features):
    label_columns = [
        "ts_code",
        "trade_date",
        "horizon_end_date",
        "positive20",
        "class20",
    ]
    labels = pd.read_parquet(args.labels, columns=label_columns)
    labels["ts_code"] = labels["ts_code"].astype(str)
    labels["trade_date"] = labels["trade_date"].astype(str)
    parts = []
    raw_features = [
        feature
        for feature in features
        if feature
        not in {
            "industry_id",
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
        }
    ]
    for path in sorted(args.factor_dir.glob("group_*.parquet")):
        part = pd.read_parquet(
            path, columns=["ts_code", "trade_date", "industry", *raw_features]
        )
        parts.append(part)
    factors = pd.concat(parts, ignore_index=True)
    factors["ts_code"] = factors["ts_code"].astype(str)
    factors["trade_date"] = factors["trade_date"].astype(str)
    confirmation = factors.merge(
        labels,
        on=["ts_code", "trade_date"],
        how="inner",
        validate="one_to_one",
    )

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
    extra = pd.read_parquet(ROOT / "output/regime_extra/regime_extra.parquet")
    extra["trade_date"] = extra["trade_date"].astype(str)
    regime = regime.merge(extra, on="trade_date", how="left")
    regime_columns = [
        "trade_date",
        *[feature for feature in features if feature in regime.columns],
    ]
    confirmation = confirmation.merge(
        regime[regime_columns], on="trade_date", how="left", validate="many_to_one"
    )
    meta = json.loads(
        (ROOT / "output/lgbm_maxgain/feature_meta.json").read_text(encoding="utf-8")
    )
    industry_map = meta.get("industry_map", {})
    confirmation["industry_id"] = (
        confirmation["industry"]
        .fillna("unknown")
        .astype(str)
        .map(lambda value: industry_map.get(value, -1))
    )
    basic = pd.read_parquet(
        ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "name"]
    )
    st_codes = set(
        basic.loc[basic["name"].fillna("").str.contains("ST", regex=False), "ts_code"]
        .astype(str)
    )
    confirmation = confirmation[
        ~confirmation["ts_code"].isin(st_codes)
        & (confirmation["positive20"] >= 0)
    ].copy()
    for feature in features:
        if feature not in confirmation:
            confirmation[feature] = np.nan
        confirmation[feature] = pd.to_numeric(
            confirmation[feature], errors="coerce"
        )
    return confirmation.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _topk_rows(frame, candidate, column):
    return [
        {
            "candidate": candidate,
            **daily_topk_metrics(
                frame, probability_col=column, target_col="positive20", k=k
            ),
        }
        for k in (10, 20, 50)
    ]


def _monthly_top20(frame, candidate, column):
    rows = []
    work = frame.copy()
    work["month"] = work["trade_date"].str[:6]
    for month, group in work.groupby("month"):
        rows.append(
            {
                "month": month,
                "candidate": candidate,
                **daily_topk_metrics(
                    group,
                    probability_col=column,
                    target_col="positive20",
                    k=20,
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    development, features, audit = _prepare(args)
    features = [feature for feature in features if feature not in PORTABLE_EXCLUDED_FEATURES]
    mature = development["horizon_end_date"].astype(str) < CONFIRMATION_START
    dates = development["trade_date"].astype(str)
    ends = development["horizon_end_date"].astype(str)

    anchor_fit = development[(dates <= "20250630") & (ends < "20250701")]
    anchor_tune = development[
        dates.between("20250701", "20250831") & (ends < "20250901")
    ]
    final_anchor_fit = development[mature]
    residual_fit = development[
        dates.between("20250901", "20251031")
        & (ends < "20251101")
        & (development["positive20"] >= 0)
    ].copy()
    residual_tune = development[
        dates.between("20251101", "20260126")
        & mature
        & (development["positive20"] >= 0)
    ].copy()
    reference_calibration = development[
        dates.between("20251101", "20260126") & mature
    ].copy()
    print(
        {
            "anchor_fit": len(anchor_fit),
            "anchor_tune": len(anchor_tune),
            "anchor_refit": len(final_anchor_fit),
            "residual_fit": len(residual_fit),
            "residual_tune": len(residual_tune),
            "reference_calibration": len(reference_calibration),
        },
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    early_anchor = _fit_anchor(
        anchor_fit, anchor_tune, features, seeds[0] - 503, args.num_threads
    )
    residual_fit["r20_anchor"] = early_anchor.predict(
        residual_fit[features], num_iteration=early_anchor.best_iteration
    )
    final_anchor = _refit_anchor(
        final_anchor_fit,
        features,
        seeds[0] - 503,
        args.num_threads,
        early_anchor.best_iteration,
    )
    residual_tune["r20_anchor"] = final_anchor.predict(residual_tune[features])

    residual_models = []
    from train_s20_20r_residual import _fit_residual

    for seed in seeds:
        model = _fit_residual(
            residual_fit, residual_tune, features, seed, args.num_threads
        )
        model.save_model(str(args.output_dir / f"residual_seed{seed}.txt"))
        residual_models.append(model)

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
    rank_fit, rank_features, fit_groups = _ranking_frame(
        residual_fit, features, selected
    )
    rank_tune, tune_features, tune_groups = _ranking_frame(
        residual_tune, features, selected
    )
    if rank_features != tune_features:
        raise AssertionError("final rank feature schemas differ")
    rankers = []
    for seed in seeds:
        model = _fit_ranker(
            rank_fit,
            rank_tune,
            rank_features,
            fit_groups,
            tune_groups,
            seed,
            args.num_threads,
        )
        model.save_model(str(args.output_dir / f"ranker_seed{seed}.txt"))
        rankers.append(model)

    # Freeze the original R20-P v2 algorithm for the displayed reference probability.
    ordinal = _fit_ordinal(
        anchor_fit, anchor_tune, features, 20260831, args.num_threads
    )
    direct = _fit_binary_models(
        anchor_fit, anchor_tune, features, 20260831, args.num_threads
    )
    ordinal_cal_raw = tier_probabilities(
        ordinal.predict(reference_calibration[features])
    )
    ordinal_iso = _fit_calibrators(ordinal_cal_raw, reference_calibration)
    direct_cal_raw = _predict_binary_models(direct, reference_calibration, features)
    direct_platt = _fit_platt_calibrators(
        direct_cal_raw, reference_calibration, use_logit=True
    )

    confirmation = _load_confirmation(args, features).copy()
    confirmation["r20_anchor"] = final_anchor.predict(confirmation[features])
    confirmation["stage1_probability"], _ = _residual_probability(
        residual_models, confirmation, features
    )
    confirmation_rank, confirmation_features, _ = _ranking_frame(
        confirmation, features, selected
    )
    if confirmation_features != rank_features:
        raise AssertionError("confirmation feature schema differs from training")
    confirmation_rank["lambdarank_score"] = np.mean(
        [model.predict(confirmation_rank[rank_features]) for model in rankers], axis=0
    )
    confirmation_rank["s20_20r_rank"] = 0.5 * _daily_percentile(
        confirmation_rank, "stage1_probability"
    ) + 0.5 * _daily_percentile(confirmation_rank, "lambdarank_score")
    ordinal_raw = tier_probabilities(ordinal.predict(confirmation_rank[features]))
    ordinal_probability = _apply_calibrators(ordinal_iso, ordinal_raw)
    direct_raw = _predict_binary_models(direct, confirmation_rank, features)
    direct_probability = _apply_platt_calibrators(
        direct_platt, direct_raw, use_logit=True
    )
    confirmation_rank["r20_p20_reference"] = (
        0.5 * ordinal_probability[:, 1] + 0.5 * direct_probability[:, 1]
    )

    candidates = {
        "s20_20r_rank": "s20_20r_rank",
        "r20_reference": "r20_p20_reference",
        "stage1_residual": "stage1_probability",
    }
    topk_rows = []
    monthly_rows = []
    for candidate, column in candidates.items():
        topk_rows.extend(_topk_rows(confirmation_rank, candidate, column))
        monthly_rows.extend(_monthly_top20(confirmation_rank, candidate, column))
    topk = pd.DataFrame(topk_rows)
    monthly = pd.DataFrame(monthly_rows)
    primary = topk[topk["k"] == 20].set_index("candidate")
    selected_top20 = (
        confirmation_rank.sort_values(
            ["trade_date", "s20_20r_rank", "ts_code"],
            ascending=[True, False, True],
        )
        .groupby("trade_date", sort=False)
        .head(20)
    )
    selected_probability_error = float(
        abs(
            selected_top20["positive20"].mean()
            - selected_top20["r20_p20_reference"].mean()
        )
    )
    monthly_pivot = monthly.pivot(
        index="month", columns="candidate", values="precision"
    )
    worst_month_delta = float(
        (monthly_pivot["s20_20r_rank"] - monthly_pivot["r20_reference"]).min()
    )
    checks = {
        "aggregate_top20_precision_improvement_vs_r20_reference_pp_gte_2": bool(
            primary.loc["s20_20r_rank", "precision"]
            - primary.loc["r20_reference", "precision"]
            >= 0.02
        ),
        "aggregate_top20_lift_gte_1_5": bool(
            primary.loc["s20_20r_rank", "lift"] >= 1.5
        ),
        "positive_pick_days_rate_gte_0_9": bool(
            primary.loc["s20_20r_rank", "positive_pick_days_rate"] >= 0.9
        ),
        "worst_month_precision_regression_vs_r20_reference_pp_lte_2": bool(
            worst_month_delta >= -0.02
        ),
        "r20_reference_probability_error_on_selected_top20_lte_0_05": bool(
            selected_probability_error <= 0.05
        ),
    }
    report = {
        "contract_version": "s20-20r-r20-anchor-v1.4-20260907",
        "status": "confirmation_passed_ready_for_shadow"
        if all(checks.values())
        else "confirmation_failed_not_shadow_eligible",
        "confirmation_window": [CONFIRMATION_START, CONFIRMATION_END],
        "development_sample_bps": args.sample_bps,
        "feature_count": len(features),
        "development_audit": audit,
        "confirmation_rows": len(confirmation_rank),
        "confirmation_symbols": int(confirmation_rank["ts_code"].nunique()),
        "confirmation_dates": int(confirmation_rank["trade_date"].nunique()),
        "topk_metrics": topk_rows,
        "monthly_top20": monthly_rows,
        "selected_top20": {
            "event_rate": float(selected_top20["positive20"].mean()),
            "r20_reference_mean_probability": float(
                selected_top20["r20_p20_reference"].mean()
            ),
            "absolute_probability_error": selected_probability_error,
        },
        "worst_month_precision_delta_vs_r20_reference": worst_month_delta,
        "checks": checks,
        "passed": all(checks.values()),
        "confirmation_open_count": 1,
    }
    columns = [
        "ts_code",
        "trade_date",
        "positive20",
        "class20",
        "s20_20r_rank",
        "stage1_probability",
        "lambdarank_score",
        "r20_p20_reference",
    ]
    confirmation_rank[columns].to_parquet(
        args.output_dir / "predictions.parquet", index=False
    )
    topk.to_csv(args.output_dir / "daily_topk_metrics.csv", index=False)
    monthly.to_csv(args.output_dir / "monthly_top20.csv", index=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(topk.to_string(index=False), flush=True)
    print(monthly.to_string(index=False), flush=True)
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
