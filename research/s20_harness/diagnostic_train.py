"""Bounded P-track diagnostic training on local daily quotes.

This is not H01/H02/H04 acceptance and does not authorize production.
It fits registered diagnostic families on market-calendar safe-profit labels
with 20-session delayed label availability, then reports TopN reliability.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .joint_run import build
from .labels import P_TRACK_BUY_COST, P_TRACK_SELL_COST
from .rsi_features import RSI_COLUMNS, compute as rsi_compute, shuffle_time
from .runtime import atomic_json, digest, now

FEATURE_COLUMNS = [
    "raw_return20", "raw_ma20_distance", "raw_mean_tr14_pct",
    "raw_return_vol20", "volume_ratio20",
]
SHUFFLED_RSI_COLUMNS = tuple(f"shuffle_{name}" for name in RSI_COLUMNS)
ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + list(RSI_COLUMNS) + list(SHUFFLED_RSI_COLUMNS)


def load_quotes(daily_dir: Path) -> tuple[list[str], pd.DataFrame]:
    files = sorted(daily_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("no daily parquet files")
    parts = []
    for path in files:
        frame = pd.read_parquet(path, columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol"])
        parts.append(frame)
    daily = pd.concat(parts, ignore_index=True)
    daily["ts_code"] = daily["ts_code"].astype(str)
    daily["trade_date"] = daily["trade_date"].map(lambda x: str(int(x)) if str(x).isdigit() else str(x))
    daily = daily[daily["ts_code"].str.endswith((".SH", ".SZ"))].copy()
    if daily.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate quotes")
    calendar = sorted(daily["trade_date"].unique())
    return calendar, daily.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _pivot(daily: pd.DataFrame, calendar: list[str], stocks: list[str], column: str) -> np.ndarray:
    table = daily.pivot(index="trade_date", columns="ts_code", values=column)
    return table.reindex(index=calendar, columns=stocks).to_numpy(dtype=float)


def build_panel(calendar, daily, *, stock_mod=7, date_stride=5):
    stocks = sorted(
        code for code in daily["ts_code"].unique()
        if code.endswith((".SH", ".SZ")) and int(code[:6]) % stock_mod == 0
    )
    opens = _pivot(daily, calendar, stocks, "open")
    highs = _pivot(daily, calendar, stocks, "high")
    lows = _pivot(daily, calendar, stocks, "low")
    closes = _pivot(daily, calendar, stocks, "close")
    vols = _pivot(daily, calendar, stocks, "vol")
    lookback, horizon = 20, 20
    signal_idx = list(range(lookback, len(calendar) - horizon, date_stride))
    buy, sell = P_TRACK_BUY_COST, P_TRACK_SELL_COST
    samples, features, labels = [], [], []
    for i in signal_idx:
        signal = calendar[i]
        if signal < "20240601" or signal >= "20260101":
            continue
        entry = opens[i + 1]
        exit_px = closes[i + horizon]
        with np.errstate(all="ignore"):
            window_low = np.nanmin(lows[i + 1:i + 1 + horizon], axis=0)
        window_ok = np.isfinite(entry) & (entry > 0) & np.isfinite(exit_px) & np.isfinite(window_low)
        look = closes[i - lookback:i + 1]
        look_high = highs[i - lookback:i + 1]
        look_low = lows[i - lookback:i + 1]
        look_vol = vols[i - lookback:i + 1]
        complete = np.isfinite(look).all(axis=0) & np.isfinite(look_high).all(axis=0) & np.isfinite(look_low).all(axis=0)
        complete &= (look > 0).all(axis=0) & (look_low > 0).all(axis=0)
        usable = window_ok & complete
        if not usable.any():
            continue
        entry_cost = entry * (1.0 + buy)
        terminal = exit_px * (1.0 - sell) / entry_cost - 1.0
        b5 = window_low < entry_cost * 0.95
        up = terminal > 0
        cls = np.where(up & ~b5, "A", np.where(up & b5, "B", np.where(~up & ~b5, "C", "D")))
        prev = look[:-1]
        tr = np.maximum(look_high[1:] - look_low[1:], np.maximum(np.abs(look_high[1:] - prev), np.abs(look_low[1:] - prev)))
        rets = look[1:] / look[:-1] - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            feat = np.column_stack([
                look[-1] / look[0] - 1.0,
                look[-1] / look[1:].mean(axis=0) - 1.0,
                tr[-14:].mean(axis=0) / look[-1],
                rets.std(axis=0, ddof=0),
                look_vol[-1] / np.maximum(look_vol[:-1].mean(axis=0), 1e-12),
            ])
        rsi = rsi_compute(look, look_high, look_low)
        shuffled = rsi_compute(*shuffle_time(look, look_high, look_low, np.random.default_rng(int(signal) ^ 20260918)))
        extra = np.column_stack([rsi[name] for name in RSI_COLUMNS] + [shuffled[name] for name in RSI_COLUMNS])
        feat = np.column_stack([feat, extra])
        feature_names = ALL_FEATURE_COLUMNS
        pred_at = f"{signal[:4]}-{signal[4:6]}-{signal[6:]}T21:00:00+08:00"
        feat_at = f"{signal[:4]}-{signal[4:6]}-{signal[6:]}T15:00:00+08:00"
        end = calendar[i + horizon]
        label_at = f"{end[:4]}-{end[4:6]}-{end[6:]}T15:01:00+08:00"
        horizon_at = f"{end[:4]}-{end[4:6]}-{end[6:]}T15:00:00+08:00"
        for j in np.flatnonzero(usable):
            sid = f"{stocks[j]}-{signal}"
            samples.append(dict(
                sample_id=sid, entity_id=stocks[j], signal_date=signal,
                prediction_at=pred_at, feature_available_at=feat_at,
                horizon_close_at=horizon_at, label_available_at=label_at,
            ))
            row = {"sample_id": sid}
            row.update({name: float(feat[j, k]) for k, name in enumerate(feature_names)})
            features.append(row)
            labels.append(dict(sample_id=sid, target=str(cls[j])))
    if len(samples) > 100000:
        raise ValueError("diagnostic sample cap exceeded")
    return pd.DataFrame(samples), pd.DataFrame(features), pd.DataFrame(labels)


def default_boundaries():
    return [
        dict(name="fit", start_at="2024-06-01T00:00:00+08:00", end_at="2024-12-04T00:00:00+08:00"),
        dict(name="tune", start_at="2025-01-21T00:00:00+08:00", end_at="2025-03-01T00:00:00+08:00"),
        dict(name="calibration", start_at="2025-03-01T00:00:00+08:00", end_at="2025-06-01T00:00:00+08:00"),
        dict(name="selection-policy", start_at="2025-07-01T00:00:00+08:00", end_at="2025-09-01T00:00:00+08:00"),
        dict(name="outer-test", start_at="2025-09-01T00:00:00+08:00", end_at="2026-01-01T00:00:00+08:00"),
    ]


def _segment_labels(samples, labels, boundaries, segment):
    from .splits import assign_segments, training_ids
    assignment, _ = assign_segments(samples, boundaries)
    ids = set(training_ids(assignment, segment))
    return labels[labels.sample_id.isin(ids)].copy()


def plan_for(samples, features, labels, calendar, family, *, seed=20, weights=None, columns=None):
    bounds = default_boundaries()
    fit = _segment_labels(samples, labels, bounds, "fit")
    cal = _segment_labels(samples, labels, bounds, "calibration")
    policy = dict(
        selection=dict(
            policy_id="v4-diag-joint", target_id="P.safe.v4", risk_target_id="P.down5.v4",
            mode="risk_gated", frozen_at="2024-05-01T00:00:00+08:00",
            n_cap=20, min_score=0.0, max_risk=0.45,
        ),
        weights={"lambda": 1.0, "mu": 2.0, "nu": 0.25},
        ranking="penalized_utility",
    )
    columns = list(columns or FEATURE_COLUMNS)
    feat = features[["sample_id", *columns]]
    plan = dict(
        schema_version="joint-1", evidence_mode="supplied_reference", target_id="P.joint.v4",
        samples=samples.to_dict("records"), features=feat.to_dict("records"),
        fit_labels=fit.to_dict("records"), calibration_labels=cal.to_dict("records"),
        boundaries=bounds,
        feature_contract=dict(role="prediction_features", columns=columns,
                              source_provenance_verified=True),
        policy=policy, calendar=list(calendar),
        model_family=family, random_seed=seed,
        calibration_method="bounded_scalar_temperature",
    )
    if family == "cost_sensitive_joint":
        plan["class_weights"] = weights or {"A": 1.0, "B": 1.5, "C": 1.0, "D": 3.0}
    return plan


def evaluate_ledger(ledger: pd.DataFrame, labels: pd.DataFrame) -> dict:
    merged = ledger.merge(labels, on="sample_id", how="left")
    selected = merged[merged.selected].copy()
    if selected.empty:
        return dict(n_selected=0, coverage=0.0, precision=None, risk=None, mean_pA=None, actual_A=None)
    actual_a = float((selected.target == "A").mean())
    actual_down = float(selected.target.isin(["B", "D"]).mean())
    return dict(
        n_selected=int(len(selected)),
        dates=int(selected.signal_date.nunique()),
        coverage=float(len(selected) / (20 * selected.signal_date.nunique())),
        actual_A=actual_a,
        actual_down=actual_down,
        mean_cal_pA=float(selected.p_A.mean()),
        mean_cal_p_down5=float((selected.p_B + selected.p_D).mean()),
        pA_gap=float(selected.p_A.mean() - actual_a),
        down_gap=float((selected.p_B + selected.p_D).mean() - actual_down),
        class_counts=selected.target.fillna("UNKNOWN").value_counts().to_dict(),
    )


def run(root: Path) -> dict:
    root = Path(root).resolve()
    calendar, daily = load_quotes(root / "output/tushare_cache/daily")
    samples, features, labels = build_panel(calendar, daily)
    if samples.empty:
        raise ValueError("no diagnostic samples")
    out_dir = root / "output/experiments/s20_safe_v4/sources" / "v4-diagnostic-campaign"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = dict(at=now(), rows=len(samples), symbols=int(samples.entity_id.nunique()),
                   dates=int(samples.signal_date.nunique()),
                   class_counts=labels.target.value_counts().to_dict(),
                   formal_training_authorized=False, h01_passed=False)
    atomic_json(out_dir / "dataset.json", dataset)
    samples.to_parquet(out_dir / "samples.parquet", index=False)
    features.to_parquet(out_dir / "features.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    results = []
    for family in ("mature_frequency", "multinomial", "conditional_three"):
        plan = plan_for(samples, features, labels, calendar, family)
        plan_path = out_dir / f"plan_{family}.json"
        atomic_json(plan_path, plan)
        report = build(root, plan_path, digest(plan_path))
        ledger = pd.read_parquet(Path(report["directory"]) / "candidate_ledger.parquet")
        metrics = evaluate_ledger(ledger, labels)
        results.append(dict(family=family, run=report, outer=metrics))
    summary = dict(
        at=now(), status="COMPLETED_DIAGNOSTIC",
        formal_H04_accepted=False, formal_training_authorized=False,
        production_eligible=False, absolute_probability_validated=False,
        dataset=dataset, models=results,
        note="H01 data gate has not passed; raw-quote P-track diagnostic only",
    )
    atomic_json(out_dir / "campaign_summary.json", summary)
    extra = evaluate_saved(out_dir, labels)
    summary["ranking_and_universe"] = extra
    atomic_json(out_dir / "campaign_summary.json", summary)
    return summary


def evaluate_saved(out_dir: Path, labels: pd.DataFrame) -> list:
    from .joint_policy import apply
    from .runtime import load_plan
    summary = load_plan(out_dir / "campaign_summary.json")
    rows = []
    for model in summary["models"]:
        directory = Path(model["run"]["directory"])
        pred = pd.read_parquet(directory / "calibrated_predictions.parquet")
        samples = pd.read_parquet(out_dir / "samples.parquet")
        outer = pred[pred.segment == "outer-test"].copy()
        joined = outer.merge(labels, on="sample_id", how="left")
        universe = {
            "n": int(len(joined)),
            "actual_A": float((joined.target == "A").mean()),
            "actual_down": float(joined.target.isin(["B", "D"]).mean()),
            "mean_cal_pA": float(joined.cal_p_A.mean()),
            "mean_cal_p_down5": float(joined.cal_p_down5.mean()),
            "pA_gap": float(joined.cal_p_A.mean() - (joined.target == "A").mean()),
            "down_gap": float(joined.cal_p_down5.mean() - joined.target.isin(["B", "D"]).mean()),
        }
        candidates = samples.set_index("sample_id").loc[outer.sample_id, ["entity_id", "signal_date", "prediction_at"]].reset_index()
        for cls in "ABCD":
            candidates[f"p_{cls}"] = outer[f"cal_p_{cls}"].to_numpy()
        calendar = sorted(candidates.signal_date.unique().tolist())
        policy = dict(
            selection=dict(
                policy_id="v4-diag-rank", target_id="P.safe.v4", risk_target_id="P.down5.v4",
                mode="risk_gated", frozen_at="2024-05-01T00:00:00+08:00",
                n_cap=20, min_score=0.0, max_risk=1.0,
            ),
            weights={"lambda": 1.0, "mu": 2.0, "nu": 0.25},
            ranking="penalized_utility",
        )
        ledger, _ = apply(candidates, policy, calendar)
        ranked = evaluate_ledger(ledger, labels)
        rows.append(dict(family=model["family"], universe=universe, top20_max_risk_1=ranked))
    atomic_json(out_dir / "ranking_eval.json", rows)
    return rows


def run_rsi_campaign(root: Path) -> dict:
    """Same P-track labels and splits; only the feature allowlist changes."""
    root = Path(root).resolve()
    calendar, daily = load_quotes(root / "output/tushare_cache/daily")
    samples, features, labels = build_panel(calendar, daily)
    out_dir = root / "output/experiments/s20_safe_v4/sources" / "v4-rsi-campaign"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = dict(at=now(), rows=len(samples), symbols=int(samples.entity_id.nunique()),
                   dates=int(samples.signal_date.nunique()),
                   class_counts=labels.target.value_counts().to_dict(),
                   idea="split average gain vs average loss (Wilder RSI), keep magnitudes",
                   formal_training_authorized=False, h01_passed=False)
    atomic_json(out_dir / "dataset.json", dataset)
    samples.to_parquet(out_dir / "samples.parquet", index=False)
    features.to_parquet(out_dir / "features.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    arms = {
        "base": FEATURE_COLUMNS,
        "rsi": FEATURE_COLUMNS + list(RSI_COLUMNS),
        "shuffle": FEATURE_COLUMNS + list(SHUFFLED_RSI_COLUMNS),
    }
    results = []
    for name, columns in arms.items():
        plan = plan_for(samples, features, labels, calendar, "multinomial", columns=columns)
        plan_path = out_dir / f"plan_{name}.json"
        atomic_json(plan_path, plan)
        report = build(root, plan_path, digest(plan_path))
        ledger = pd.read_parquet(Path(report["directory"]) / "candidate_ledger.parquet")
        results.append(dict(family=f"multinomial_{name}", columns=list(columns),
                            run=report, outer=evaluate_ledger(ledger, labels)))
    summary = dict(
        at=now(), status="COMPLETED_DIAGNOSTIC", idea="rsi_split_up_down",
        formal_H04_accepted=False, formal_training_authorized=False,
        production_eligible=False, absolute_probability_validated=False,
        dataset=dataset, models=results,
        note="RSI increment vs time-shuffled RSI vs ATR-containing base; not H01/H04",
    )
    atomic_json(out_dir / "campaign_summary.json", summary)
    extra = evaluate_saved(out_dir, labels)
    summary["ranking_and_universe"] = extra
    atomic_json(out_dir / "campaign_summary.json", summary)
    return summary
