"""Diagnose why S20-v3 predicted probabilities cannot be read as hit rates.

The v3 trainer fits a *static* six-class logistic calibrator on a short mature
window, then applies those intercepts to later months. TopN ranking further
selects the optimistic tail. This module measures those two gaps on saved
artifacts and reproduces the base-rate trap on synthetic data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

CLASS_NAMES = (
    "immediate20", "immediate15", "delayed20",
    "negative_flat", "negative_down", "negative_unsafe15",
)
IMMEDIATE = {0, 1}
DOWN = {2, 4, 5}


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def implied_prior(intercepts) -> dict:
    p = softmax(intercepts)
    return {
        "class_probs": {name: float(p[i]) for i, name in enumerate(CLASS_NAMES)},
        "p_immediate": float(p[0] + p[1]),
        "p_down": float(p[2] + p[4] + p[5]),
        "p_negative": float(p[3] + p[4] + p[5]),
    }


def class_mix(frame: pd.DataFrame, class_col: str = "s20_class") -> dict:
    counts = frame[class_col].value_counts().to_dict()
    n = len(frame)
    mix = {name: float(counts.get(i, 0) / n) if n else 0.0 for i, name in enumerate(CLASS_NAMES)}
    mix["p_immediate"] = mix["immediate20"] + mix["immediate15"]
    mix["p_down"] = mix["delayed20"] + mix["negative_down"] + mix["negative_unsafe15"]
    mix["rows"] = int(n)
    mix["dates"] = int(frame["trade_date"].nunique()) if "trade_date" in frame.columns else None
    return mix


def reliability_table(y, p, bins=10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges, right=True) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "p_lo": float(edges[b]),
            "p_hi": float(edges[b + 1]),
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "actual_rate": float(y[mask].mean()),
            "gap": float(p[mask].mean() - y[mask].mean()),
        })
    return pd.DataFrame(rows)


def top_rows(frame: pd.DataFrame, column: str, k: int = 20) -> pd.DataFrame:
    return (
        frame.sort_values(["trade_date", column, "ts_code"], ascending=[True, False, True])
        .groupby("trade_date", sort=False)
        .head(k)
    )


def selection_gap(frame: pd.DataFrame, p_col: str, y_col: str, score_col: str, k: int = 20) -> dict:
    selected = top_rows(frame, score_col, k)
    return {
        "universe_mean_p": float(frame[p_col].mean()),
        "universe_actual": float(frame[y_col].mean()),
        "universe_gap": float(frame[p_col].mean() - frame[y_col].mean()),
        "topn_mean_p": float(selected[p_col].mean()),
        "topn_actual": float(selected[y_col].mean()),
        "topn_gap": float(selected[p_col].mean() - selected[y_col].mean()),
        "n_selected": int(len(selected)),
        "dates": int(selected.trade_date.nunique()),
    }


def static_calibrator_base_rate_trap(
    p_raw_cal, y_cal, p_raw_test, y_test, seed: int = 0
) -> dict:
    """Reproduce v3: logistic on log(raw p) fitted on cal, applied to a shifted test period."""
    cal = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    x_cal = np.log(np.clip(np.asarray(p_raw_cal, dtype=float), 1e-8, 1.0))
    if x_cal.ndim == 1:
        x_cal = x_cal.reshape(-1, 1)
        x_test = np.log(np.clip(np.asarray(p_raw_test, dtype=float), 1e-8, 1.0)).reshape(-1, 1)
        cal.fit(x_cal, np.asarray(y_cal, dtype=int))
        p_hat = cal.predict_proba(x_test)[:, 1]
    else:
        x_test = np.log(np.clip(np.asarray(p_raw_test, dtype=float), 1e-8, 1.0))
        cal.fit(x_cal, np.asarray(y_cal, dtype=int))
        p_hat = cal.predict_proba(x_test)
        if p_hat.ndim == 2:
            p_hat = p_hat[:, 1] if p_hat.shape[1] == 2 else p_hat.sum(axis=1)
    y_test = np.asarray(y_test, dtype=float)
    return {
        "cal_actual": float(np.mean(y_cal)),
        "test_actual": float(y_test.mean()),
        "test_mean_predicted": float(np.mean(p_hat)),
        "test_gap": float(np.mean(p_hat) - y_test.mean()),
        "brier": float(brier_score_loss(y_test, p_hat)),
        "constant_brier": float(brier_score_loss(y_test, np.full_like(y_test, y_test.mean()))),
    }


def monthly_reliability(frame: pd.DataFrame, p_col: str, y_col: str) -> pd.DataFrame:
    rows = []
    month = frame["trade_date"].astype(str).str[:6]
    for key, g in frame.groupby(month, sort=True):
        rows.append({
            "month": key,
            "dates": int(g.trade_date.nunique()),
            "rows": int(len(g)),
            "mean_predicted": float(g[p_col].mean()),
            "actual": float(g[y_col].mean()),
            "gap": float(g[p_col].mean() - g[y_col].mean()),
        })
    return pd.DataFrame(rows)


def audit_saved_v3(root: Path) -> dict:
    out = Path(root) / "output/experiments/s20_v3"
    labels = pd.read_parquet(out / "labels.parquet", columns=["ts_code", "trade_date", "s20_class", "immediate", "down_risk"])
    labels = labels[labels.s20_class >= 0]
    pred = pd.read_parquet(out / "predictions.parquet")
    diag = pred[pred["period"] == "diagnostic2026"].copy()
    if "mode" in diag.columns:
        diag_full = diag[diag["mode"] == "portable_full"].copy()
    else:
        diag_full = diag
    schema = json.loads((out / "models/diagnostic2026/portable_full/schema.json").read_text(encoding="utf-8"))
    prior = implied_prior(schema["calibration_intercepts"])
    cal_window = labels[labels.trade_date.between("20251103", "20251225")]
    test_window = labels[labels.trade_date.between("20260127", "20260805")]
    comparison = pd.read_parquet(out / "comparison_diagnostic2026.parquet")
    # 0.5-risk score reconstructed the same way as evaluate_s20_v3_tradeoff.py
    good = comparison["portable_full_up"]
    down = 1 + good - comparison["portable_full"] / 50
    comparison = comparison.copy()
    comparison["p_immediate"] = good
    comparison["p_down"] = down
    comparison["score_equal"] = comparison["portable_full"]
    comparison["score_half"] = 100 * (0.5 + good - 0.5 * down) / 1.5

    report = {
        "calibrator_implied_prior": prior,
        "calibration_window_mix": class_mix(cal_window),
        "diagnostic_label_mix": class_mix(test_window),
        "diagnostic_prediction_universe": {
            "mean_p_immediate": float(diag_full.p_immediate.mean()),
            "actual_immediate": float(diag_full.immediate.mean()),
            "mean_p_down": float(diag_full.p_down.mean()),
            "actual_down": float(diag_full.down_risk.mean()),
            "brier_immediate": float(brier_score_loss(diag_full.immediate, diag_full.p_immediate)),
            "brier_down": float(brier_score_loss(diag_full.down_risk, diag_full.p_down)),
        },
        "selection_immediate_equal_weight": selection_gap(comparison, "p_immediate", "immediate", "score_equal"),
        "selection_down_equal_weight": selection_gap(comparison, "p_down", "down_risk", "score_equal"),
        "selection_immediate_half_risk": selection_gap(comparison, "p_immediate", "immediate", "score_half"),
        "selection_down_half_risk": selection_gap(comparison, "p_down", "down_risk", "score_half"),
        "monthly_immediate": monthly_reliability(diag_full, "p_immediate", "immediate").to_dict("records"),
        "monthly_down": monthly_reliability(diag_full, "p_down", "down_risk").to_dict("records"),
        "reliability_immediate": reliability_table(diag_full.immediate, diag_full.p_immediate).to_dict("records"),
        "reliability_down": reliability_table(diag_full.down_risk, diag_full.p_down).to_dict("records"),
        "window_mixes": {
            name: class_mix(labels[labels.trade_date.between(a, b)])
            for name, (a, b) in {
                "wf1_cal": ("20250102", "20250123"),
                "wf1_test": ("20250303", "20250430"),
                "wf2_cal": ("20250506", "20250530"),
                "wf2_test": ("20250701", "20250829"),
                "wf3_cal": ("20250901", "20250925"),
                "wf3_test": ("20251103", "20260126"),
                "diagnostic_cal": ("20251103", "20251225"),
                "diagnostic_test": ("20260127", "20260805"),
            }.items()
        },
        "mechanism": [
            "static_multiclass_logistic_intercepts_copy_calibration_window_base_rate",
            "test_period_down_rate_roughly_doubles_while_predicted_down_stays_near_cal_prior",
            "TopN_selects_optimistic_tail_so_mean_p_on_picks_is_even_more_inflated",
            "score_is_linear_utility_not_a_hit_probability",
        ],
    }
    return report


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
