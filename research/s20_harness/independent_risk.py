"""Independent P(drawdown ≤ −10%) gate. Ranking scores cannot buy off this check."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .baseline_model import run as fit_binary
from .diagnostic_train import FEATURE_COLUMNS, default_boundaries, _segment_labels
from .policy_replay import dual_improves, evaluate_policy, select_pool

B10_TAUS = (0.12, 0.15, 0.18, 0.20, 0.25, 1.0)
REPLAY_MASS_QUANTILES = (0.50, 0.70, 0.85, 0.95)
TARGET_ID = "P.b10.v4"
MIN_CAL_ROWS = 200
COVERAGE_FLOOR = 0.3


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return (np.log(p) - np.log1p(-p)).reshape(-1, 1)


def b10_labels(four_class: pd.DataFrame) -> pd.DataFrame:
    if "target" not in four_class.columns:
        raise ValueError("paired four-class labels required")
    out = four_class[["sample_id"]].copy()
    out["target"] = four_class["target"].isin(["B", "D"]).map(bool)
    return out


def fit_p_b10(samples, features, labels, *, columns=None):
    bounds = default_boundaries()
    columns = list(columns or FEATURE_COLUMNS)
    y = b10_labels(labels)
    fit = _segment_labels(samples, y, bounds, "fit")
    contract = dict(role="prediction_features", columns=columns, source_provenance_verified=True)
    feat = features[["sample_id", *columns]]
    pred, card = fit_binary(samples, feat, fit, bounds, contract, target_id=TARGET_ID)
    return pred.rename(columns={"raw_probability": "p_b10"}), card


def attach_gate(rank_frame, risk_pred, risk_col="p_b10_cal") -> pd.DataFrame:
    if risk_col not in risk_pred.columns:
        raise ValueError("independent B10 probabilities missing")
    extra = [c for c in ("p_b10", "p_b10_cal", "calibrator_n") if c in risk_pred.columns]
    cols = list(dict.fromkeys(["sample_id", risk_col, *extra]))
    out = rank_frame.merge(risk_pred[cols], on="sample_id", how="left")
    if out[risk_col].isna().all():
        raise ValueError("independent B10 probabilities missing")
    return out


def rolling_calibrate_p_b10(samples, raw_pred, labels, *, min_cal_rows=MIN_CAL_ROWS):
    """Expanding-window Platt on logit(raw p). Only labels matured before prediction_at."""
    needed = {"sample_id", "prediction_at", "label_available_at"}
    if not needed.issubset(samples.columns):
        raise ValueError("prediction_at and label_available_at required for rolling B10 calibration")
    if "p_b10" not in raw_pred.columns or "segment" not in raw_pred.columns:
        raise ValueError("raw independent B10 probabilities missing")
    y = b10_labels(labels)
    frame = samples[["sample_id", "prediction_at", "label_available_at"]].merge(
        raw_pred[["sample_id", "p_b10", "segment"]], on="sample_id"
    )
    frame = frame.merge(y, on="sample_id", how="left")
    frame["_pred"] = pd.to_datetime(frame["prediction_at"], utc=True)
    frame["_lab"] = pd.to_datetime(frame["label_available_at"], utc=True)
    frame["p_b10_cal"] = np.nan
    frame["calibrator_n"] = 0
    forward = frame.segment.isin(["selection-policy", "outer-test"])
    dates = sorted(frame.loc[forward, "_pred"].dropna().unique())
    last_model = None
    last_n = 0
    for ts in dates:
        mature = frame["_lab"].lt(ts) & frame.target.notna() & frame.p_b10.notna()
        cal = frame.loc[mature]
        if len(cal) >= min_cal_rows and int(cal.target.nunique()) == 2:
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            model.fit(_logit(cal.p_b10), cal.target.astype(int))
            last_model, last_n = model, int(len(cal))
        day = forward & frame["_pred"].eq(ts) & frame.p_b10.notna()
        if not day.any():
            continue
        if last_model is None:
            frame.loc[day, "p_b10_cal"] = frame.loc[day, "p_b10"]
        else:
            frame.loc[day, "p_b10_cal"] = last_model.predict_proba(_logit(frame.loc[day, "p_b10"]))[:, 1]
        frame.loc[day, "calibrator_n"] = last_n
    leftover = forward & frame.p_b10.notna() & frame.p_b10_cal.isna()
    if leftover.any():
        frame.loc[leftover, "p_b10_cal"] = frame.loc[leftover, "p_b10"]
    return frame.drop(columns=["target", "_pred", "_lab"])


def calibration_snapshot(frame, labels) -> dict:
    y = b10_labels(labels)
    merged = frame.merge(y, on="sample_id", how="left")
    def _part(rows):
        if "p_b10_cal" not in rows.columns:
            return dict(n=0)
        scored = rows[rows.p_b10_cal.notna() & rows.target.notna()]
        if not len(scored):
            return dict(n=0)
        out = dict(
            n=int(len(scored)),
            mean_cal=float(scored.p_b10_cal.mean()),
            realized_b10=float(scored.target.mean()),
        )
        if "p_b10" in scored.columns:
            out["mean_raw"] = float(scored.p_b10.mean())
        if "calibrator_n" in scored.columns and scored.calibrator_n.notna().any():
            out["calibrator_n_min"] = int(scored.calibrator_n.min())
            out["calibrator_n_max"] = int(scored.calibrator_n.max())
        return out
    snap = dict(all_scored=_part(merged))
    if "segment" in merged.columns:
        for seg in ("selection-policy", "outer-test"):
            snap[seg] = _part(merged[merged.segment == seg])
    return snap


def replay_mass_taus(dream_frame, *, risk_col="p_b10_cal", n_cap=20,
                     quantiles=REPLAY_MASS_QUANTILES):
    """Extra τ from replay-segment calibrated score mass. No labels, no outer."""
    if risk_col not in dream_frame.columns:
        raise ValueError("independent risk column missing")
    picked = select_pool(dream_frame, n_cap=n_cap, max_risk=1.0, risk_col=risk_col)
    mass = picked.loc[picked.selected, risk_col].astype(float).dropna()
    if mass.empty:
        return ()
    values = []
    registered = set(B10_TAUS)
    for q in quantiles:
        tau = round(float(mass.quantile(float(q))), 2)
        if 0.0 < tau < 1.0 and tau not in registered:
            registered.add(tau)
            values.append(tau)
    return tuple(values)


def tau_grid(pi0=None, risk_col="p_b10_cal", extra_taus=()):
    if pi0 is None:
        pi0 = dict(n_cap=20, ranking="penalized_utility", lambda_=1.0, mu=2.0, nu=0.25,
                   max_risk=1.0, risk_col=risk_col)
    if pi0.get("risk_col") != risk_col:
        pi0 = dict(pi0, risk_col=risk_col)
    grid = [dict(pi0)]
    for tau in (*B10_TAUS, *extra_taus):
        item = dict(pi0, max_risk=float(tau))
        if item not in grid:
            grid.append(item)
    return grid, pi0


def replay_b10_gate(dream_frame, dream_labels, online_frame, online_labels, *, extra_taus=None):
    mass_taus = replay_mass_taus(dream_frame) if extra_taus is None else tuple(extra_taus)
    grid, pi0 = tau_grid(extra_taus=mass_taus)
    dream_scores = [evaluate_policy(dream_frame, dream_labels, p) for p in grid]
    incumbent = next(s for s in dream_scores if s["policy"] == pi0)
    def _eligible(score):
        if score["policy"] == pi0:
            return True
        if (score.get("coverage") or 0) < COVERAGE_FLOOR:
            return False
        return dual_improves(score, incumbent)
    eligible = [s for s in dream_scores if _eligible(s)]
    champion = max(eligible, key=lambda s: (s["hold_profit_rate"] is not None,
                                            s["hold_profit_rate"] or -1,
                                            -(s["hold_drawdown_rate"] or 1)))
    online_pi0 = evaluate_policy(online_frame, online_labels, pi0)
    online_champ = evaluate_policy(online_frame, online_labels, champion["policy"])
    online_grid = [evaluate_policy(online_frame, online_labels, p) for p in grid]
    transferred = dual_improves(online_champ, online_pi0)
    return dict(
        pi0=pi0,
        champion_policy=champion["policy"],
        shipped_equals_pi0=champion["policy"] == pi0,
        online_transfer_ok=transferred,
        recommended_policy=champion["policy"] if transferred else pi0,
        dream_pi0=incumbent,
        dream_champion=champion,
        dream_grid=dream_scores,
        online_grid=online_grid,
        online_pi0=online_pi0,
        online_champion=online_champ,
        online_recommended=evaluate_policy(online_frame, online_labels,
                                           champion["policy"] if transferred else pi0),
        n_policies=len(grid),
        n_dream_improvers=len(eligible) - 1,
        replay_mass_taus=list(mass_taus),
        coverage_floor=COVERAGE_FLOOR,
        formal_training_authorized=False,
        production_eligible=False,
        gate="rolling-calibrated independent P(B10); ranking cannot offset the threshold",
        calibration="expanding Platt on logit(raw p); labels only if label_available_at < prediction_at",
        tau_source="registered B10_TAUS plus replay-segment calibrated mass quantiles; never outer",
    )
