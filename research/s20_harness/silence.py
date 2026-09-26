"""Causal silence screen: matured cooldown and predicted P(silent). Not data cleaning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .baseline_model import run as fit_binary
from .diagnostic_train import FEATURE_COLUMNS, default_boundaries, _segment_labels
from .independent_risk import MIN_CAL_ROWS, _logit
from .policy_replay import (
    PI0_N_CAP,
    dual_improves_specified_rise,
    evaluate_policy,
    select_pool,
)
from .take_profit import SILENCE_MAX_GAIN, is_silent, specified_rise, window_max_gain

SILENT_TARGET_ID = "P.silent.v4"
HIT15_TARGET_ID = "P.hit15.v4"
COVERAGE_FLOOR = 0.3


def silence_from_path(entry, high, path_risk, *, cut=SILENCE_MAX_GAIN) -> bool:
    return is_silent(window_max_gain(entry, high), path_risk, cut=cut)


def silence_labels(frame) -> pd.DataFrame:
    if "silent" not in frame.columns:
        raise ValueError("silence labels required")
    out = frame[["sample_id"]].copy()
    out["target"] = frame["silent"].map(bool)
    return out


def hit15_labels(frame) -> pd.DataFrame:
    if "hit15" not in frame.columns:
        raise ValueError("specified-rise labels required")
    out = frame[["sample_id"]].copy()
    out["target"] = frame["hit15"].map(bool)
    return out


def _fit_event(samples, features, event, *, target_id, raw_col, columns=None):
    bounds = default_boundaries()
    columns = list(columns or FEATURE_COLUMNS)
    fit = _segment_labels(samples, event, bounds, "fit")
    contract = dict(role="prediction_features", columns=columns, source_provenance_verified=True)
    feat = features[["sample_id", *columns]]
    pred, card = fit_binary(samples, feat, fit, bounds, contract, target_id=target_id)
    return pred.rename(columns={"raw_probability": raw_col}), card


def fit_p_silent(samples, features, labels, *, columns=None):
    y = silence_labels(labels)
    return _fit_event(samples, features, y, target_id=SILENT_TARGET_ID, raw_col="p_silent", columns=columns)


def fit_p_hit15(samples, features, labels, *, columns=None):
    y = hit15_labels(labels)
    return _fit_event(samples, features, y, target_id=HIT15_TARGET_ID, raw_col="p_hit15", columns=columns)


def rolling_calibrate_binary(samples, raw_pred, labels, *, raw_col, cal_col, min_cal_rows=MIN_CAL_ROWS):
    """Expanding Platt. Labels only if label_available_at < prediction_at."""
    needed = {"sample_id", "prediction_at", "label_available_at"}
    if not needed.issubset(samples.columns):
        raise ValueError("prediction_at and label_available_at required for rolling calibration")
    if raw_col not in raw_pred.columns or "segment" not in raw_pred.columns:
        raise ValueError("raw event probabilities missing")
    if "target" not in labels.columns:
        raise ValueError("boolean event labels required")
    y = labels[["sample_id", "target"]].copy()
    y["target"] = y["target"].map(bool)
    frame = samples[["sample_id", "prediction_at", "label_available_at"]].merge(
        raw_pred[["sample_id", raw_col, "segment"]], on="sample_id"
    )
    frame = frame.merge(y, on="sample_id", how="left")
    frame["_pred"] = pd.to_datetime(frame["prediction_at"], utc=True)
    frame["_lab"] = pd.to_datetime(frame["label_available_at"], utc=True)
    frame[cal_col] = np.nan
    frame["calibrator_n"] = 0
    forward = frame.segment.isin(["selection-policy", "outer-test"])
    dates = sorted(frame.loc[forward, "_pred"].dropna().unique())
    last_model = None
    last_n = 0
    for ts in dates:
        mature = frame["_lab"].lt(ts) & frame.target.notna() & frame[raw_col].notna()
        cal = frame.loc[mature]
        if len(cal) >= min_cal_rows and int(cal.target.nunique()) == 2:
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            model.fit(_logit(cal[raw_col]), cal.target.astype(int))
            last_model, last_n = model, int(len(cal))
        day = forward & frame["_pred"].eq(ts) & frame[raw_col].notna()
        if not day.any():
            continue
        if last_model is None:
            frame.loc[day, cal_col] = frame.loc[day, raw_col]
        else:
            frame.loc[day, cal_col] = last_model.predict_proba(_logit(frame.loc[day, raw_col]))[:, 1]
        frame.loc[day, "calibrator_n"] = last_n
    leftover = forward & frame[raw_col].notna() & frame[cal_col].isna()
    if leftover.any():
        frame.loc[leftover, cal_col] = frame.loc[leftover, raw_col]
    return frame.drop(columns=["target", "_pred", "_lab"])


def matured_silence_cooldown(samples, silent_table) -> pd.DataFrame:
    """Block an entity only after a *matured* silent episode. Current-window path is unused."""
    need_s = {"sample_id", "entity_id", "prediction_at"}
    need_y = {"sample_id", "silent", "label_available_at"}
    if not need_s.issubset(samples.columns):
        raise ValueError("entity_id and prediction_at required for silence cooldown")
    if not need_y.issubset(silent_table.columns):
        raise ValueError("matured silence labels required")
    pred = samples[["sample_id", "entity_id", "prediction_at"]].copy()
    hist = silent_table.merge(samples[["sample_id", "entity_id"]], on="sample_id", how="left")
    hist = hist.dropna(subset=["silent", "label_available_at", "entity_id"])
    pred["pred_at"] = pd.to_datetime(pred["prediction_at"], utc=True)
    hist["lab_at"] = pd.to_datetime(hist["label_available_at"], utc=True)
    hist = hist.rename(columns={"sample_id": "source_sample_id"})
    grouped = {
        entity: group.sort_values(["lab_at", "source_sample_id"]).reset_index(drop=True)
        for entity, group in hist.groupby("entity_id", sort=False)
    }
    blocked = []
    for row in pred.itertuples(index=False):
        group = grouped.get(row.entity_id)
        if group is None or not len(group) or pd.isna(row.pred_at):
            blocked.append(False)
            continue
        idx = int(group["lab_at"].searchsorted(row.pred_at, side="left")) - 1
        flag = False
        while idx >= 0:
            source = group.loc[idx, "source_sample_id"]
            if source != row.sample_id and group.loc[idx, "lab_at"] < row.pred_at:
                flag = bool(group.loc[idx, "silent"])
                break
            idx -= 1
        blocked.append(flag)
    out = pred[["sample_id"]].copy()
    out["silent_cooldown"] = blocked
    return out


def silence_mass_taus(dream_frame, *, col="p_silent_cal", quantiles=(0.50, 0.70, 0.85, 0.95)):
    """Extra silence τ from replay-segment score mass. No labels, no outer."""
    if col not in dream_frame.columns:
        return ()
    risk_col = "p_b10_cal" if "p_b10_cal" in dream_frame.columns else "p_down5"
    picked = select_pool(dream_frame, n_cap=PI0_N_CAP, max_risk=1.0, risk_col=risk_col)
    mass = picked.loc[picked.selected, col].astype(float).dropna()
    if mass.empty:
        return ()
    values = []
    seen = {1.0}
    for q in quantiles:
        tau = round(float(mass.quantile(float(q))), 2)
        if 0.0 < tau < 1.0 and tau not in seen:
            seen.add(tau)
            values.append(tau)
    return tuple(values)


def screen_grid(pi0=None, extra_silence_taus=()):
    if pi0 is None:
        pi0 = dict(
            n_cap=20, ranking="penalized_utility", lambda_=1.0, mu=2.0, nu=0.25,
            max_risk=1.0, risk_col="p_b10_cal",
            max_silence=1.0, silence_col="p_silent_cal", cooldown=False,
        )
    grid = [dict(pi0)]
    for item in (
        dict(pi0, cooldown=True),
        dict(pi0, ranking="specified_rise"),
        dict(pi0, ranking="specified_rise", cooldown=True),
    ):
        if item not in grid:
            grid.append(item)
    for tau in extra_silence_taus:
        for item in (
            dict(pi0, cooldown=True, max_silence=float(tau)),
            dict(pi0, ranking="specified_rise", cooldown=True, max_silence=float(tau)),
        ):
            if item not in grid:
                grid.append(item)
    return grid, pi0


def replay_screen(dream_frame, dream_labels, online_frame, online_labels, *, extra_silence_taus=None):
    mass = silence_mass_taus(dream_frame) if extra_silence_taus is None else tuple(extra_silence_taus)
    grid, pi0 = screen_grid(extra_silence_taus=mass)
    dream_scores = [evaluate_policy(dream_frame, dream_labels, p) for p in grid]
    incumbent = next(s for s in dream_scores if s["policy"] == pi0)

    def _eligible(score):
        if score["policy"] == pi0:
            return True
        if (score.get("coverage_vs_pi0") or 0) < COVERAGE_FLOOR:
            return False
        return dual_improves_specified_rise(score, incumbent)

    eligible = [s for s in dream_scores if _eligible(s)]
    champion = max(eligible, key=lambda s: (
        s.get("specified_rise_rate") is not None,
        s.get("specified_rise_rate") or -1,
        -(s.get("hold_drawdown_rate") or 1),
    ))
    online_pi0 = evaluate_policy(online_frame, online_labels, pi0)
    online_champ = evaluate_policy(online_frame, online_labels, champion["policy"])
    transferred = dual_improves_specified_rise(online_champ, online_pi0)
    rec = champion["policy"] if transferred else pi0
    return dict(
        pi0=pi0,
        champion_policy=champion["policy"],
        shipped_equals_pi0=champion["policy"] == pi0,
        online_transfer_ok=transferred,
        recommended_policy=rec,
        dream_pi0=incumbent,
        dream_champion=champion,
        dream_grid=dream_scores,
        online_grid=[evaluate_policy(online_frame, online_labels, p) for p in grid],
        online_pi0=online_pi0,
        online_champion=online_champ,
        online_recommended=evaluate_policy(online_frame, online_labels, rec),
        n_policies=len(grid),
        n_dream_improvers=len(eligible) - 1,
        replay_mass_taus=list(mass),
        coverage_floor=COVERAGE_FLOOR,
        coverage_denominator="pi0 20 x dates",
        specified_rise="+15% take-profit after T+1; not horizon grind",
        silence="max gain < 8% and not -10%; current window unused at prediction",
        formal_training_authorized=False,
        production_eligible=False,
        screen="matured silence cooldown and/or calibrated P(silent); independent P(B10)",
    )
