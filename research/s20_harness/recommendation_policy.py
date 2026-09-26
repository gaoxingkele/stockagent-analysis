"""Frozen daily score policy; no outcome access or implicit order placement."""
from __future__ import annotations

import hashlib
import json
import math

import pandas as pd

from .contracts import ALLOWED_TOP_N
from .label_availability import _instant


def apply(candidates, policy, calendar, *, controls=None):
    """Reference selection over all ex-ante candidates and all calendar days.

    score_only_control is explicitly a comparator without risk protection.
    risk_gated requires a separately supplied downside probability: safe-event
    complement is not silently interpreted as probability of a large decline.
    Caller must bind model/calibration artifacts before formal stage acceptance.
    """
    fields = {"policy_id", "target_id", "risk_target_id", "mode", "frozen_at",
              "n_cap", "min_score", "max_risk"}
    simple_control = policy.get("mode") == "atr_liquidity_control"
    if simple_control:
        fields |= {"max_atr_fraction", "min_traded_value_cny"}
        for key in ("max_atr_fraction", "min_traded_value_cny"):
            value = policy.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("finite nonnegative ATR/liquidity thresholds required")
    elif controls is not None:
        raise ValueError("control inputs require explicit ATR/liquidity policy")
    if set(policy) != fields:
        raise ValueError("exact policy schema required")
    for key in ("policy_id", "target_id"):
        if not isinstance(policy[key], str) or not policy[key].strip():
            raise ValueError("named policy and target required")
    if type(policy["n_cap"]) is not int or policy["n_cap"] not in ALLOWED_TOP_N:
        raise ValueError("invalid TopN cap")
    for key in ("min_score", "max_risk"):
        value = policy[key]
        if key == "max_risk" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("finite unit-interval policy thresholds required")
    if policy["mode"] in {"score_only_control", "atr_liquidity_control"}:
        if policy["max_risk"] is not None or policy["risk_target_id"] is not None:
            raise ValueError("score-only control cannot claim risk gate")
    elif policy["mode"] == "risk_gated":
        if (policy["max_risk"] is None or not isinstance(policy["risk_target_id"], str)
                or not policy["risk_target_id"].strip()):
            raise ValueError("separate downside target and threshold required")
    else:
        raise ValueError("unknown policy mode")
    frozen = _instant(policy["frozen_at"])
    columns = {"sample_id", "entity_id", "signal_date", "prediction_at", "score", "risk"}
    if candidates.columns.duplicated().any() or set(candidates.columns) != columns:
        raise ValueError("exact candidate schema without outcomes required")
    if candidates[["sample_id", "entity_id", "signal_date", "prediction_at"]].isna().any().any():
        raise ValueError("missing candidate identity or time")
    if candidates.sample_id.duplicated().any() or candidates.duplicated(["signal_date", "entity_id"]).any():
        raise ValueError("duplicate candidate or stock-day")
    if any(not isinstance(v, str) or not v.strip() for v in candidates.sample_id):
        raise ValueError("string sample identities required")
    if not calendar or list(calendar) != sorted(set(calendar)):
        raise ValueError("nonempty unique chronological calendar required")
    for date in calendar:
        if not isinstance(date, str) or len(date) != 8:
            raise ValueError("YYYYMMDD calendar required")
        pd.to_datetime(date, format="%Y%m%d", errors="raise")
    if not set(candidates.signal_date).issubset(calendar):
        raise ValueError("candidate outside calendar")
    for row in candidates.itertuples(index=False):
        at = _instant(row.prediction_at)
        if at <= frozen:
            raise ValueError("policy must be frozen strictly before prediction")
        if at.tz_convert("Asia/Shanghai").strftime("%Y%m%d") != row.signal_date:
            raise ValueError("signal date and prediction time mismatch")
    for column in ("score", "risk"):
        if not pd.api.types.is_numeric_dtype(candidates[column]) or pd.api.types.is_bool_dtype(candidates[column]):
            raise ValueError("numeric score and risk required")
        known = candidates[column].dropna()
        if not known.between(0, 1).all():
            raise ValueError("score or risk outside unit interval")
    rows = candidates.copy().reset_index(drop=True)
    rows["selected"] = False
    rows["reject_reason"] = "not_in_topn"
    eligible = rows.score.notna() & rows.score.ge(policy["min_score"])
    rows.loc[rows.score.lt(policy["min_score"]), "reject_reason"] = "score_below_threshold"
    rows.loc[rows.score.isna(), "reject_reason"] = "score_unavailable"
    if policy["mode"] == "risk_gated":
        eligible &= rows.risk.notna() & rows.risk.le(policy["max_risk"])
        rows.loc[rows.risk.gt(policy["max_risk"]), "reject_reason"] = "risk_high"
        rows.loc[rows.risk.isna(), "reject_reason"] = "risk_unavailable"
    if simple_control:
        required = {"sample_id", "available_at", "atr_fraction", "traded_value_cny"}
        if (not isinstance(controls, pd.DataFrame) or controls.columns.duplicated().any()
                or set(controls.columns) != required or controls.sample_id.isna().any()
                or controls.sample_id.duplicated().any()
                or set(controls.sample_id) != set(rows.sample_id)):
            raise ValueError("exact candidate-aligned ATR/liquidity control schema required")
        aligned = controls.set_index("sample_id").loc[rows.sample_id].reset_index(drop=True)
        for column in ("atr_fraction", "traded_value_cny"):
            values = aligned[column]
            if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
                    or any(not math.isfinite(v) or v < 0 for v in values.dropna())):
                raise ValueError("nonnegative numeric ATR/liquidity values required")
        for i, row in aligned.iterrows():
            if pd.isna(row.available_at):
                if pd.notna(row.atr_fraction) or pd.notna(row.traded_value_cny):
                    raise ValueError("known controls require availability time")
            elif _instant(row.available_at) > _instant(rows.loc[i, "prediction_at"]):
                raise ValueError("future ATR/liquidity control unavailable at prediction")
        ready = aligned.atr_fraction.notna() & aligned.traded_value_cny.notna()
        passed = ready & aligned.atr_fraction.le(policy["max_atr_fraction"]) & aligned.traded_value_cny.ge(policy["min_traded_value_cny"])
        eligible &= passed
        rows.loc[~passed & ready, "reject_reason"] = "atr_liquidity_filtered"
        rows.loc[~ready, "reject_reason"] = "atr_liquidity_unavailable"
        rows["control_available_at"] = aligned.available_at
        rows["atr_fraction"] = aligned.atr_fraction
        rows["traded_value_cny"] = aligned.traded_value_cny
        rows["control_passed"] = passed
    daily = []
    for date in calendar:
        available = rows.loc[rows.signal_date.eq(date) & eligible]
        chosen = available.sort_values(["score", "sample_id"], ascending=[False, True]).head(policy["n_cap"])
        rows.loc[chosen.index, "selected"] = True
        rows.loc[chosen.index, "reject_reason"] = None
        daily.append({"signal_date": date, "candidates": int(rows.signal_date.eq(date).sum()),
                      "eligible": len(available), "selected": len(chosen),
                      "vacancies": policy["n_cap"] - len(chosen), "precision": None, "risk_rate": None})
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    rows["policy_sha256"] = policy_hash
    report = {"policy": policy, "policy_sha256": policy_hash, "daily": daily,
                  "rows": len(rows), "selected": int(rows.selected.sum()),
                  "active_day_coverage": sum(d["selected"] > 0 for d in daily) / len(daily),
                  "all_candidates_retained": True, "outcomes_consumed": False,
                  "risk_gate_applied": policy["mode"] == "risk_gated",
                  "model_lineage_verified": False, "formal_H05_accepted": False,
                  "orders_created": False, "production_eligible": False}
    if simple_control:
        report.update(simple_control_applied=True, control_model_fits=0,
                      control_availability_independently_verified=False,
                      filtered_score_is_new_probability=False)
    return rows, report
