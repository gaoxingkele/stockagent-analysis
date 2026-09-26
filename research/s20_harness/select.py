"""TopN utility ranking with an independent risk gate and nullable empty metrics."""
from __future__ import annotations

import math

import pandas as pd

from research.s20_harness.contracts import ALLOWED_TOP_N


def validate_utility_weights(lambda_, mu, nu) -> tuple[float, float, float]:
    lam, mu_v, nu_v = float(lambda_), float(mu), float(nu)
    if not (mu_v > lam > nu_v >= 0):
        raise ValueError("require mu > lambda > nu >= 0")
    return lam, mu_v, nu_v


def utility_score(pA, pB, pC, pD, lambda_=1.0, mu=2.0, nu=0.25) -> float:
    lam, mu_v, nu_v = validate_utility_weights(lambda_, mu, nu)
    return float(pA) - lam * float(pB) - mu_v * float(pD) - nu_v * float(pC)


def joint_from_four_class(pA, pB, pC, pD) -> dict:
    """Joint events from a four-class simplex. Never p_up * p_down under independence."""
    vals = [float(pA), float(pB), float(pC), float(pD)]
    if min(vals) < -1e-12 or abs(sum(vals) - 1.0) > 1e-8:
        raise ValueError("pA,pB,pC,pD must be a probability simplex")
    p_up = vals[0] + vals[1]
    p_down5 = vals[1] + vals[3]
    p_down_given_up = vals[1] / p_up if p_up > 0 else float("nan")
    independence_product = p_up * p_down5
    return {
        "pA": vals[0],
        "pB": vals[1],
        "pC": vals[2],
        "pD": vals[3],
        "p_up": p_up,
        "p_down5": p_down5,
        "p_safe_profit": vals[0],
        "p_down_given_up": p_down_given_up,
        "independence_product": independence_product,
        "used_independence_product": False,
    }


def empty_selection_metrics(n_cap: int) -> dict:
    if n_cap not in ALLOWED_TOP_N:
        raise ValueError(f"N must be one of {ALLOWED_TOP_N}")
    return {
        "n_selected": 0,
        "n_cap": int(n_cap),
        "coverage": 0.0,
        "precision": None,
        "risk": None,
        "empty": True,
    }


def _metrics(selected: pd.DataFrame, n_cap: int) -> dict:
    if selected.empty:
        return empty_selection_metrics(n_cap)
    precision = risk = None
    if "p_class" in selected.columns:
        realized = selected["p_class"].dropna()
        if len(realized):
            precision = float((realized == "A").mean())
            risk = float(realized.isin(["B", "D"]).mean())
    return {
        "n_selected": int(len(selected)),
        "n_cap": int(n_cap),
        "coverage": float(len(selected) / n_cap),
        "precision": precision,
        "risk": risk,
        "empty": False,
    }


def select_topn(
    frame: pd.DataFrame,
    n: int,
    *,
    lambda_=1.0,
    mu=2.0,
    nu=0.25,
    max_p_down5: float | None = None,
    min_p_up: float | None = None,
    min_utility: float | None = None,
) -> dict:
    """Rank by utility; apply an independent risk gate; N is a cap, vacancies allowed."""
    if n not in ALLOWED_TOP_N:
        raise ValueError(f"N must be one of {ALLOWED_TOP_N}")
    lam, mu_v, nu_v = validate_utility_weights(lambda_, mu, nu)
    rows = frame.copy()
    for col in ("pA", "pB", "pC", "pD"):
        if col not in rows.columns:
            raise ValueError(f"missing {col}")
    joints = [joint_from_four_class(r.pA, r.pB, r.pC, r.pD) for r in rows.itertuples(index=False)]
    rows["p_up"] = [j["p_up"] for j in joints]
    rows["p_down5"] = [j["p_down5"] for j in joints]
    rows["p_down_given_up"] = [j["p_down_given_up"] for j in joints]
    rows["used_independence_product"] = False
    rows["utility"] = [
        utility_score(r.pA, r.pB, r.pC, r.pD, lam, mu_v, nu_v) for r in rows.itertuples(index=False)
    ]
    reasons = []
    eligible = []
    for i, row in rows.iterrows():
        reason = None
        if max_p_down5 is not None and row.p_down5 > max_p_down5:
            reason = "risk_high"
        elif min_p_up is not None and row.p_up < min_p_up:
            reason = "upside_low"
        elif min_utility is not None and row.utility < min_utility:
            reason = "upside_low"
        elif any(math.isnan(float(row[c])) for c in ("pA", "pB", "pC", "pD")):
            reason = "calibration_unreliable"
        reasons.append(reason)
        eligible.append(reason is None)
    rows["reject_reason"] = reasons
    rows["eligible"] = eligible
    ranked = rows.loc[rows["eligible"]].sort_values(
        ["utility", "ts_code"] if "ts_code" in rows.columns else ["utility"],
        ascending=[False, True] if "ts_code" in rows.columns else [False],
    )
    picked = ranked.head(n).copy()
    picked["selected"] = True
    rejected = rows.loc[~rows["eligible"]].copy()
    rejected["selected"] = False
    leftover = ranked.iloc[len(picked):].copy()
    leftover["selected"] = False
    leftover["reject_reason"] = leftover["reject_reason"].fillna("not_in_topn")
    metrics = _metrics(picked, n)
    return {
        "selected": picked.reset_index(drop=True),
        "rejected": pd.concat([rejected, leftover], ignore_index=True),
        "universe": rows.reset_index(drop=True),
        "metrics": metrics,
        "n_cap": n,
        "used_independence_product": False,
    }
