"""Dream-RSI style policy replay on frozen joint probabilities.

Zheng et al., Dream-RSI (arXiv:2609.14858, 2026): historical discovery traces
are a replay simulator; a challenger ships only if it does not score worse than
the incumbent π0. This module evolves *selection policy* (N, risk gate, ranking),
not model weights, Hold-to-maturity mapping (if the position is *not* taken off early):
  A: terminal net > 0 and never −5%
  B: terminal net > 0 but saw −5%
  C: terminal net ≤ 0 and never −5%
  D: terminal net ≤ 0 and saw −5%
Buy-and-hold 20d profitability = A∪B. That is not the only trading win:
take-profit inside the window is a separate evaluation (`take_profit.py`).
"""
from __future__ import annotations

from .select import utility_score, validate_utility_weights

HOLD_PROFIT = frozenset({"A", "B"})
HOLD_DRAWDOWN = frozenset({"B", "D"})
RESEARCH_TOP_N = (20, 50)


def hold_outcomes(target) -> dict:
    label = None if target is None or (isinstance(target, float) and target != target) else str(target)
    return {
        "hold_profit": label in HOLD_PROFIT,
        "hold_drawdown": label in HOLD_DRAWDOWN,
        "safe_profit": label == "A",
        "dangerous_profit": label == "B",
        "safe_loss": label == "C",
        "dangerous_loss": label == "D",
        "class": label,
    }


PI0_N_CAP = 20


def pool_metrics(selected_targets, n_cap, n_dates) -> dict:
    labels = [str(t) for t in selected_targets if t == t and t is not None]
    n = len(labels)
    dates = int(n_dates)
    coverage_vs_pi0 = float(n / (PI0_N_CAP * max(dates, 1)))
    if n == 0:
        return dict(n_selected=0, n_cap=int(n_cap), dates=dates, coverage=0.0,
                    coverage_vs_pi0=coverage_vs_pi0,
                    hold_profit_rate=None, hold_drawdown_rate=None, safe_profit_rate=None,
                    specified_rise_rate=None, class_counts={})
    counts = {k: int(sum(x == k for x in labels)) for k in "ABCD"}
    return dict(
        n_selected=n,
        n_cap=int(n_cap),
        dates=dates,
        coverage=float(n / (int(n_cap) * max(dates, 1))),
        coverage_vs_pi0=coverage_vs_pi0,
        hold_profit_rate=float(sum(x in HOLD_PROFIT for x in labels) / n),
        hold_drawdown_rate=float(sum(x in HOLD_DRAWDOWN for x in labels) / n),
        safe_profit_rate=float(sum(x == "A" for x in labels) / n),
        specified_rise_rate=None,
        class_counts=counts,
    )


def score_row(pA, pB, pC, pD, ranking, lambda_, mu, nu) -> float:
    if ranking == "penalized_utility":
        return utility_score(pA, pB, pC, pD, lambda_, mu, nu)
    if ranking == "safe_probability":
        return float(pA)
    if ranking == "hold_profit":
        return float(pA) + float(pB)
    raise ValueError("unsupported ranking")


def select_pool(frame, *, n_cap, ranking="penalized_utility", lambda_=1.0, mu=2.0, nu=0.25,
                max_risk=1.0, risk_col="p_down5", max_silence=1.0, silence_col=None,
                blocked_col=None):
    """Research ranking for N=20/50. Risk/silence columns are independent gates, not score terms."""
    if n_cap not in RESEARCH_TOP_N:
        raise ValueError("research TopN is 20 or 50")
    validate_utility_weights(lambda_, mu, nu)
    required = {"sample_id", "signal_date", "p_A", "p_B", "p_C", "p_D"}
    if not required.issubset(frame.columns):
        raise ValueError("joint probabilities and identity required")
    if risk_col not in frame.columns and risk_col != "p_down5":
        raise ValueError("independent risk column missing")
    rows = frame.copy()
    if "p_down5" not in rows.columns:
        rows["p_down5"] = rows.p_B + rows.p_D
    if risk_col not in rows.columns:
        rows[risk_col] = rows.p_down5
    if ranking == "specified_rise":
        hit_col = "p_hit15_cal" if "p_hit15_cal" in rows.columns else "p_hit15"
        if hit_col not in rows.columns:
            raise ValueError("specified-rise probability missing")
        rows["score"] = rows[hit_col].astype(float)
    else:
        rows["score"] = [
            score_row(r.p_A, r.p_B, r.p_C, r.p_D, ranking, lambda_, mu, nu)
            for r in rows.itertuples(index=False)
        ]
    rows["selected"] = False
    eligible = rows[risk_col].le(max_risk) & rows.p_A.notna()
    if silence_col:
        if silence_col not in rows.columns:
            raise ValueError("silence probability missing")
        eligible = eligible & rows[silence_col].le(max_silence)
    if blocked_col:
        if blocked_col not in rows.columns:
            raise ValueError("causal cooldown column missing")
        eligible = eligible & ~rows[blocked_col].fillna(False).astype(bool)
    for date, group in rows.loc[eligible].groupby("signal_date", sort=False):
        chosen = group.sort_values(["score", "sample_id"], ascending=[False, True]).head(n_cap)
        rows.loc[chosen.index, "selected"] = True
    return rows


def dual_improves(challenger, incumbent) -> bool:
    """Dream-RSI ship rule: never worse than π0 on hold profit *and* hold drawdown."""
    if challenger["n_selected"] == 0 or incumbent["hold_profit_rate"] is None:
        return False
    if challenger["hold_profit_rate"] is None or challenger["hold_drawdown_rate"] is None:
        return False
    profit_ok = challenger["hold_profit_rate"] >= incumbent["hold_profit_rate"]
    risk_ok = challenger["hold_drawdown_rate"] <= incumbent["hold_drawdown_rate"]
    strict = (challenger["hold_profit_rate"] > incumbent["hold_profit_rate"]
              or challenger["hold_drawdown_rate"] < incumbent["hold_drawdown_rate"])
    return bool(profit_ok and risk_ok and strict)


def dual_improves_specified_rise(challenger, incumbent) -> bool:
    """Ship rule for the specified +15% rise: never worse on rise rate *and* paired −10%."""
    if challenger["n_selected"] == 0 or incumbent.get("specified_rise_rate") is None:
        return False
    if challenger.get("specified_rise_rate") is None or challenger.get("hold_drawdown_rate") is None:
        return False
    rise_ok = challenger["specified_rise_rate"] >= incumbent["specified_rise_rate"]
    risk_ok = challenger["hold_drawdown_rate"] <= incumbent["hold_drawdown_rate"]
    strict = (challenger["specified_rise_rate"] > incumbent["specified_rise_rate"]
              or challenger["hold_drawdown_rate"] < incumbent["hold_drawdown_rate"])
    return bool(rise_ok and risk_ok and strict)


def evaluate_policy(frame, labels, policy) -> dict:
    keys = ("n_cap", "ranking", "lambda_", "mu", "nu", "max_risk")
    kwargs = {k: policy[k] for k in keys}
    for k in ("risk_col", "max_silence", "silence_col", "blocked_col"):
        if k in policy:
            kwargs[k] = policy[k]
    if policy.get("cooldown"):
        kwargs["blocked_col"] = policy.get("blocked_col", "silent_cooldown")
    picked = select_pool(frame, **kwargs)
    selected = picked.loc[picked.selected, ["sample_id", "signal_date"]].merge(labels, on="sample_id", how="left")
    metrics = pool_metrics(selected.target if "target" in selected.columns else [],
                           policy["n_cap"], picked.signal_date.nunique())
    if "hit15" in selected.columns:
        known = selected.hit15.notna()
        metrics["specified_rise_rate"] = float(selected.loc[known, "hit15"].astype(bool).mean()) if known.any() else None
        metrics["n_specified_rise"] = int(selected.hit15.fillna(False).astype(bool).sum())
    metrics["policy"] = dict(policy)
    metrics["n_universe"] = int(len(frame))
    return metrics


def default_grid(pi0=None):
    """Finite registered challengers. π0 is always included."""
    if pi0 is None:
        pi0 = dict(n_cap=20, ranking="penalized_utility", lambda_=1.0, mu=2.0, nu=0.25, max_risk=1.0)
    grid = [dict(pi0)]
    for n_cap in RESEARCH_TOP_N:
        for ranking in ("penalized_utility", "hold_profit"):
            for mu in (2.0, 3.0):
                for max_risk in (1.0, 0.60):
                    item = dict(n_cap=n_cap, ranking=ranking, lambda_=1.0, mu=mu, nu=0.25, max_risk=max_risk)
                    if item not in grid:
                        grid.append(item)
    return grid, pi0


def replay(dream_frame, dream_labels, online_frame, online_labels, *, grid=None, pi0=None):
    """Refine policy on the dream/replay segment; freeze before online scoring."""
    grid, pi0 = default_grid(pi0) if grid is None else (list(grid), pi0 or grid[0])
    if pi0 not in grid:
        grid = [dict(pi0), *grid]
    dream_scores = [evaluate_policy(dream_frame, dream_labels, p) for p in grid]
    incumbent = next(s for s in dream_scores if s["policy"] == pi0)
    eligible = [s for s in dream_scores if s["policy"] == pi0 or dual_improves(s, incumbent)]
    champion = max(eligible, key=lambda s: (s["hold_profit_rate"] is not None,
                                            s["hold_profit_rate"] or -1,
                                            -(s["hold_drawdown_rate"] or 1),
                                            s["n_cap"]))
    online_pi0 = evaluate_policy(online_frame, online_labels, pi0)
    online_champ = evaluate_policy(online_frame, online_labels, champion["policy"])
    transferred = dual_improves(online_champ, online_pi0)
    return dict(
        pi0=pi0,
        champion_policy=champion["policy"],
        shipped_equals_pi0=champion["policy"] == pi0,
        online_transfer_ok=transferred,
        recommended_policy=champion["policy"] if transferred else pi0,
        dream_pi0=incumbent,
        dream_champion=champion,
        online_pi0=online_pi0,
        online_champion=online_champ,
        n_policies=len(grid),
        n_dream_improvers=len(eligible) - 1,
        formal_training_authorized=False,
        production_eligible=False,
        source="Dream-RSI replay of selection policy; model weights frozen",
    )
