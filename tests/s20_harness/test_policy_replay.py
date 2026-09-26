import pandas as pd
import pytest

from research.s20_harness.policy_replay import (
    dual_improves,
    hold_outcomes,
    pool_metrics,
    replay,
    select_pool,
)


def test_hold_mapping_separates_profit_from_touch_and_from_path_risk():
    assert hold_outcomes("A")["hold_profit"] and not hold_outcomes("A")["hold_drawdown"]
    assert hold_outcomes("B")["hold_profit"] and hold_outcomes("B")["hold_drawdown"]
    assert not hold_outcomes("C")["hold_profit"] and not hold_outcomes("C")["hold_drawdown"]
    assert not hold_outcomes("D")["hold_profit"] and hold_outcomes("D")["hold_drawdown"]
    empty = pool_metrics([], 50, 2)
    assert empty["hold_profit_rate"] is None and empty["coverage"] == 0.0


def test_top50_is_a_cap_and_risk_gate_can_leave_vacancies():
    frame = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(6)],
        "signal_date": ["20250102"] * 6,
        "p_A": [0.7, 0.6, 0.1, 0.2, 0.15, 0.05],
        "p_B": [0.1, 0.05, 0.4, 0.1, 0.1, 0.1],
        "p_C": [0.15, 0.3, 0.1, 0.2, 0.25, 0.2],
        "p_D": [0.05, 0.05, 0.4, 0.5, 0.5, 0.65],
    })
    wide = select_pool(frame, n_cap=50, max_risk=1.0)
    assert int(wide.selected.sum()) == 6
    gated = select_pool(frame, n_cap=50, max_risk=0.2)
    assert set(gated.loc[gated.selected, "sample_id"]) == {"s0", "s1"}


def test_dream_rsi_keeps_pi0_unless_dual_hold_improvement():
    pi0 = dict(n_cap=20, ranking="penalized_utility", lambda_=1.0, mu=2.0, nu=0.25, max_risk=1.0)
    worse = dict(n_selected=10, hold_profit_rate=0.4, hold_drawdown_rate=0.5)
    better = dict(n_selected=10, hold_profit_rate=0.6, hold_drawdown_rate=0.3)
    incumbent = dict(n_selected=10, hold_profit_rate=0.5, hold_drawdown_rate=0.4)
    assert dual_improves(better, incumbent)
    assert not dual_improves(worse, incumbent)
    assert not dual_improves(dict(n_selected=0, hold_profit_rate=None, hold_drawdown_rate=None), incumbent)

    rows = []
    labels = []
    for i, (date, cls, pA, pD) in enumerate([
        ("d1", "A", 0.7, 0.05), ("d1", "D", 0.2, 0.6),
        ("d2", "B", 0.55, 0.2), ("d2", "C", 0.3, 0.1),
    ]):
        rows.append(dict(sample_id=f"x{i}", signal_date=date, p_A=pA, p_B=0.1, p_C=0.15, p_D=pD))
        labels.append(dict(sample_id=f"x{i}", target=cls))
    frame = pd.DataFrame(rows)
    lab = pd.DataFrame(labels)
    out = replay(frame, lab, frame, lab, grid=[pi0], pi0=pi0)
    assert out["shipped_equals_pi0"]
    assert out["champion_policy"] == pi0
    assert out["recommended_policy"] == pi0
