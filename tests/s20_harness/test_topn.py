import math

import pandas as pd
import pytest

from research.s20_harness.select import (
    empty_selection_metrics,
    joint_from_four_class,
    select_topn,
    utility_score,
    validate_utility_weights,
)


def test_dual_high_and_dangerous_loss_rank_below_safe_profit():
    frame = pd.DataFrame([
        {"ts_code": "SAFE", "pA": 0.70, "pB": 0.00, "pC": 0.30, "pD": 0.00},
        {"ts_code": "DUAL", "pA": 0.00, "pB": 0.70, "pC": 0.30, "pD": 0.00},
        {"ts_code": "LOSS", "pA": 0.00, "pB": 0.00, "pC": 0.30, "pD": 0.70},
    ])
    out = select_topn(frame, 3, lambda_=1.0, mu=2.0, nu=0.25)
    order = out["selected"]["ts_code"].tolist()
    assert order == ["SAFE", "DUAL", "LOSS"]
    assert out["used_independence_product"] is False
    assert utility_score(0.7, 0, 0.3, 0) > utility_score(0, 0.7, 0.3, 0)
    assert utility_score(0, 0.7, 0.3, 0) > utility_score(0, 0, 0.3, 0.7)


def test_risk_gate_rejects_high_upside_name_with_reason():
    frame = pd.DataFrame([
        {"ts_code": "HOT", "pA": 0.80, "pB": 0.15, "pC": 0.00, "pD": 0.05},
        {"ts_code": "OK", "pA": 0.40, "pB": 0.02, "pC": 0.56, "pD": 0.02},
    ])
    assert utility_score(0.80, 0.15, 0.00, 0.05) > utility_score(0.40, 0.02, 0.56, 0.02)
    out = select_topn(frame, 3, max_p_down5=0.10)
    assert out["selected"]["ts_code"].tolist() == ["OK"]
    rejected = out["rejected"].set_index("ts_code")
    assert rejected.loc["HOT", "reject_reason"] == "risk_high"
    assert len(out["selected"]) == 1


def test_n_cap_allows_vacancy_and_empty_is_null_not_perfect():
    frame = pd.DataFrame([
        {"ts_code": "ONLY", "pA": 0.60, "pB": 0.05, "pC": 0.30, "pD": 0.05},
    ])
    one = select_topn(frame, 3)
    assert len(one["selected"]) == 1
    assert one["metrics"]["n_cap"] == 3
    blocked = select_topn(frame, 3, max_p_down5=0.01)
    assert blocked["selected"].empty
    metrics = blocked["metrics"]
    assert metrics == empty_selection_metrics(3)
    assert metrics["precision"] is None
    assert metrics["risk"] is None
    assert metrics["coverage"] == 0.0
    assert metrics["n_selected"] == 0


def test_joint_down_given_up_is_not_independence_product():
    joint = joint_from_four_class(0.10, 0.40, 0.40, 0.10)
    assert joint["p_up"] == pytest.approx(0.50)
    assert joint["p_down5"] == pytest.approx(0.50)
    assert joint["p_down_given_up"] == pytest.approx(0.80)
    assert joint["independence_product"] == pytest.approx(0.25)
    assert joint["p_down_given_up"] != joint["independence_product"]
    assert joint["used_independence_product"] is False
    frame = pd.DataFrame([{"ts_code": "X", "pA": 0.10, "pB": 0.40, "pC": 0.40, "pD": 0.10}])
    out = select_topn(frame, 1)
    value = float(out["selected"]["p_down_given_up"].iloc[0])
    assert value == pytest.approx(0.80)
    assert not math.isclose(value, 0.25)
    assert out["used_independence_product"] is False


def test_weight_constraint_mu_gt_lambda_gt_nu():
    with pytest.raises(ValueError):
        validate_utility_weights(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        select_topn(pd.DataFrame([{"ts_code": "Z", "pA": 1, "pB": 0, "pC": 0, "pD": 0}]), 1, lambda_=1, mu=0.5, nu=0)
