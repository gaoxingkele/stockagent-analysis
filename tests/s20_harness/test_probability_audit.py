import numpy as np

from research.s20_harness.probability_audit import (
    audit_saved_v3,
    reliability_table,
    static_calibrator_base_rate_trap,
)
from pathlib import Path


def test_static_calibrator_stays_near_cal_base_rate_after_shift():
    rng = np.random.default_rng(20260916)
    n = 8000
    y_cal = rng.binomial(1, 0.20, n)
    y_test = rng.binomial(1, 0.55, n)
    p_raw_cal = np.clip(0.20 + 0.04 * rng.normal(size=n), 0.02, 0.98)
    p_raw_test = np.clip(0.20 + 0.04 * rng.normal(size=n), 0.02, 0.98)
    out = static_calibrator_base_rate_trap(p_raw_cal, y_cal, p_raw_test, y_test, seed=0)
    assert out["cal_actual"] < 0.25
    assert out["test_actual"] > 0.50
    assert out["test_mean_predicted"] < 0.30
    assert out["test_gap"] < -0.20


def test_reliability_gap_grows_when_p_is_optimistic():
    y = np.array([0, 0, 0, 1, 0, 0, 1, 1], dtype=float)
    p = np.array([0.1, 0.15, 0.2, 0.25, 0.7, 0.8, 0.85, 0.9])
    table = reliability_table(y, p, bins=2)
    high = table.loc[table.p_lo >= 0.5].iloc[0]
    assert high.mean_predicted > high.actual_rate
    assert high.gap > 0.2


def test_saved_v3_top20_probabilities_are_not_hit_rates():
    root = Path(__file__).resolve().parents[2]
    pred = root / "output/experiments/s20_v3/predictions.parquet"
    if not pred.exists():
        return
    report = audit_saved_v3(root)
    uni = report["diagnostic_prediction_universe"]
    half = report["selection_immediate_half_risk"]
    down = report["selection_down_half_risk"]
    assert uni["mean_p_immediate"] - uni["actual_immediate"] > 0.15
    assert uni["actual_down"] - uni["mean_p_down"] > 0.20
    assert abs(half["topn_mean_p"] - 0.672) < 0.02
    assert abs(half["topn_actual"] - 0.321) < 0.02
    assert down["topn_actual"] > 0.55
    assert down["topn_mean_p"] < 0.25
