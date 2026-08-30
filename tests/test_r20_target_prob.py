import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stockagent_analysis.r20_target_prob import (
    DEFAULT_FOLDS,
    build_target_frame,
    ordered_probabilities,
    probability_metrics,
    tier_probabilities,
)


def test_experiment_contract_cannot_replace_production_v1():
    root = Path(__file__).parents[1]
    contract = json.loads(
        (root / "config/r20_target_prob_v2_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["status"] == "research_only"
    assert contract["production_impact"] == "none"
    assert contract["production_contract"] == "config/pool_a_r20_target_v1.json"
    assert contract["industry_cap"] is None
    assert contract["top_n"] is None


def test_walk_forward_segments_are_strictly_chronological():
    for fold in DEFAULT_FOLDS:
        assert fold.fit_end < fold.tune_start <= fold.tune_end
        assert fold.tune_end < fold.calibration_start <= fold.calibration_end
        assert fold.calibration_end < fold.test_start <= fold.test_end


def _labels():
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 5,
            "trade_date": [f"2025010{i}" for i in range(1, 6)],
            "entry_open": [10.0] * 5,
            "max_gain_20": [14.9, 15.0, 20.0, 25.0, 40.0],
            "max_dd_20": [-2.0, -15.0, -14.0, -15.1, -10.0],
            "r20_close": [5.0, 10.0, 18.0, 20.0, 30.0],
        }
    )


def test_safe_targets_are_nested_and_respect_drawdown_boundary():
    result = build_target_frame(_labels())

    assert result["target_p15_safe"].tolist() == [0, 1, 1, 0, 1]
    assert result["target_p20_safe"].tolist() == [0, 0, 1, 0, 1]
    assert result["target_p25_safe"].tolist() == [0, 0, 0, 0, 1]
    assert result["target_tier"].tolist() == [0, 1, 2, 0, 3]
    assert result["target_close25_safe"].tolist() == [0, 0, 0, 0, 1]


def test_impossible_close_above_max_gain_is_rejected():
    labels = _labels()
    labels.loc[0, "r20_close"] = 20.0
    with pytest.raises(ValueError, match="r20_close"):
        build_target_frame(labels)


def test_tier_probabilities_are_nested():
    class_prob = np.array([[0.10, 0.20, 0.30, 0.40]])
    result = tier_probabilities(class_prob)
    np.testing.assert_allclose(result, [[0.90, 0.70, 0.40]])


def test_ordered_projection_is_conservative():
    result = ordered_probabilities(np.array([[0.60, 0.80, 0.50]]))
    np.testing.assert_allclose(result, [[0.60, 0.60, 0.50]])


def test_probability_metrics_reward_an_informative_forecast():
    result = probability_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["brier_skill_vs_constant"] > 0
    assert result["roc_auc"] == 1.0
    assert result["pr_auc"] == 1.0
