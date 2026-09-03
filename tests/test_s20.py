import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stockagent_analysis.s20 import (
    S20_V2_TARGET_DRAWDOWN,
    build_daily_first_passage_labels,
    build_daily_s20_v2_labels,
    build_first_passage_labels,
    build_s20_v2_path_labels,
    cumulative_incidence,
    daily_topk_metrics,
    purged_walk_forward_masks,
    s20_score,
)


def test_s20_v2_contract_freezes_five_target_risk_budgets():
    root = Path(__file__).parents[1]
    contract = json.loads(
        (root / "config/s20_v2_training_contract.json").read_text(encoding="utf-8")
    )
    assert contract["status"] == "frozen_research_only"
    assert contract["production_impact"] == "none"
    assert contract["target_drawdown_pct"] == {
        "15": -8.0,
        "20": -10.0,
        "25": -12.0,
        "30": -12.0,
        "35": -12.0,
    }
    assert contract["industry_cap"] is None


def test_s20_contract_is_independent_and_research_only():
    root = Path(__file__).parents[1]
    contract = json.loads(
        (root / "config/s20_experiment_v1.json").read_text(encoding="utf-8")
    )
    assert contract["status"] == "research_only"
    assert contract["production_impact"] == "none"
    assert contract["industry_cap"] is None
    assert "never replaces" in contract["relationship_to_r20"]


def test_first_passage_labels_preserve_event_order_and_ambiguity():
    entry = [100, 100, 100, 100]
    highs = np.array(
        [
            [126, 127, 128],  # upside first
            [105, 126, 127],  # downside first
            [105, 126, 127],  # both barriers on day 2: unknown intraday order
            [105, 110, 114],  # neither barrier
        ]
    )
    lows = np.array(
        [
            [99, 84, 90],
            [84, 90, 92],
            [99, 84, 90],
            [99, 96, 91],
        ]
    )

    result = build_first_passage_labels(entry, highs, lows)

    assert result["event25"].tolist() == [
        "up_first",
        "down_first",
        "ambiguous",
        "censored",
    ]
    assert result["target25_up_first"].tolist() == [1, 0, 0, 0]
    assert result["down_day"].tolist() == [2, 1, 2, 0]


def test_s20_score_is_a_probability_scale_not_a_rank_score():
    np.testing.assert_allclose(s20_score([0.0, 0.375, 1.0]), [0.0, 37.5, 100.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        s20_score([1.01])


def test_daily_label_timing_uses_next_open_and_twenty_session_path():
    rows = 22
    daily = {
        "ts_code": ["000001.SZ"] * rows,
        "trade_date": [f"202501{i:02d}" for i in range(1, rows + 1)],
        "open": [100.0] * rows,
        "high": [101.0] * rows,
        "low": [99.0] * rows,
        "close": [100.0] * rows,
    }
    daily["open"][1] = 80.0
    daily["high"][1] = 100.0  # +25% on entry day for signal row zero.
    daily["low"][2] = 67.0  # downside barrier is reached one day later.

    result = build_daily_first_passage_labels(pd.DataFrame(daily))

    assert len(result) == 2
    assert result.loc[0, "entry_date"] == "20250102"
    assert result.loc[0, "horizon_end_date"] == "20250121"
    assert result.loc[0, "entry_open"] == 80.0
    assert result.loc[0, "event25"] == "up_first"
    assert result.loc[0, "late_down_after_up25"] == 1
    assert result.loc[0, "target25_window_safe"] == 0


def test_competing_risk_probabilities_combine_with_survival_weighting():
    result = cumulative_incidence(
        [
            np.array([[0.8, 0.1, 0.1]]),
            np.array([[0.5, 0.3, 0.2]]),
        ]
    )
    np.testing.assert_allclose(result["upside"], [0.34])
    np.testing.assert_allclose(result["downside"], [0.26])
    np.testing.assert_allclose(result["survival"], [0.40])


def test_daily_topk_measures_stock_selection_instead_of_global_deciles():
    frame = pd.DataFrame(
        {
            "trade_date": ["20250101"] * 3 + ["20250102"] * 3,
            "probability": [0.9, 0.2, 0.1, 0.8, 0.7, 0.1],
            "target": [1, 0, 0, 0, 1, 0],
        }
    )
    result = daily_topk_metrics(
        frame, probability_col="probability", target_col="target", k=1
    )
    assert result["dates"] == 2
    assert result["selected_rows"] == 2
    assert result["precision"] == 0.5
    assert result["lift"] == 1.5


def test_s20_v2_labels_partition_positive_and_three_negative_reasons():
    entry = [100.0] * 7
    highs = np.array(
        [
            [105, 116, 117, 118],  # positive after controlled pullback
            [116, 117, 118, 119],  # positive: target immediately, then holds entry
            [105, 110, 112, 114],  # N1: quiet miss
            [105, 106, 116, 117],  # N2: stop before target
            [116, 118, 110, 105],  # N3: target then below entry
            [116, 117, 118, 119],  # ambiguous on target day
            [105, 116, 117, 118],  # N2 remains known after an earlier stop
        ],
        dtype=float,
    )
    lows = np.array(
        [
            [95, 101, 102, 103],
            [100, 101, 102, 103],
            [98, 97, 96, 95],
            [91, 93, 101, 102],
            [101, 102, 99, 98],
            [99, 101, 102, 103],
            [91, 99, 101, 102],
        ],
        dtype=float,
    )
    result = build_s20_v2_path_labels(
        entry, highs, lows, target_drawdown={15.0: -8.0}
    )
    assert result["reason15"].tolist() == [
        "positive",
        "positive",
        "n1_miss_no_stop",
        "n2_stop_before_target",
        "n3_post_target_below_entry",
        "ambiguous_target_day",
        "n2_stop_before_target",
    ]
    assert result["class15"].tolist() == [0, 0, 1, 2, 3, -1, 2]
    assert result["positive15"].tolist() == [1, 1, 0, 0, 0, -1, 0]


def test_s20_v2_daily_builder_uses_next_open_and_all_five_targets():
    rows = 22
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * rows,
            "trade_date": [f"202502{i:02d}" for i in range(1, rows + 1)],
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [100.0] * rows,
            "close": [100.0] * rows,
        }
    )
    daily.loc[1, "open"] = 80.0
    daily.loc[1:20, "low"] = 80.0
    daily.loc[2, "high"] = 108.0  # +35% from the D+1 entry.
    result = build_daily_s20_v2_labels(daily)
    assert len(result) == 2
    assert result.loc[0, "entry_open"] == 80.0
    assert result.loc[0, "entry_date"] == "20250202"
    assert result.loc[0, "horizon_end_date"] == "20250221"
    assert tuple(S20_V2_TARGET_DRAWDOWN) == (15.0, 20.0, 25.0, 30.0, 35.0)
    for target in (15, 20, 25, 30, 35):
        assert result.loc[0, f"positive{target}"] == 1


def test_purged_masks_require_labels_to_mature_before_next_segment():
    class Fold:
        fit_end = "20250103"
        tune_start = "20250104"
        tune_end = "20250106"
        calibration_start = "20250107"
        calibration_end = "20250108"
        test_start = "20250109"
        test_end = "20250110"

    dates = pd.Series(
        ["20250101", "20250102", "20250104", "20250105", "20250107", "20250109"]
    )
    ends = pd.Series(
        ["20250103", "20250104", "20250106", "20250107", "20250108", "20250110"]
    )
    masks = purged_walk_forward_masks(dates, ends, Fold())
    assert masks["fit"].tolist() == [True, False, False, False, False, False]
    assert masks["tune"].tolist() == [False, False, True, False, False, False]
    assert masks["calibration"].tolist() == [False, False, False, False, True, False]
    assert masks["test"].tolist() == [False, False, False, False, False, True]
