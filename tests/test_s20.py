import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from stockagent_analysis.s20 import (
    build_daily_first_passage_labels,
    build_first_passage_labels,
    cumulative_incidence,
    s20_score,
)


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
