import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.v12_scoring import V12Scorer


def _scorer_without_models():
    return object.__new__(V12Scorer)


def test_incomplete_high_scores_do_not_consume_top_percentile_slots():
    valid_n = 100
    missing_n = 20
    df = pd.DataFrame({
        "ts_code": ([f"V{i:05d}.SZ" for i in range(valid_n)]
                    + [f"M{i:05d}.BJ" for i in range(missing_n)]),
        "r20_pred": list(range(valid_n)) + list(range(1000, 1000 + missing_n)),
        "pyr_velocity_20_60": (
            [0.0] * (valid_n - 6) + [-1.0] * 6 + [np.nan] * missing_n
        ),
        "f1_neg1": [0.0] * valid_n + [np.nan] * missing_n,
        "f2_pos1": [0.0] * valid_n + [np.nan] * missing_n,
        "is_zombie": [False] * (valid_n + missing_n),
        "industry_mom_60d_rank": [0.5] * (valid_n + missing_n),
    })
    scorer = _scorer_without_models()

    eligible = scorer._v7c_eligibility_mask(df)
    selected = scorer._apply_v7c_rules(df)

    assert int(eligible.sum()) == valid_n
    assert not selected[df["ts_code"].str.endswith(".BJ")].any()
    assert int(selected.sum()) == 6
    assert set(df.loc[selected, "ts_code"]) == {
        "V00094.SZ", "V00095.SZ", "V00096.SZ",
        "V00097.SZ", "V00098.SZ", "V00099.SZ",
    }


def test_non_finite_hard_rule_features_are_ineligible():
    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
        "r20_pred": [1.0, 2.0, 3.0, np.inf],
        "pyr_velocity_20_60": [0.0, np.nan, np.inf, 0.0],
        "f1_neg1": [0.0, 0.0, 0.0, 0.0],
        "f2_pos1": [0.0, 0.0, 0.0, 0.0],
    })

    eligible = _scorer_without_models()._v7c_eligibility_mask(df)

    assert eligible.tolist() == [True, False, False, False]


def test_complete_bj_stock_is_reserved_for_pool_g():
    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "920001.BJ"],
        "r20_pred": [1.0, 10.0],
        "pyr_velocity_20_60": [0.0, 0.0],
        "f1_neg1": [0.0, 0.0],
        "f2_pos1": [0.0, 0.0],
    })

    eligible = _scorer_without_models()._v7c_eligibility_mask(df)

    assert eligible.tolist() == [True, False]
