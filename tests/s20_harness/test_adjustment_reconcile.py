import numpy as np
import pandas as pd

from research.s20_harness.adjustment_reconcile import reconcile_panel


def test_split_like_factor_explains_price_change_not_profit():
    panel = pd.DataFrame({"ts_code": ["000001.SZ"] * 3, "trade_date": ["20240102", "20240103", "20240104"],
                          "close": [10., 5., 4.], "pre_close": [10., 5., 4.], "adj_factor": [1., 2., 2.]})
    triggers, summary = reconcile_panel(panel)
    assert len(triggers) == 2
    assert summary["trigger_resolution_counts"] == {"consistent_with_factor_ratio": 1, "unexplained": 1}
    assert triggers.iloc[0].factor_implied_reference == 5.


def test_missing_factors_and_cross_stock_boundaries():
    panel = pd.DataFrame({"ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                          "trade_date": ["20240102", "20240103", "20240103"],
                          "close": [10., 5., 3.], "pre_close": [10., 5., 3.], "adj_factor": [1., np.nan, 1.]})
    triggers, summary = reconcile_panel(panel)
    assert len(triggers) == 1
    assert summary["trigger_resolution_counts"] == {"factor_missing": 1}
    assert summary["all_transition_status_counts"]["first_observation"] == 2


def test_factor_anomaly_without_raw_discontinuity_is_not_lost():
    panel = pd.DataFrame({"ts_code": ["603081.SH"] * 2, "trade_date": ["20240624", "20240625"],
                          "close": [9.05, 9.08], "pre_close": [9.05, 9.05], "adj_factor": [1.062, 1.07]})
    triggers, summary = reconcile_panel(panel)
    assert len(triggers) == 0
    assert len(summary["unexplained_transitions"]) == 1
    assert summary["unexplained_transitions"][0]["raw_discontinuity"] is False
