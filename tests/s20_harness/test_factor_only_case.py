import pandas as pd
import pytest

from research.s20_harness.adjustment_reconcile import reconcile_panel


def test_factor_only_break_is_detected_even_without_raw_discontinuity():
    frame = pd.DataFrame([
        dict(ts_code="603081.SH", trade_date="20240624", close=9.05, pre_close=9.64, adj_factor=1.062),
        dict(ts_code="603081.SH", trade_date="20240625", close=9.08, pre_close=9.05, adj_factor=1.070),
    ])
    original = frame.copy(deep=True)
    triggers, summary = reconcile_panel(frame)
    assert triggers.empty
    assert len(summary["unexplained_transitions"]) == 1
    case = summary["unexplained_transitions"][0]
    assert case["raw_discontinuity"] is False
    assert case["adjusted_reference_error"] == pytest.approx(.06766355140186953)
    assert case["trade_date"] == "20240625"
    pd.testing.assert_frame_equal(frame, original)
