import pandas as pd
import pytest

from research.s20_harness.price_limit_audit import inspect_limits


def test_invalid_duplicate_and_wrong_day_are_not_usable():
    frame = pd.DataFrame({"ts_code": ["A", "B", "B", "C", "D", "E"],
                          "trade_date": ["20240102"] * 5 + ["20240103"],
                          "up_limit": [11, 11, 11, float("inf"), 8, 11],
                          "down_limit": [9, 9, 9, 9, 9, 9]})
    result = inspect_limits(frame, "20240102", set("ABCDEF"))
    assert result["usable_expected_codes"] == 1
    assert result["raw_missing_expected_codes"] == ["F"]
    assert result["invalid_limit_rows"] == 2
    assert result["duplicate_code_rows"] == 2
    assert result["wrong_date_rows"] == 1


def test_missing_columns_do_not_become_no_limit():
    with pytest.raises(ValueError, match="missing limit fields"):
        inspect_limits(pd.DataFrame(), "20240102", {"A"})
