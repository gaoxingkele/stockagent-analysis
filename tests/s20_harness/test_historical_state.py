import pandas as pd
import pytest

from research.s20_harness.historical_state import attach_names


def states():
    return pd.DataFrame([dict(source_code="A", valid_from="20240103", valid_until_exclusive="20240106",
                              name="STa", name_st_proxy=True, status="known_name_proxy")])


def test_join_keeps_order_unknown_and_duplicate_recommendations():
    panel = pd.DataFrame({"trading_code": ["A", "A", "B", "A"],
                          "trade_date": ["20240104", "20240102", "20240104", "20240104"],
                          "value": [1, 2, 3, 4]}, index=[8, 3, 9, 1])
    joined = attach_names(panel, states())
    assert joined.index.tolist() == panel.index.tolist()
    assert joined.value.tolist() == [1, 2, 3, 4]
    assert joined.name_proxy_status.tolist() == ["known_name_proxy", "unknown", "unknown", "known_name_proxy"]
    assert pd.isna(joined.name_st_proxy.iloc[1])
    assert pd.isna(joined.name_st_proxy.iloc[2])


def test_overlapping_intervals_do_not_multiply_rows():
    with pytest.raises(ValueError, match="overlapping"):
        attach_names(pd.DataFrame({"trading_code": ["A"], "trade_date": ["20240104"]}),
                     pd.concat([states(), states()], ignore_index=True))
