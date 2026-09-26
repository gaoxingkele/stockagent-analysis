import pandas as pd
import pytest

from research.s20_harness.listing_edges import expand


def test_listing_edges_include_entirely_missing_and_exclude_delisting_day():
    dates = ["20240102", "20240103", "20240104"]
    data = pd.DataFrame([
        dict(ts_code="a", list_date="20200101", delist_date=None, observed_days=1,
             observed_first="20240103", observed_last="20240103", expected_listing_days=3, unobserved_listing_days=2),
        dict(ts_code="b", list_date="20200101", delist_date="20240104", observed_days=0,
             observed_first=None, observed_last=None, expected_listing_days=2, unobserved_listing_days=2),
    ])
    rows, unknown = expand(data, dates)
    assert len(rows) == 4
    assert not unknown
    assert rows.loc[rows.ts_code.eq("b"), "trade_date"].tolist() == dates[:2]
    assert rows.edge_kind.tolist()[:2] == ["before_first_quote", "after_last_quote"]
    data.loc[0, "expected_listing_days"] = 4
    with pytest.raises(ValueError, match="count disagreement"):
        expand(data, dates)
