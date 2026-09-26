import pandas as pd
import pytest

from research.s20_harness.universe_coverage import listing_coverage


def test_retired_and_missing_stocks_not_dropped():
    basic = pd.DataFrame([
        dict(ts_code="old", list_status="D", list_date="20200101", delist_date="20240104"),
        dict(ts_code="missing", list_status="L", list_date="20200101", delist_date=None),
        dict(ts_code="future", list_status="L", list_date="20250101", delist_date=None),
        dict(ts_code="unknown", list_status="D", list_date="20200101", delist_date=None)])
    obs = pd.DataFrame({"ts_code": ["old", "old", "alias"], "trade_date": ["20240102", "20240104", "20240103"]})
    table, report = listing_coverage(basic, obs, ["20240102", "20240103", "20240104"])
    assert len(table) == 4
    assert table.iloc[0].outside_listing_days == 1
    assert table.iloc[0].unobserved_listing_days == 1
    assert table.iloc[1].status == "no_observations_in_listing_window"
    assert table.iloc[2].status == "outside_audit_window"
    assert table.iloc[3].status == "unknown_listing_interval"
    assert report["observed_codes_missing_metadata"] == ["alias"]
    assert not report["formal_universe_verified"]
    with pytest.raises(ValueError, match="duplicate"):
        listing_coverage(basic, pd.concat([obs, obs]), ["20240102", "20240103", "20240104"])
