import pandas as pd
import pytest
from research.s20_harness.runtime import digest
from research.s20_harness.suspension_daily_source import collect
from research.s20_harness.suspension_daily_audit import audit


def test_partial_receipts_quote_conflict_and_pins(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    for date in ["20240102", "20240103"]:
        pd.DataFrame({"ts_code": ["000046.SZ"]}).to_parquet(daily / (date + ".parquet"))
    def query(date):
        return pd.DataFrame(dict(ts_code=["000046.SZ"], trade_date=[date], suspend_type=["S"], suspend_timing=[None]))
    result = collect(tmp_path, query, max_requests=1, pause=0)
    directory = tmp_path / "output/experiments/s20_safe_v4/sources" / result["source_id"]
    sha = digest(directory / "collection_plan.json")
    checked = audit(tmp_path, directory, expected_plan_sha256=sha)
    assert checked["committed_dates"] == 1
    assert checked["uncommitted_dates"] == ["20240103"]
    assert checked["full_day_candidates_with_quotes"] == 1
    assert not checked["complete_receipt_coverage"]
    with pytest.raises(ValueError, match="pin mismatch"):
        audit(tmp_path, directory, expected_plan_sha256="wrong")
