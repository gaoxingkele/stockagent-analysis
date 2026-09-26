import pandas as pd
import pytest
from research.s20_harness.suspension_daily_source import collect


def test_resume_uses_receipts_and_rejects_mutation(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    for date in ["20240102", "20240103"]:
        pd.DataFrame({"ts_code": ["a"]}).to_parquet(daily / (date + ".parquet"))
    calls = []
    def query(date):
        calls.append(date)
        return pd.DataFrame(dict(ts_code=["000046.SZ"], trade_date=[date], suspend_type=["S"], suspend_timing=[None]))
    first = collect(tmp_path, query, max_requests=1, pause=0)
    second = collect(tmp_path, query, first["source_id"], pause=0)
    assert second["complete_acquisition"] and calls == ["20240102", "20240103"]
    assert second["requests_used"] == 1
    path = tmp_path / "output/experiments/s20_safe_v4/sources" / first["source_id"] / "20240102.parquet"
    pd.DataFrame({"bad": [1]}).to_parquet(path)
    with pytest.raises(ValueError, match="receipt mismatch"):
        collect(tmp_path, query, first["source_id"], pause=0)
