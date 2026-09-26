import pandas as pd
import pytest

from research.s20_harness.adjustment_source import collect, validate_factors


def factors(trade_date):
    return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [trade_date], "adj_factor": [2.]})


def test_missing_codes_reported_not_filled():
    result = validate_factors(factors("20240102"), "20240102", ["000001.SZ", "000002.SZ"])
    assert result["valid"]
    assert result["missing_daily_codes"] == ["000002.SZ"]
    assert not validate_factors(factors("20240102"), "20240103", ["000001.SZ"])["valid"]


def test_resume_hash_check_and_request_cap(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    for date in ("20240102", "20240103"):
        factors(date).to_parquet(daily / f"{date}.parquet")
    first = collect(tmp_path, factors, max_requests=1, pause=0)
    assert not first["complete_acquisition"]
    second = collect(tmp_path, factors, source_id=first["source_id"], max_requests=1, pause=0)
    assert second["complete_acquisition"]
    base = tmp_path / "output/experiments/s20_safe_v4/sources" / first["source_id"]
    (base / "20240102.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        collect(tmp_path, factors, source_id=first["source_id"], pause=0)
