import pandas as pd
import pytest

from research.s20_harness.full_data_audit import audit_dataset
from research.s20_harness.runtime import atomic_json, digest


def row(date, code="000001.SZ", close=10., pre=10.):
    return dict(ts_code=code, trade_date=date, open=close, high=close + 1,
                low=close - 1, close=close, pre_close=pre, vol=100., amount=1000.)


def test_full_scan_retains_bad_rows_gaps_and_discontinuity(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    pd.DataFrame([row("20240102")]).to_parquet(daily / "20240102.parquet")
    pd.DataFrame([row("20240103", "000002.SZ")]).to_parquet(daily / "20240103.parquet")
    bad = row("20240105", "000003.SZ")
    bad["high"] = 5.
    pd.DataFrame([row("20240104", pre=8.), bad]).to_parquet(daily / "20240104.parquet")
    result = audit_dataset(tmp_path, tmp_path / "audit")
    assert result["files_scanned"] == 3
    assert result["rows_scanned"] == 4
    assert result["reference_close_discontinuities"] == 1
    assert result["missing_observed_sessions"] == 1
    assert result["anomaly_counts"]["invalid_ohlc_bounds"] == 1
    assert result["anomaly_counts"]["file_date_mismatch"] == 1
    assert not result["formal_gate_passed"]
    assert len(pd.read_parquet(tmp_path / "audit/anomaly_ledger.parquet")) == 4


def test_clean_daily_does_not_prove_pit_or_calendar(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    pd.DataFrame([row("20240102")]).to_parquet(daily / "20240102.parquet")
    result = audit_dataset(tmp_path, tmp_path / "audit")
    assert result["anomaly_counts"] == {}
    assert not result["formal_gate_passed"]
    assert result["observed_dates_are_not_verified_exchange_calendar"]
    assert any("exchange_calendar" in gap for gap in result["acceptance_gaps"])


def test_no_data_is_not_pass(tmp_path):
    result = audit_dataset(tmp_path, tmp_path / "audit")
    assert result["files_scanned"] == 0
    assert "no daily files" in result["acceptance_gaps"]


def test_registered_calendar_hash_and_semantics(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    pd.DataFrame([row("20240102")]).to_parquet(daily / "20240102.parquet")
    path = tmp_path / "calendar.parquet"
    pd.DataFrame({"exchange": ["SSE", "SZSE"], "cal_date": ["20240102"] * 2,
                  "is_open": [1, 1], "pretrade_date": ["20231229"] * 2}).to_parquet(path)
    (tmp_path / "config").mkdir()
    atomic_json(tmp_path / "config/s20_v4_data_sources.json", {"sources": [{
        "role": "exchange_calendar", "status": "validated_calendar_source", "path": "calendar.parquet",
        "sha256": digest(path), "start": "20240102", "end": "20240102"}]})
    result = audit_dataset(tmp_path, tmp_path / "audit")
    assert result["support_sources"]["exchange_calendar"]["semantic_validation"] == "validated_SH_SZ"
    assert not result["formal_gate_passed"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_dataset(tmp_path, tmp_path / "other")
