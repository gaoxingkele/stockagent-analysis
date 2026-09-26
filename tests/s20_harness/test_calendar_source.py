import pandas as pd
import pytest

from research.s20_harness.calendar_source import acquire, validate_calendar


def fixture_calendar(exchange="SSE"):
    return pd.DataFrame({"exchange": [exchange] * 3, "cal_date": ["20240101", "20240102", "20240103"],
                         "is_open": [0, 1, 1], "pretrade_date": ["20231229", "20231229", "20240102"]})


def test_calendar_chain_and_full_day_coverage():
    frame = fixture_calendar()
    assert validate_calendar(frame, "SSE", "20240101", "20240103")["valid"]
    assert not validate_calendar(frame.iloc[1:], "SSE", "20240101", "20240103")["valid"]
    frame.loc[2, "pretrade_date"] = "20231229"
    assert "pretrade chain mismatch" in validate_calendar(frame, "SSE", "20240101", "20240103")["errors"]


def test_acquisition_receipts_do_not_pass_full_data_gate(tmp_path):
    result = acquire(tmp_path, lambda **p: fixture_calendar(p["exchange"]), "20240101", "20240103")
    assert result["calendar_validation"]["SSE"]["valid"]
    assert not result["formal_H01_gate_passed"]
    assert result["daily_file_comparison"]["SZSE"]["calendar_open_missing_daily_file"] == ["20240102", "20240103"]


def test_sensitive_provider_error_is_not_persisted(tmp_path):
    def fail(**params):
        raise RuntimeError("secret-token-here")
    with pytest.raises(RuntimeError, match="sanitized"):
        acquire(tmp_path, fail, "20240101", "20240103")
    receipt = next(tmp_path.rglob("failure.json"))
    assert "secret-token-here" not in receipt.read_text()
