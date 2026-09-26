import pandas as pd
import pytest

from research.s20_harness.dividend_source import FIELDS, collect, validate_events


def empty(**params):
    return pd.DataFrame(columns=FIELDS.split(","))


def test_zero_events_are_valid_but_no_ledger_claim():
    result = validate_events(empty(), "20240102")
    assert result["valid"]
    assert not result["economic_ledger_validated"]
    frame = pd.DataFrame([{**{c: None for c in FIELDS.split(",")}, "ex_date": "20240102"}] * 2000)
    assert not validate_events(frame, "20240102")["valid"]


def test_resume_preserves_zero_event_receipt(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    empty().to_parquet(daily / "20240102.parquet")
    result = collect(tmp_path, empty, pause=0)
    assert result["complete_acquisition"]
    assert result["zero_event_dates"] == 1
    def forbidden(**params):
        raise AssertionError("cached date queried again")
    again = collect(tmp_path, forbidden, source_id=result["source_id"], pause=0)
    assert again["new_requests"] == 0
