import pandas as pd
from research.s20_harness.daily_gap_join import overlay


def test_missing_date_is_not_no_event_and_state_types_preserved():
    gaps = pd.DataFrame({"ts_code": list("abcdef"), "trade_date": ["20240102"]*5+["20240103"]})
    events = pd.DataFrame({"ts_code":list("abcd"), "trade_date":["20240102"]*4,
        "suspend_type":["S","S","R","S"], "suspend_timing":[None,"10:00-11:00",None,None],
        "quote_present":[False,False,False,True], "receipt_sha256":["x"]*4, "received_at":["2026-09-14T00:00:00+08:00"]*4})
    joined, report = overlay(gaps,events,["20240102"])
    assert joined.daily_evidence_state.tolist()==[
        "provider_full_day_candidate","provider_intraday_or_unknown_timing","provider_resumption_but_gap",
        "quote_conflict_with_original_gap","no_provider_record_on_queried_date","date_not_yet_queried"]
    assert report["gap_stock_dates"]==6
    assert not report["formal_training_authorized"]


def test_naive_receipt_time_is_not_historical_evidence():
    import pytest
    gaps = pd.DataFrame({"ts_code":["a"], "trade_date":["20240102"]})
    events = pd.DataFrame({"ts_code":["a"], "trade_date":["20240102"],
        "suspend_type":["S"], "suspend_timing":[None], "quote_present":[False],
        "receipt_sha256":["x"], "received_at":["2026-09-14"]})
    with pytest.raises(ValueError, match="timezone-aware"):
        overlay(gaps, events, ["20240102"])
