import pandas as pd
import pytest

from research.s20_harness.suspension_crosscheck import compare


def test_crosscheck_preserves_unmatched_and_conflicting_dates():
    exchange = pd.DataFrame([
        dict(ts_code=f"00000{i}.SZ", trade_date="20240109", source_sha256="a" * 64,
             quote_present=False, status="missing_consistent_with_reported_suspension")
        for i in range(1, 5)
    ])
    daily = pd.DataFrame([
        dict(ts_code=code, trade_date="20240109", suspend_type=kind, suspend_timing=None,
             quote_present=quote, receipt_sha256="b" * 64, received_at="2026-09-14T00:00:00Z")
        for code, kind, quote in [("000001.SZ", "S", False), ("000002.SZ", "R", False),
                                  ("000003.SZ", "S", True)]
    ])
    result, report = compare(exchange, daily, ["20240109"])
    assert len(result) == 4
    assert result.ts_code.tolist() == exchange.ts_code.tolist()
    assert result.crosscheck_consistent.tolist() == [True, False, False, False]
    assert report["consistent_stock_dates"] == 1
    assert report["formal_training_authorized"] is False
    assert result.iloc[-1].daily_evidence_state == "no_provider_record_on_queried_date"
    with pytest.raises(ValueError, match="unique"):
        compare(pd.concat([exchange, exchange.iloc[:1]]), daily, ["20240109"])
    exchange.loc[0, "quote_present"] = True
    with pytest.raises(ValueError, match="quote conflict"):
        compare(exchange, daily, ["20240109"])
