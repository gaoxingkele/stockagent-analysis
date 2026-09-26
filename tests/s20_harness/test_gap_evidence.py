import pandas as pd
import pytest
from research.s20_harness.gap_evidence import join_gaps


def test_gap_count_conservation_evidence_deduplication_and_unmatched():
    a = pd.DataFrame([dict(ts_code="a", trade_date="20240105", issue="missing_observed_sessions",
                           detail="previous=20240102;count=2;not_proven_suspension")])
    e = pd.DataFrame([dict(ts_code="a", trade_date=d, status="missing_consistent_with_reported_suspension",
                           source_sha256=h, quote_present=False) for d,h in [("20240103","x"),("20240103","y"),("20240102","x")]])
    calendar = ["20240102","20240103","20240104","20240105"]
    table, report = join_gaps(a,e,calendar)
    assert len(table)==2 and table.iloc[0].source_hashes==["x","y"]
    assert report["states"] == {"reported_suspension_consistent":1,"unresolved":1}
    assert report["evidence_stock_dates_outside_original_gaps"]==1
    with pytest.raises(ValueError,match="count"):
        join_gaps(a,e,calendar[:2]+calendar[3:])
