import pandas as pd
import pytest

from research.s20_harness.gap_validity import combine


def test_independent_support_preserved_and_conflict_not_hidden():
    gaps = pd.DataFrame([dict(ts_code=str(i), trade_date="20260115", evidence_state=exchange,
                             daily_evidence_state="provider_full_day_candidate",
                             daily_receipt_hashes=["abc"], daily_received_at=["2026-09-13T00:00:00Z"])
                         for i, exchange in enumerate(["reported_suspension_consistent", "unresolved", "unresolved", "reported_suspension_consistent"])])
    sem = pd.DataFrame([dict(ts_code=str(i), trade_date="20260115", review_state=state,
                            review_resolved_at="2026-09-14T00:00:00Z", receipt_sha256="abc",
                            received_at="2026-09-13T00:00:00Z", historical_prediction_eligible=False,
                            executable_fill_proven=False)
                        for i, state in enumerate(["not_event_reviewed", "full_day_suspended", "not_event_reviewed", "resumption_announced_quote_observed"])])
    result, report = combine(gaps, sem)
    assert result.suspension_support.tolist() == ["exchange_interval_supported", "event_review_supported", "provider_candidate_only", "conflicting_evidence"]
    assert report["gap_stock_dates"] == 4
    assert not report["formal_training_authorized"]
    pd.testing.assert_frame_equal(result[gaps.columns], gaps)
    with pytest.raises(ValueError, match="missing semantic"):
        combine(gaps, sem.iloc[1:])
    sem.loc[0, "receipt_sha256"] = "bad"
    with pytest.raises(ValueError, match="lineage"):
        combine(gaps, sem)
