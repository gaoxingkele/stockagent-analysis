import pandas as pd
from research.s20_harness.suspension_reconcile import reconcile


def test_full_partial_resume_and_unknown_intervals():
    def event(start, end, code):
        return dict(ts_code=code, source_sha256="a", suspended_at=start, resumed_at=end,
                    interval_status="unknown_end" if end is None else "closed_reported")
    intervals = pd.DataFrame([
        event("2024-01-02T09:30:00+08:00", "2024-01-04T09:30:00+08:00", "a"),
        event("2024-01-02T10:00:00+08:00", "2024-01-02T14:00:00+08:00", "b"),
        event("2024-01-02T09:30:00+08:00", None, "c")])
    observed = pd.DataFrame({"ts_code": ["a", "a", "b"], "trade_date": ["20240103", "20240104", "20240102"]})
    events, detail, report = reconcile(intervals, observed, ["20240102", "20240103", "20240104"])
    assert detail.trade_date.tolist() == ["20240102", "20240103"]
    assert report["conflicting_quote_rows"] == 1
    assert report["unknown_end_events"] == 1
    assert report["partial_session_intersections"] == 1
    assert not report["formal_coverage_verified"]
