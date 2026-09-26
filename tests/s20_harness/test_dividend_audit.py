import pandas as pd

from research.s20_harness.dividend_audit import audit_events, compare_reference_transitions


def event():
    return dict(ts_code="000001.SZ", ex_date="20240103", record_date="20240102", pay_date="20240105",
                div_listdate=None, imp_ann_date="20231229", ann_date="20231201", div_proc="实施",
                cash_div_tax=.1, stk_div=0.)


def test_terms_pass_does_not_prove_net_pit():
    records, summary = audit_events(pd.DataFrame([event()]))
    assert records.event_terms_usable_for_gross_reference_diagnostic.all()
    assert not summary["cash_tax_policy_validated"]
    assert not summary["historical_feed_revision_policy_validated"]


def test_unknown_cash_not_zero_and_missing_bonus_listing():
    item = event()
    item.update(cash_div_tax=None, stk_div=.1)
    records, summary = audit_events(pd.DataFrame([item]))
    assert not records.event_terms_usable_for_gross_reference_diagnostic.any()
    assert summary["reason_counts"]["cash_rate_unknown_or_invalid"] == 1
    assert summary["reason_counts"]["bonus_listing_unknown_or_invalid"] == 1


def test_duplicate_events_and_late_announcement_not_silently_resolved():
    item = event()
    item["imp_ann_date"] = "20240104"
    records, summary = audit_events(pd.DataFrame([item, item]))
    assert len(records) == 2
    assert summary["reason_counts"]["multiple_events_same_stock_ex_date"] == 2
    assert summary["reason_counts"]["implementation_announcement_after_record"] == 2


def test_reference_diagnostic_matches_cash_but_keeps_unknown_stock():
    records, _ = audit_events(pd.DataFrame([event()]))
    transitions = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"],
                                "trade_date": ["20240103"] * 2, "previous_close": [10., 10.], "pre_close": [9.9, 9.9]})
    result = compare_reference_transitions(transitions, records)
    assert result.event_reference_status.tolist() == ["consistent_with_simple_gross_distribution", "no_matching_distribution"]
