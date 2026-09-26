import pandas as pd

from research.s20_harness.distribution_adapter import adapt, row_fingerprint, window_events
from research.s20_harness.distribution_adapter import refusal_summary


def frame():
    return pd.DataFrame([dict(normalized_event_id="e", ts_code="A", record_date="20240103", ex_date="20240104",
        terms_known_not_before_date="20240101", cash_div_tax=1., stk_div=0., pay_date="20240105", div_listdate=None,
        event_terms_usable_for_gross_reference_diagnostic=True, conflicting_variants_same_identity=False)])


def review(data):
    return {"e": dict(row_sha256=row_fingerprint(data.iloc[0]), beneficiary_scope="existing_shareholders_verified", source="reviewed_fixture")}


def test_refusal_routing_does_not_approve_clean_looking_cash():
    data = frame()
    _, decisions = adapt(data, {}, rate_policy="gross_reference_diagnostic")
    result = refusal_summary(data, decisions)
    assert result["routes"] == {"cash_only_beneficiary_review_missing": 1}
    assert result["routing_is_not_acceptance"]
    assert not decisions[0]["accepted_for_accounting"]
    data.loc[0, "stk_div"] = 1.
    assert refusal_summary(data, decisions)["routes"] == {"unreviewed_share_distribution": 1}


def test_requires_bound_beneficiary_review_and_preserves_unresolved_window():
    data = frame()
    events, decisions = adapt(data, {}, rate_policy="gross_reference_diagnostic")
    assert events == [] and decisions[0]["reasons"] == ["beneficiary_review_missing"]
    assert window_events(data, events, decisions, ["A"], "20240102", "20240110")["status"] == "unknown_event_terms"
    events, decisions = adapt(data, review(data), rate_policy="gross_reference_diagnostic")
    assert events[0].cash_per_share == 1 and events[0].known_date == "20240102"
    assert not decisions[0]["formal_training_eligible"]
    assert len(window_events(data, events, decisions, ["A"], "20240102", "20240110")["events"]) == 1


def test_changed_terms_and_net_tax_cannot_reuse_gross_review():
    data = frame()
    checked = review(data)
    data.loc[0, "cash_div_tax"] = 2.
    assert "review_row_hash_mismatch" in adapt(data, checked, rate_policy="gross_reference_diagnostic")[1][0]["reasons"]
    assert "net_cash_policy_unverified" in adapt(data, review(data), rate_policy="reviewed_net_cash")[1][0]["reasons"]


def test_missing_record_date_cannot_disappear_from_window():
    data = frame()
    data.loc[0, "record_date"] = None
    events, decisions = adapt(data, {}, rate_policy="gross_reference_diagnostic")
    result = window_events(data, events, decisions, ["A"], "20240102", "20240110")
    assert result["unresolved_event_ids"] == ["e"]


def test_cdr_requires_beneficiary_and_quote_unit_review_together():
    data = frame()
    reviews = review(data)
    reviews["e"]["beneficiary_scope"] = "registered_cdr_holders_verified"
    assert "cdr_quote_unit_review_required" in adapt(data, reviews, rate_policy="gross_reference_diagnostic")[1][0]["reasons"]
    unit = {"e": {"row_sha256": row_fingerprint(data.iloc[0]), "ts_code": "A", "record_date": "20240103",
            "ex_date": "20240104", "source": "issuer", "source_unit": "underlying_share", "quote_unit": "domestic_CDR",
            "quoted_units_per_source_unit": 10, "source_gross_cash": 1., "gross_cash_per_quote_unit": .1}}
    events, decisions = adapt(data, reviews, rate_policy="gross_reference_diagnostic", unit_reviews=unit)
    assert events[0].cash_per_share == .1 and events[0].quote_unit == "domestic_CDR"
    assert decisions[0]["accepted_for_accounting"]


def test_multiple_entitlements_need_complete_group_review_and_never_override_conflicts():
    data = pd.concat([frame(), frame()], ignore_index=True)
    data.loc[1, "normalized_event_id"] = "f"
    data.loc[1, "cash_div_tax"] = 2.
    data["audit_reasons"] = "multiple_events_same_stock_ex_date"
    data["event_terms_usable_for_gross_reference_diagnostic"] = False
    reviews = {str(r.normalized_event_id): dict(row_sha256=row_fingerprint(r), source="fixture",
                beneficiary_scope="existing_shareholders_verified", independent_entitlement_group=["e", "f"])
               for _, r in data.iterrows()}
    events, decisions = adapt(data, reviews, rate_policy="gross_reference_diagnostic")
    assert len(events) == 2 and sum(e.cash_per_share for e in events) == 3
    assert not adapt(data, {"e": reviews["e"]}, rate_policy="gross_reference_diagnostic")[0]
    data.loc[1, "conflicting_variants_same_identity"] = True
    reviews["f"]["row_sha256"] = row_fingerprint(data.iloc[1])
    assert not adapt(data, reviews, rate_policy="gross_reference_diagnostic")[0]
