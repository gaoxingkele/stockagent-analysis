import pandas as pd

from research.s20_harness.cash_event_bounds import evaluate


def test_cash_envelope_stable_vs_ambiguous_without_approving_events():
    raw = pd.DataFrame({k: [10., 8.5, 8.5] for k in ("open", "high", "low", "close")},
                       index=["20260505", "20260506", "20260507"])
    event = pd.DataFrame([{"record_date": "20260505", "ex_date": "20260506", "stk_div": 0.,
        "cash_div_tax": .1, "event_terms_usable_for_gross_reference_diagnostic": True,
        "conflicting_variants_same_identity": False, "normalized_event_id": "cash"}])
    stable = evaluate(raw, event, buy_cost=0, sell_cost=0)
    assert stable["p_class"] == "D"
    assert not stable["beneficiary_verified"]
    event["cash_div_tax"] = 2.
    ambiguous = evaluate(raw, event, buy_cost=0, sell_cost=0)
    assert ambiguous["p_class"] is None
    assert ambiguous["class_outer_set"] == ["A", "B", "C", "D"]
    event["stk_div"] = 1.
    assert evaluate(raw, event)["status"] == "unsupported_or_unresolved_event_terms"
