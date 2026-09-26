import pandas as pd

from research.s20_harness.cash_rate_audit import compare


def test_cash_rate_consistency_and_scale_ambiguity():
    events = pd.DataFrame([{"normalized_event_id": str(i), "ts_code": code,
        "record_date": "20260529", "ex_date": "20260601", "cash_div_tax": cash,
        "event_terms_usable_for_gross_reference_diagnostic": True,
        "conflicting_variants_same_identity": False, "stk_div": 0.}
        for i, (code, cash) in enumerate([("000001.SZ", .5), ("000002.SZ", 5.), ("000003.SZ", .001)])])
    quotes = pd.DataFrame([{"ts_code": code, "trade_date": d, "close": 10., "pre_close": pre}
        for code, pre in [("000001.SZ", 9.5), ("000002.SZ", 9.5), ("000003.SZ", 10.)]
        for d in ["20260529", "20260601"]])
    result = compare(events, quotes, ["20260529", "20260601"])
    assert result.status.tolist() == ["consistent_declared_rate", "possible_unit_scale_mismatch", "rounding_scale_ambiguous"]
    assert not result.formal_eligible.any()
