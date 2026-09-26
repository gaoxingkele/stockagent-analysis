import pandas as pd
import pytest

from research.s20_harness.label_batch import materialize


def test_batch_keeps_missing_quotes_and_unresolved_events():
    dates = pd.bdate_range("2024-01-01", periods=21).strftime("%Y%m%d").tolist()
    candidates = pd.DataFrame([{"sample_id": k, "entity_id": k, "signal_date": dates[0], "event_codes": [k]}
                               for k in ("ok", "missing", "event")])
    quotes = pd.DataFrame([{"entity_id": k, "trade_date": d, "open": 10., "high": 11., "low": 10., "close": 11.}
                           for k in ("ok", "event") for d in dates])
    distributions = pd.DataFrame([{"ts_code": "event", "record_date": dates[2], "ex_date": dates[3],
                                    "normalized_event_id": "unresolved"}])
    result, report = materialize(candidates, quotes, dates, distributions, [], [])
    assert result.sample_id.tolist() == ["ok", "missing", "event"]
    assert result.p_class.tolist()[0] == "A"
    assert result.p_class.iloc[1:].isna().all()
    assert result.label_status.iloc[2] == "unknown_event_terms"
    assert report["candidate_order_and_denominator_preserved"]
    assert not result.formal_training_eligible.any()
    with pytest.raises(ValueError, match="duplicate/missing"):
        materialize(pd.concat([candidates, candidates]), quotes, dates, distributions, [], [])
    contexts = {"ok": {"acquisition_date": dates[1], "transfer_settlement_date": None,
                       "investor_scope": "personal_public_market_unrestricted_single_lot",
                       "market": "SH", "cash_rate_basis": "gross_per_share"}}
    taxed, report = materialize(candidates, quotes, dates, distributions, [], [], tax_contexts=contexts)
    assert taxed.taxed_p_class.iloc[0] == "A"
    assert taxed.taxed_status.tolist() == ["tax_bounds_diagnostic", "upstream_outcome_unresolved", "upstream_outcome_unresolved"]
    pd.testing.assert_frame_equal(taxed[result.columns], result)
    assert report["tax_bounds_requested"]
