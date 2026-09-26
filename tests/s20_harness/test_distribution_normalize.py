import pandas as pd

from research.s20_harness.distribution_normalize import normalize
from research.s20_harness.dividend_source import FIELDS


def row(**changes):
    value = {key: None for key in FIELDS.split(",")}
    value.update(ts_code="000001.SZ", end_date="20231231", ann_date="20231220", div_proc="实施",
                 record_date="20240102", ex_date="20240103", imp_ann_date="20231229", pay_date="20240105",
                 stk_div=0., cash_div=.1, cash_div_tax=.1)
    return {**value, **changes}


def test_repeated_announcement_one_entitlement_complete_lineage():
    data, lineage, summary = normalize(pd.DataFrame([row(), row(ann_date="20231225")]))
    assert len(data) == 1
    assert len(lineage) == 2
    assert lineage.normalized_event_id.nunique() == 1
    assert data.cash_div_tax.iloc[0] == .1
    assert data.terms_known_not_before_date.iloc[0] == "20231229"
    assert summary["announcement_repetitions_collapsed"] == 1


def test_conflicting_rates_and_null_zero_not_collapsed():
    data, lineage, summary = normalize(pd.DataFrame([row(cash_div_tax=None), row(cash_div_tax=0.)]))
    assert len(data) == 2
    assert summary["conflicting_variant_rows_retained"] == 2
    assert not data.event_terms_usable_for_gross_reference_diagnostic.any()


def test_different_periods_retained_and_late_observation_rejected():
    data, lineage, summary = normalize(pd.DataFrame([row(), row(end_date="20230630"), row(ann_date="20240110")]))
    assert len(data) == 2
    assert len(lineage) == 3
    assert "conservative_terms_date_after_record" in summary["final_reason_counts"]
