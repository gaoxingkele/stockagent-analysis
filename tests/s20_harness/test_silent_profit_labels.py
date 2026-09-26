import numpy as np
import pandas as pd
import pytest

from research.s20_harness.silent_profit_labels import split_silent_profit


def test_only_silent_a_moves_and_eight_percent_boundary_stays_a():
    original = pd.DataFrame(dict(sample_id=list("abcdefg"),
                                 target=["A", "A", "B", "C", "D", None, "A"],
                                 max_gain=[0.0799, 0.08, 0.03, 0.03, 0.03, 0.03, np.nan]))
    result = split_silent_profit(original)
    assert result.target.fillna("UNKNOWN").tolist() == ["S", "A", "B", "C", "D", "UNKNOWN", "UNKNOWN"]
    assert result.sample_id.tolist() == original.sample_id.tolist()
    pd.testing.assert_series_equal(result.legacy_target, original.target, check_names=False)
    assert original.target.iloc[0] == "A"


def test_accounting_preserves_profit_and_risk_with_s_in_profit():
    original = pd.DataFrame(dict(sample_id=list("abcde"), target=list("AABCD"),
                                 max_gain=[0.04, 0.12, 0.12, 0.03, 0.01]))
    result = split_silent_profit(original)
    assert original.target.isin(["A", "B"]).equals(result.target.isin(["A", "B", "S"]))
    assert original.target.isin(["B", "D"]).equals(result.target.isin(["B", "D"]))


@pytest.mark.parametrize("target,ids", [(["S", "A"], ["a", "b"]), (["A", "B"], ["a", "a"])])
def test_rejects_wrong_version_or_duplicate_ids(target, ids):
    with pytest.raises(ValueError):
        split_silent_profit(pd.DataFrame(dict(sample_id=ids, target=target, max_gain=[0.03, 0.04])))
