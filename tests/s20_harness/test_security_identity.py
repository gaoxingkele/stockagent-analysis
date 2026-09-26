import pandas as pd
import pytest

from research.s20_harness.security_identity import canonicalize_aliases

ALIAS = dict(entity_id="issuer", old_code="300114.SZ", new_code="302132.SZ", effective_date="20250217", share_conversion_ratio=1.)


def test_equal_alias_rows_not_two_stocks_and_date_effective_code():
    frame = pd.DataFrame({"ts_code": ["300114.SZ", "302132.SZ", "302132.SZ"],
                          "trade_date": ["20250214", "20250214", "20250217"], "close": [72.18, 72.18, 68.01]})
    canonical, lineage, conflicts = canonicalize_aliases(frame, [ALIAS])
    assert canonical.ts_code.tolist() == ["300114.SZ", "302132.SZ"]
    assert canonical.entity_id.nunique() == 1
    assert len(lineage) == 3 and conflicts.empty


def test_conflicting_alias_quotes_have_no_arbitrary_winner():
    frame = pd.DataFrame({"ts_code": ["300114.SZ", "302132.SZ"], "trade_date": ["20250214"] * 2, "close": [70., 72.]})
    canonical, lineage, conflicts = canonicalize_aliases(frame, [ALIAS])
    assert canonical.empty and len(lineage) == 2 and len(conflicts) == 1


def test_equal_quotes_alone_do_not_merge_different_companies():
    frame = pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "trade_date": ["20250214"] * 2, "close": [10., 10.]})
    canonical, _, conflicts = canonicalize_aliases(frame, [ALIAS])
    assert len(canonical) == 2
    assert canonical.entity_id.nunique() == 2
    with pytest.raises(ValueError):
        canonicalize_aliases(frame, [{**ALIAS, "share_conversion_ratio": 2.}])
