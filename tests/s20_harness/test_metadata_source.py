import pandas as pd
import pytest

from research.s20_harness.metadata_source import collect


def test_empty_history_is_unknown_not_never_st(tmp_path):
    def query(api, fields, params):
        return pd.DataFrame(columns=fields.split(","))
    result = collect(tmp_path, query, page_size=2, page_cap=2)
    assert result["pagination_finished"]
    assert not result["global_endpoint_coverage_proven"]
    assert not result["formal_H01_gate_passed"]


def test_ignored_offset_is_not_silently_complete(tmp_path):
    def query(api, fields, params):
        if api == "stock_basic":
            return pd.DataFrame(columns=fields.split(","))
        return pd.DataFrame([{c: "value" for c in fields.split(",")}])
    with pytest.raises(ValueError, match="repeated page"):
        collect(tmp_path, query, page_size=1, page_cap=3)
