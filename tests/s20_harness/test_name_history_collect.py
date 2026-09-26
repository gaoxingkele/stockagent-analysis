import pandas as pd
import pytest

from research.s20_harness.name_history_collect import collect
from research.s20_harness.metadata_source import NAME_FIELDS


def test_bounded_resume_preserves_empty_unknown_codes(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"]}).to_parquet(daily / "20240102.parquet")
    def query(code):
        return pd.DataFrame(columns=NAME_FIELDS.split(","))
    first = collect(tmp_path, query, max_requests=1, pause=0)
    assert not first["complete_acquisition"]
    second = collect(tmp_path, query, source_id=first["source_id"], max_requests=1, pause=0)
    assert second["complete_acquisition"]
    assert len(second["empty_codes"]) == 2
    assert not second["formal_H01_gate_passed"]
    path = tmp_path / "output/experiments/s20_safe_v4/sources" / first["source_id"] / "000001.SZ.parquet"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        collect(tmp_path, query, source_id=first["source_id"], pause=0)
