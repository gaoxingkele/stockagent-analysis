import pandas as pd
import pytest

from research.s20_harness.price_limit_source import collect
from research.s20_harness.price_limit_audit import audit
from research.s20_harness.runtime import atomic_json, load_plan


def test_resume_preserves_invalid_limits_and_checks_hash(tmp_path):
    daily = tmp_path / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    for date in ("20240102", "20240103"):
        pd.DataFrame({"ts_code": ["000001.SZ"]}).to_parquet(daily / (date + ".parquet"))
    def query(date):
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date], "up_limit": [0.], "down_limit": [0.]})
    first = collect(tmp_path, query, max_requests=1, pause=0)
    assert not first["complete_acquisition"]
    second = collect(tmp_path, query, first["source_id"], max_requests=1, pause=0)
    assert second["complete_acquisition"] and second["invalid_limit_rows"] == 2
    assert not second["formal_H01_gate_passed"]
    output = tmp_path / "output/experiments/s20_safe_v4/sources" / first["source_id"]
    verified = audit(tmp_path, output)
    assert verified["acquisition_receipts_verified"]
    assert verified["invalid_limit_rows"] == 2
    receipt = load_plan(output / "20240102.json")
    original = dict(receipt)
    receipt["received_at"] = "2000-01-01T00:00:00+00:00"
    atomic_json(output / "20240102.json", receipt)
    with pytest.raises(ValueError, match="receipt timing"):
        audit(tmp_path, output)
    atomic_json(output / "20240102.json", original)
    (output / "20240102.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="receipt mismatch"):
        collect(tmp_path, query, first["source_id"], pause=0)
