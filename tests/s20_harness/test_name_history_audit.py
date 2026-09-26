import pandas as pd
import pytest

from research.s20_harness.metadata_source import NAME_FIELDS
from research.s20_harness.name_history_audit import audit
from research.s20_harness.runtime import atomic_json, digest


def make_source(tmp_path, rows):
    atomic_json(tmp_path / "collection_plan.json", {"codes": ["000001.SZ", "000002.SZ"]})
    frame = pd.DataFrame(rows, columns=NAME_FIELDS.split(","))
    path = tmp_path / "000001.SZ.parquet"
    frame.to_parquet(path, index=False)
    atomic_json(tmp_path / "000001.SZ.json", {"code": "000001.SZ", "rows": len(frame), "sha256": digest(path)})
    return tmp_path


def test_partial_empty_is_unknown_not_normal(tmp_path):
    result = audit(make_source(tmp_path, []))
    assert result["empty_unknown_codes"] == ["000001.SZ"]
    assert result["uncollected_codes"] == ["000002.SZ"]
    assert not result["complete_receipt_coverage"]
    assert not result["official_ST_status_proven"]


def test_semantic_errors_retained(tmp_path):
    rows = []
    for name in ("normal", "STother"):
        row = dict.fromkeys(NAME_FIELDS.split(","))
        row.update(ts_code="000001.SZ", name=name, start_date="20240102", end_date="20230101", ann_date="20240101")
        rows.append(row)
    result = audit(make_source(tmp_path, rows))
    assert result["totals"]["reversed_period_rows"] == 2
    assert result["totals"]["conflicting_effective_announcement_groups"] == 1


def test_tampered_input_rejected(tmp_path):
    source = make_source(tmp_path, [])
    (source / "000001.SZ.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit(source)
