import json
from pathlib import Path

import pandas as pd
import pytest

from research.s20_harness.entity_coverage import build
from research.s20_harness.entity_coverage_verify import verify
from research.s20_harness.runtime import atomic_json, digest
from research.s20_harness.source_evidence import inspect_sources
from tests.s20_harness.test_source_evidence import identity_source


def setup(tmp_path):
    inv = identity_source(tmp_path)
    source = inv["sources"][0]
    panel = tmp_path/source["path"]
    basic = tmp_path/"basic.parquet"
    pd.DataFrame([dict(ts_code=c, list_status="L", list_date="20200101", delist_date=None)
                  for c in ("000002.SZ", "000003.SZ")]).to_parquet(basic, index=False)
    cal = tmp_path/"cal.parquet"
    pd.DataFrame([dict(exchange=ex, is_open=1, cal_date=d) for ex in ("SSE", "SZSE")
                  for d in ("20240102", "20240103", "20240104")]).to_parquet(cal, index=False)
    out = Path(build(tmp_path, panel, source["summary_sha256"], basic, digest(basic), cal, digest(cal))["directory"])
    return out


def test_reconstruction_retains_completely_unobserved_entity(tmp_path):
    out = setup(tmp_path)
    result = verify(tmp_path, out, digest(out/"summary.json"))
    assert result["unobserved_listing_entities"] == ["000003.SZ"]
    assert result["observed_entity_dates"] == 3 and not result["formal_universe_verified"]
    records, _ = inspect_sources(tmp_path, {"sources": [{"role": "entity_listing_coverage", "path": str(out),
                                "summary_sha256": digest(out/"summary.json")}]}, revalidate_coverage=True)
    assert records[0]["fresh_coverage_validation"]["semantic_reconstruction_verified"]


def test_rehashed_false_coverage_rejected(tmp_path):
    out = setup(tmp_path)
    table = pd.read_parquet(out/"coverage.parquet")
    table.loc[1, "observed_days"] = 999
    table.to_parquet(out/"coverage.parquet", index=False)
    summary = json.loads((out/"summary.json").read_text())
    summary["table_sha256"] = digest(out/"coverage.parquet")
    atomic_json(out/"summary.json", summary)
    with pytest.raises(ValueError, match="semantic table"):
        verify(tmp_path, out, digest(out/"summary.json"))
