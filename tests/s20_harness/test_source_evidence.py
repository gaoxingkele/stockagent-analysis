import pytest

from research.s20_harness.runtime import atomic_json, digest
from research.s20_harness.source_evidence import inspect_sources


def test_component_snapshot_not_semantic_approval(tmp_path):
    base = tmp_path / "output/component"
    base.mkdir(parents=True)
    atomic_json(base / "summary.json", {"rows": 7})
    inventory = {"sources": [{"role": "adjustment_factors", "path": "output/component",
                               "summary_sha256": digest(base / "summary.json"),
                               "reconciliation": "summary.json"}]}
    records, artifacts = inspect_sources(tmp_path, inventory)
    assert records[0]["present"] and not records[0]["semantic_acceptance"]
    assert records[0]["evidence"][0]["hash_verified"] is False
    assert records[0]["evidence"][1]["hash_verified"] is True
    assert artifacts and records[0]["formal_requirement"] == "adjustment_factors"
    atomic_json(base / "summary.json", {"rows": 8})
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_sources(tmp_path, inventory)


def test_missing_component_evidence_retained(tmp_path):
    records, artifacts = inspect_sources(tmp_path, {"sources": [{
        "role": "per_code_historical_names", "path": "missing", "audit": "summary.json"}]})
    assert not records[0]["present"]
    assert not records[0]["evidence"][0]["present"]
    assert artifacts == []


def identity_source(tmp_path):
    from pathlib import Path
    from research.s20_harness.identity_panel import build
    from tests.s20_harness.test_identity_panel import sample, ALIAS
    daily = tmp_path/"output/tushare_cache/daily"
    daily.mkdir(parents=True)
    (tmp_path/"config").mkdir()
    atomic_json(tmp_path/"config/s20_v4_security_aliases.json", {"aliases": [ALIAS]})
    for date, frame in sample().groupby("trade_date"):
        frame = frame.assign(pre_close=10., vol=1., amount=100.)
        frame.to_parquet(daily/(date+".parquet"), index=False)
    panel = Path(build(tmp_path)["directory"])
    return {"sources": [{"role": "identity_aware_daily_panel", "path": str(panel.relative_to(tmp_path)),
                          "summary_sha256": digest(panel/"summary.json")}]}


def test_fresh_identity_reconstruction_not_just_summary(tmp_path):
    import pandas as pd
    inventory = identity_source(tmp_path)
    records, _ = inspect_sources(tmp_path, inventory, revalidate_identity=True)
    assert records[0]["identity_mapping_verified"]
    assert records[0]["fresh_identity_validation"]["verified_partitions"] == 3
    assert not records[0]["semantic_acceptance"]
    source = tmp_path/"output/tushare_cache/daily/20240102.parquet"
    frame = pd.read_parquet(source)
    frame.loc[0, "close"] = 123.
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_sources(tmp_path, inventory, revalidate_identity=True)


def test_h01_integrates_identity_without_promoting_universe(tmp_path):
    from research.s20_harness.full_data_audit import audit_dataset
    inventory = identity_source(tmp_path)
    atomic_json(tmp_path/"config/s20_v4_data_sources.json", inventory)
    report = audit_dataset(tmp_path, tmp_path/"audit")
    identity = report["support_sources"]["security_identity"]
    assert identity["semantic_validation"] == "validated_identity_mapping"
    assert not identity["PIT_universe_verified"] and not report["formal_gate_passed"]
    assert any("historical universe" in gap for gap in report["acceptance_gaps"])
