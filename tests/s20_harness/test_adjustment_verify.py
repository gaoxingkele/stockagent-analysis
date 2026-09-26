import pandas as pd
import pytest

from research.s20_harness.adjustment_source import collect
from research.s20_harness.adjustment_reconcile import run
from research.s20_harness.adjustment_verify import verify
from research.s20_harness.runtime import digest, atomic_json, load_plan
from research.s20_harness.source_evidence import inspect_sources


def fixture(root, unexplained=False):
    daily = root / "output/tushare_cache/daily"
    daily.mkdir(parents=True)
    for date, price in [("20240102", 10.), ("20240103", 5.)]:
        pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [date],
                      "close": [price], "pre_close": [4. if unexplained and date == "20240103" else price]}).to_parquet(daily / (date + ".parquet"))
    def query(trade_date):
        return pd.DataFrame({"ts_code": ["000001.SZ"], "trade_date": [trade_date],
                             "adj_factor": [1. if trade_date == "20240102" else 2.]})
    collect(root, query, "adjustments-test", pause=0)
    from pathlib import Path
    return Path(run(root, "adjustments-test")["directory"])


def test_fresh_reconstruction_and_source_integration(tmp_path):
    directory = fixture(tmp_path)
    result = verify(tmp_path, directory, digest(directory / "summary.json"))
    assert result["semantic_reconstruction_verified"]
    assert result["raw_discontinuities"] == 1
    assert not result["PIT_verified"] and not result["formal_H01_gate_passed"]
    records, pins = inspect_sources(tmp_path, {"sources": [{"role": "adjustment_factors",
        "path": str(directory.parent), "reconciliation": str(directory / "summary.json")}]},
        revalidate_adjustments=True)
    assert records[0]["fresh_adjustment_validation"]["dates"] == 2
    assert not records[0]["semantic_acceptance"] and pins


def test_false_rehashed_summary_and_changed_table_rejected(tmp_path):
    directory = fixture(tmp_path)
    summary = load_plan(directory / "summary.json")
    summary["raw_discontinuities"] = 0
    atomic_json(directory / "summary.json", summary)
    with pytest.raises(ValueError, match="summary mismatch"):
        verify(tmp_path, directory, digest(directory / "summary.json"))
    summary["raw_discontinuities"] = 1
    atomic_json(directory / "summary.json", summary)
    table = directory / "reference_reconciliation.parquet"
    frame = pd.read_parquet(table)
    frame["factor_implied_reference"] = 99.
    frame.to_parquet(table)
    with pytest.raises(AssertionError):
        verify(tmp_path, directory, digest(directory / "summary.json"))


def test_current_daily_inventory_change_rejected(tmp_path):
    directory = fixture(tmp_path)
    daily = tmp_path / "output/tushare_cache/daily"
    pd.read_parquet(daily / "20240103.parquet").to_parquet(daily / "20240104.parquet")
    with pytest.raises(ValueError, match="coverage changed"):
        verify(tmp_path, directory, digest(directory / "summary.json"))


def test_unexplained_json_key_order_does_not_change_semantics(tmp_path):
    directory = fixture(tmp_path, unexplained=True)
    result = verify(tmp_path, directory, digest(directory / "summary.json"))
    assert result["unexplained_transitions"] == 1
    assert not result["formal_H01_gate_passed"]
