"""Consumer verification of a pinned identity-panel manifest and its lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .runtime import digest, load_plan
from .security_identity import canonicalize_aliases


def _within(base, relative):
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("manifest path escapes allowed root")
    return path


def verify(root: Path, panel: Path, expected_summary_sha256: str) -> dict:
    root, panel = Path(root).resolve(), Path(panel).resolve()
    summary_path = panel / "summary.json"
    if digest(summary_path) != expected_summary_sha256:
        raise ValueError("pinned summary hash mismatch")
    summary = load_plan(summary_path)
    config = root / "config/s20_v4_security_aliases.json"
    if digest(config) != summary["config_sha256"]:
        raise ValueError("identity config no longer matches panel")
    aliases = load_plan(config)["aliases"]
    if digest(Path(__file__).with_name("security_identity.py")) != summary["identity_code_sha256"]:
        raise ValueError("identity algorithm changed")
    if digest(Path(__file__).with_name("identity_panel.py")) != summary["code_sha256"]:
        raise ValueError("panel builder changed")
    receipt_path = panel / "inputs_outputs.json"
    if digest(receipt_path) != summary["receipt_sha256"]:
        raise ValueError("receipt hash mismatch")
    receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipts, list):
        raise ValueError("receipt collection must be a list")
    if len({r["source"] for r in receipts}) != len(receipts):
        raise ValueError("duplicate source receipt")
    actual_sources = {p.relative_to(root).as_posix() for p in (root / "output/tushare_cache/daily").glob("*.parquet")}
    if actual_sources != {r["source"] for r in receipts}:
        raise ValueError("daily source inventory changed")
    totals = dict(raw_rows=0, canonical_rows=0, conflict_groups=0, duplicate_entity_rows=0)
    for receipt in receipts:
        source = _within(root, receipt["source"])
        canonical_path = _within(panel, receipt["canonical"])
        lineage_path = panel / (source.stem + "-lineage.parquet")
        conflicts_path = panel / (source.stem + "-conflicts.parquet")
        files = [(source, "source_sha256"), (canonical_path, "canonical_sha256"),
                 (lineage_path, "lineage_sha256"), (conflicts_path, "conflicts_sha256")]
        for path, field in files:
            if digest(path) != receipt[field]:
                raise ValueError("partition or source hash mismatch")
        raw = pd.read_parquet(source)
        raw["trade_date"] = raw.trade_date.astype(str)
        raw["daily_source_row"] = range(len(raw))
        raw["daily_source_file"] = source.name
        rebuilt, lineage, conflicts = canonicalize_aliases(raw, aliases)
        observed = pd.read_parquet(canonical_path)
        for expected, path in [(rebuilt, canonical_path), (lineage, lineage_path), (conflicts, conflicts_path)]:
            pd.testing.assert_frame_equal(expected.reset_index(drop=True), pd.read_parquet(path).reset_index(drop=True),
                                          check_dtype=False, check_column_type=False, check_exact=True)
        counts = dict(raw_rows=len(raw), canonical_rows=len(observed), conflict_groups=len(conflicts),
                      duplicate_entity_rows=int(observed.duplicated(["entity_id", "trade_date"], keep=False).sum()))
        for key, value in counts.items():
            if value != receipt[key]:
                raise ValueError("receipt count mismatch")
            totals[key] += value
        for path, field in files:
            if digest(path) != receipt[field]:
                raise ValueError("input changed during verification")
    if len(receipts) != summary["daily_files"] or any(totals[k] != summary[k] for k in totals):
        raise ValueError("summary count mismatch")
    if digest(summary_path) != expected_summary_sha256 or digest(config) != summary["config_sha256"]:
        raise ValueError("manifest or config changed during verification")
    return {"verified_partitions": len(receipts), **totals, "semantic_reconstruction_verified": True,
            "summary_sha256": expected_summary_sha256, "formal_training_eligible": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", type=Path)
    parser.add_argument("--summary-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[2], args.panel, args.summary_sha256), indent=2))
