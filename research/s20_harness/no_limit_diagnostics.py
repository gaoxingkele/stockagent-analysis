"""Investigate vendor limit sentinels without authorizing unrestricted fills."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .runtime import atomic_json, digest, now


def classify(frame, basic):
    if basic.ts_code.duplicated().any():
        raise ValueError("listing metadata must have unique codes")
    joined = frame.merge(basic[["ts_code", "list_date"]], on="ts_code", how="left", validate="many_to_one")
    joined["sentinel_99999_0"] = joined.up_limit.eq(99999.99) & joined.down_limit.eq(0)
    joined["listed_same_date"] = joined.list_date.astype("string").eq(joined.trade_date.astype("string")).fillna(False)
    joined["bse_code"] = joined.ts_code.astype(str).str.endswith(".BJ")
    joined["diagnostic"] = "unexplained_invalid_limits"
    candidate = joined.sentinel_99999_0 & joined.listed_same_date & joined.bse_code
    joined.loc[candidate, "diagnostic"] = "consistent_with_BSE_initial_listing_no_limit_candidate"
    joined["unrestricted_trading_authorized"] = False
    joined["official_listing_notice_verified"] = False
    return joined


def run(root, source, basic_path):
    root, source, basic_path = Path(root).resolve(), Path(source).resolve(), Path(basic_path).resolve()
    basic_hash = digest(basic_path)
    basic = pd.read_parquet(basic_path)
    bad, inputs = [], []
    for path in sorted(source.glob("*.parquet")):
        before = digest(path)
        frame = pd.read_parquet(path)
        up = pd.to_numeric(frame.up_limit, errors="coerce")
        down = pd.to_numeric(frame.down_limit, errors="coerce")
        mask = ~(np.isfinite(up) & np.isfinite(down) & up.gt(0) & down.gt(0) & up.ge(down))
        selected = frame.loc[mask].copy()
        selected["source_file"] = path.name
        bad.append(selected)
        if digest(path) != before:
            raise ValueError("limit file changed")
        inputs.append({"path": str(path), "sha256": before})
    if not bad:
        raise ValueError("no source partitions")
    result = classify(pd.concat(bad, ignore_index=True), basic)
    if digest(basic_path) != basic_hash:
        raise ValueError("metadata changed")
    output = root / "output/experiments/s20_safe_v4/sources" / ("no-limit-diagnostic-" + uuid.uuid4().hex)
    output.mkdir(parents=True)
    result.to_parquet(output / "invalid_limit_cases.parquet", index=False)
    atomic_json(output / "inputs.json", {"limits": inputs, "basic_path": str(basic_path), "basic_sha256": basic_hash})
    report = {"at": now(), "directory": str(output), "invalid_rows": len(result),
              "diagnostic_counts": result.diagnostic.value_counts().to_dict(),
              "data_sha256": digest(output / "invalid_limit_cases.parquet"),
              "code_sha256": digest(Path(__file__)), "formal_H01_gate_passed": False,
              "interpretation": "Metadata agreement is an explanatory lead, not effective-rule or auction evidence"}
    atomic_json(output / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("basic", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[2], args.source, args.basic), indent=2))
