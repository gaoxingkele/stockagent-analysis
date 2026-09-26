"""Compare reference-price discontinuities with factor ratios, not cash profits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .runtime import atomic_json, digest, load_plan, now


def reconcile_panel(panel):
    ordered = panel.sort_values(["ts_code", "trade_date"]).copy()
    prior = ordered.groupby("ts_code", sort=False)[["close", "adj_factor", "trade_date"]].shift()
    previous_close = prior["close"]
    previous_factor = prior["adj_factor"]
    tolerance = np.maximum(.011, previous_close.abs() * .001)
    raw_error = (ordered.pre_close - previous_close).abs()
    expected_reference = previous_close * previous_factor / ordered.adj_factor
    adjusted_error = (ordered.pre_close - expected_reference).abs()
    valid_factors = np.isfinite(previous_factor) & previous_factor.gt(0) & np.isfinite(ordered.adj_factor) & ordered.adj_factor.gt(0)
    ordered["previous_observed_date"] = prior["trade_date"]
    ordered["previous_close"] = previous_close
    ordered["previous_factor"] = previous_factor
    ordered["raw_reference_error"] = raw_error
    ordered["factor_implied_reference"] = expected_reference
    ordered["adjusted_reference_error"] = adjusted_error
    ordered["raw_discontinuity"] = raw_error.gt(tolerance)
    ordered["factor_reconciliation"] = np.select(
        [previous_close.isna(), ~valid_factors, adjusted_error.le(tolerance)],
        ["first_observation", "factor_missing", "consistent_with_factor_ratio"], default="unexplained")
    triggers = ordered.loc[ordered.raw_discontinuity].copy()
    summary = {"rows": len(ordered), "raw_discontinuities": len(triggers),
               "trigger_resolution_counts": triggers.factor_reconciliation.value_counts().to_dict(),
               "missing_factor_rows": int((~np.isfinite(ordered.adj_factor) | ordered.adj_factor.le(0)).sum()),
               "all_transition_status_counts": ordered.factor_reconciliation.value_counts().to_dict(),
               "unexplained_transitions": ordered.loc[ordered.factor_reconciliation.eq("unexplained")].to_dict("records"),
               "interpretation": "ratio consistency does not prove event identity, historical availability, tax, cash or shares"}
    return triggers, summary


def run(root, source_id):
    if Path(source_id).name != source_id or not source_id.startswith("adjustments-"):
        raise ValueError("invalid source id")
    source = root / "output/experiments/s20_safe_v4/sources" / source_id
    collection = load_plan(source / "collection_summary.json")
    if not collection["complete_acquisition"] or not collection["all_schema_valid"]:
        raise ValueError("complete schema-valid acquisition required; never mistake partial coverage for full audit")
    plan = load_plan(source / "collection_plan.json")
    frames, inputs = [], []
    for item in plan["daily"]:
        daily = root / item["path"]
        receipt_path = source / f"{item['date']}.json"
        receipt = load_plan(receipt_path)
        factors = source / receipt["file"]
        if factors.parent.resolve() != source.resolve():
            raise ValueError("unsafe factor file path")
        if digest(daily) != item["sha256"] or digest(factors) != receipt["sha256"]:
            raise ValueError("input changed after acquisition")
        prices = pd.read_parquet(daily, columns=["ts_code", "trade_date", "close", "pre_close"])
        adjustment = pd.read_parquet(factors)
        prices["trade_date"] = prices.trade_date.astype(str)
        adjustment["trade_date"] = adjustment.trade_date.astype(str)
        frames.append(prices.merge(adjustment, on=["ts_code", "trade_date"], how="left", validate="one_to_one"))
        inputs.append({"daily": item, "factor_file": factors.name, "factor_sha256": receipt["sha256"],
                       "receipt_sha256": digest(receipt_path)})
    triggers, summary = reconcile_panel(pd.concat(frames, ignore_index=True))
    destination = source / ("reconcile-" + uuid.uuid4().hex)
    destination.mkdir()
    triggers.to_parquet(destination / "reference_reconciliation.parquet", index=False)
    summary.update({"source_id": source_id, "at": now(), "directory": str(destination), "formal_H01_gate_passed": False,
                    "by_market": {market: frame.factor_reconciliation.value_counts().to_dict()
                                  for market, frame in triggers.groupby(triggers.ts_code.str.rsplit(".", n=1).str[-1])}})
    atomic_json(destination / "summary.json", summary)
    atomic_json(destination / "inputs.json", {"files": inputs, "code_hash": digest(Path(__file__))})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parents[2], args.source_id), ensure_ascii=False, indent=2))
