#!/usr/bin/env python
"""Build and audit the frozen five-target S20-v2 label dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_s20_first_passage_labels import load_daily  # noqa: E402
from stockagent_analysis.s20 import (  # noqa: E402
    S20_V2_TARGET_DRAWDOWN,
    build_daily_s20_v2_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-dir", type=Path, default=ROOT / "output/tushare_cache/daily"
    )
    parser.add_argument(
        "--legacy-labels",
        type=Path,
        default=ROOT / "output/labels/max_gain_labels.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/experiments/s20_v2_labels",
    )
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="99991231")
    return parser.parse_args()


def build_audit(labels: pd.DataFrame, legacy_path: Path) -> dict:
    audit: dict = {
        "contract_version": "s20-v2-five-target-20260903",
        "rows": len(labels),
        "symbols": int(labels["ts_code"].nunique()),
        "signal_date_min": labels["trade_date"].min(),
        "signal_date_max": labels["trade_date"].max(),
        "horizon_end_date_max": labels["horizon_end_date"].max(),
        "duplicate_keys": int(labels.duplicated(["ts_code", "trade_date"]).sum()),
        "targets": {},
    }
    for target, floor in S20_V2_TARGET_DRAWDOWN.items():
        key = f"{int(target)}"
        reasons = labels[f"reason{key}"].value_counts(dropna=False)
        if int(reasons.sum()) != len(labels):
            raise ValueError(f"target {key} reasons do not partition all rows")
        valid = labels[f"positive{key}"] >= 0
        audit["targets"][key] = {
            "drawdown_floor_pct": floor,
            "positive_rate_all": float((labels[f"positive{key}"] == 1).mean()),
            "positive_rate_valid": float(labels.loc[valid, f"positive{key}"].mean()),
            "valid_rows": int(valid.sum()),
            "reason_counts": {str(k): int(v) for k, v in reasons.items()},
            "reason_rates": {
                str(k): float(v / len(labels)) for k, v in reasons.items()
            },
            "pre_target_mae_quantiles_on_positive": {
                str(q): float(
                    labels.loc[
                        labels[f"positive{key}"] == 1, f"pre_target_mae{key}"
                    ].quantile(q)
                )
                for q in (0.05, 0.10, 0.25, 0.50)
            },
        }
    hierarchy_inversions = {}
    targets = [int(value) for value in S20_V2_TARGET_DRAWDOWN]
    for lower, upper in zip(targets, targets[1:]):
        inversion = (labels[f"positive{upper}"] == 1) & (
            labels[f"positive{lower}"] == 0
        )
        hierarchy_inversions[f"positive_{upper}_while_{lower}_negative"] = {
            "count": int(inversion.sum()),
            "rate": float(inversion.mean()),
        }
    audit["cross_target_non_nested_paths"] = hierarchy_inversions

    if legacy_path.exists():
        legacy = pd.read_parquet(
            legacy_path, columns=["ts_code", "trade_date", "entry_open"]
        )
        legacy["ts_code"] = legacy["ts_code"].astype(str)
        legacy["trade_date"] = legacy["trade_date"].astype(str)
        overlap = labels[["ts_code", "trade_date", "entry_open"]].merge(
            legacy,
            on=["ts_code", "trade_date"],
            how="inner",
            suffixes=("_s20", "_legacy"),
            validate="one_to_one",
        )
        difference = np.abs(overlap["entry_open_s20"] - overlap["entry_open_legacy"])
        audit["legacy_entry_compatibility"] = {
            "overlap_rows": len(overlap),
            "max_absolute_difference": float(difference.max()),
            "exact": bool((difference == 0).all()),
        }
    return audit


def main() -> int:
    args = parse_args()
    daily = load_daily(args.daily_dir, args.start_date, args.end_date)
    parts = []
    total = int(daily["ts_code"].nunique())
    for number, (ts_code, group) in enumerate(daily.groupby("ts_code", sort=True), 1):
        labels = build_daily_s20_v2_labels(group)
        if not labels.empty:
            parts.append(labels)
        if number % 500 == 0 or number == total:
            print(f"labeled {number:,}/{total:,} stocks ({ts_code})", flush=True)
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["ts_code", "trade_date"]
    )
    audit = build_audit(result, args.legacy_labels)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if audit["duplicate_keys"]:
        raise ValueError("S20-v2 output contains duplicate keys")
    if not audit.get("legacy_entry_compatibility", {}).get("exact", False):
        raise ValueError("S20-v2 entry timing differs from legacy labels")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_dir / "labels.parquet", index=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
