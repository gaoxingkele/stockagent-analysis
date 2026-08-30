#!/usr/bin/env python
"""Build and audit full-history path-dependent labels for S20 research."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.s20 import (  # noqa: E402
    DEFAULT_UPSIDE_PCT,
    build_daily_first_passage_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        default=ROOT / "output/experiments/s20_first_passage_v1",
    )
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="99991231")
    return parser.parse_args()


def load_daily(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    paths = [
        item
        for item in sorted(path.glob("*.parquet"))
        if start_date <= item.stem <= end_date
    ]
    if not paths:
        raise FileNotFoundError(f"no daily parquet files found beneath {path}")
    parts = [
        pd.read_parquet(
            item, columns=["ts_code", "trade_date", "open", "high", "low", "close"]
        )
        for item in paths
    ]
    daily = pd.concat(parts, ignore_index=True)
    daily["ts_code"] = daily["ts_code"].astype(str)
    daily["trade_date"] = daily["trade_date"].astype(str)
    daily = daily[daily["ts_code"].str.endswith((".SH", ".SZ"))].copy()
    duplicates = int(daily.duplicated(["ts_code", "trade_date"]).sum())
    if duplicates:
        raise ValueError(f"daily input has {duplicates} duplicate stock-date rows")
    return daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def build_audit(labels: pd.DataFrame, legacy_path: Path) -> dict:
    audit: dict = {
        "contract_version": "s20-first-passage-v1",
        "rows": len(labels),
        "symbols": int(labels["ts_code"].nunique()),
        "signal_date_min": labels["trade_date"].min(),
        "signal_date_max": labels["trade_date"].max(),
        "horizon_end_date_max": labels["horizon_end_date"].max(),
        "duplicate_keys": int(labels.duplicated(["ts_code", "trade_date"]).sum()),
        "thresholds": {},
    }
    for threshold in DEFAULT_UPSIDE_PCT:
        key = f"{int(threshold)}"
        events = labels[f"event{key}"].value_counts(normalize=True)
        audit["thresholds"][key] = {
            "up_first_rate": float((labels[f"event{key}"] == "up_first").mean()),
            "down_first_rate": float((labels[f"event{key}"] == "down_first").mean()),
            "censored_rate": float((labels[f"event{key}"] == "censored").mean()),
            "ambiguous_rate": float(events.get("ambiguous", 0.0)),
            "window_safe_rate": float(labels[f"target{key}_window_safe"].mean()),
            "late_down_after_up_rate": float(labels[f"late_down_after_up{key}"].mean()),
        }

    if legacy_path.exists():
        legacy = pd.read_parquet(
            legacy_path,
            columns=[
                "ts_code",
                "trade_date",
                "entry_open",
                "max_gain_20",
                "max_dd_20",
                "r20_close",
            ],
        )
        legacy["ts_code"] = legacy["ts_code"].astype(str)
        legacy["trade_date"] = legacy["trade_date"].astype(str)
        overlap = labels.merge(
            legacy,
            on=["ts_code", "trade_date"],
            how="inner",
            suffixes=("_s20", "_legacy"),
        )
        differences = {}
        for column in ["entry_open", "max_gain_20", "max_dd_20", "r20_close"]:
            delta = np.abs(
                overlap[f"{column}_s20"].to_numpy()
                - overlap[f"{column}_legacy"].to_numpy()
            )
            differences[column] = float(delta.max(initial=0.0))
        audit["legacy_compatibility"] = {
            "overlap_rows": len(overlap),
            "max_absolute_differences": differences,
            "tolerances": {
                "entry_open": 1e-8,
                "max_gain_20": 5.1e-5,
                "max_dd_20": 5.1e-5,
                "r20_close": 5.1e-5,
            },
        }
        tolerances = audit["legacy_compatibility"]["tolerances"]
        audit["legacy_compatibility"]["compatible_with_legacy_precision"] = all(
            differences[column] <= tolerances[column] for column in differences
        )
    return audit


def main() -> None:
    args = parse_args()
    daily = load_daily(args.daily_dir, args.start_date, args.end_date)
    parts = []
    groups = daily.groupby("ts_code", sort=True)
    total = daily["ts_code"].nunique()
    for number, (ts_code, group) in enumerate(groups, 1):
        labels = build_daily_first_passage_labels(group)
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
        raise ValueError("S20 output contains duplicate stock-date keys")
    if not audit.get("legacy_compatibility", {}).get(
        "compatible_with_legacy_precision", False
    ):
        raise ValueError("S20 timing does not reproduce legacy extrema labels")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_dir / "first_passage_labels.parquet", index=False)
    (args.output_dir / "label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
