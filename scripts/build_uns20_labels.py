#!/usr/bin/env python
"""Rebuild the unS20 dataset: 20-session down-first labels, not failed upside."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_s20_first_passage_labels import load_daily  # noqa: E402
from stockagent_analysis.s20 import build_daily_uns20_labels  # noqa: E402

OUT = ROOT / "output/experiments/s20_uns20_20260927"


def main() -> int:
    daily = load_daily(ROOT / "output/tushare_cache/daily", "20240101", "20260126")
    parts = []
    total = int(daily["ts_code"].nunique())
    for number, (ts_code, group) in enumerate(daily.groupby("ts_code", sort=True), 1):
        labels = build_daily_uns20_labels(group)
        if not labels.empty:
            parts.append(labels)
        if number % 500 == 0 or number == total:
            print(f"labeled {number:,}/{total:,} stocks ({ts_code})", flush=True)
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if result.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate unS20 keys")
    resolved = result["down_first20"] >= 0
    audit = {
        "contract": "uns20-down-first-20pct-vs-minus10-20260927",
        "label": "down_first20 = -10% rail before +20%, T+1 open, 20 sessions",
        "not_label": "positive20 == 0",
        "rows": int(len(result)),
        "symbols": int(result["ts_code"].nunique()),
        "signal_date_min": str(result["trade_date"].min()),
        "signal_date_max": str(result["trade_date"].max()),
        "horizon_end_max": str(result["horizon_end_date"].max()),
        "reserved_window_excluded": "trade_date <= 20260126; 20260127+ not labeled",
        "down_first_rate_resolved": float(result.loc[resolved, "down_first20"].mean()),
        "ambiguous_rate": float((result["down_first20"] < 0).mean()),
        "down_any_rate": float(result["down_any20"].mean()),
        "reason_counts": {
            str(k): int(v) for k, v in result["reason20"].value_counts().items()
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT / "labels.parquet", index=False)
    (OUT / "label_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
