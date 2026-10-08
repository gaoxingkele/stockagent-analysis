#!/usr/bin/env python
"""Score the development months with the frozen S20 stage1 model, for the full market.

The published development results were scored by three walk-forward fold models on a 50% row
sample. This writes what the frozen model (the one applied from 2026-01-27 on) would have listed
on the same days: output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet with
ts_code, trade_date, industry, stage1_probability, natr14 for 2025-03-03 .. 2026-08-05.

Reading aid for the dev part: the frozen residual boosters were fitted on 2025-09-01..10-31 and
early-stopped on 2025-11-01..2026-01-26; the two-tree anchor was refitted on all matured dev rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_pure_r09_funnel import natr14  # noqa: E402
from s20_frames import BASE_STORE, load_frame  # noqa: E402
from stockagent_analysis.s20_pure import _frozen_models, stage1_probability  # noqa: E402

OUT = ROOT / "output/experiments/s20_frozen_rescore_20261005"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, _, features = _frozen_models()
    dev = load_frame(BASE_STORE, "20250303", "20260126", features)
    dev["stage1_probability"] = stage1_probability(dev)
    print(f"dev full market: rows {len(dev):,} days {dev.trade_date.nunique()} rows/day {len(dev) / dev.trade_date.nunique():.0f}", flush=True)

    # the sampled rows scored earlier must agree exactly
    old = pd.read_parquet(ROOT / "output/experiments/s20_composition_diag_20261005/scores_dev.parquet",
                          columns=["ts_code", "trade_date", "p_frozen"])
    m = dev[["ts_code", "trade_date", "stage1_probability"]].merge(old, on=["ts_code", "trade_date"])
    print(f"check against the sampled re-score: rows {len(m):,} max abs diff {np.abs(m.stage1_probability - m.p_frozen).max():.2e}", flush=True)

    conf = pd.read_parquet(ROOT / "output/experiments/s20_20r_confirmation/run_v1/predictions.parquet",
                           columns=["ts_code", "trade_date", "stage1_probability"])
    conf["trade_date"] = conf["trade_date"].astype(str)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "industry"])
    conf = conf.merge(basic, on="ts_code", how="left")
    both = pd.concat([dev[["ts_code", "trade_date", "industry", "stage1_probability"]], conf], ignore_index=True)
    nat = natr14().rename(columns={"natr14_self": "natr14"})
    nat["trade_date"] = nat["trade_date"].astype(str)
    both = both.merge(nat, on=["ts_code", "trade_date"], how="left")
    both.to_parquet(OUT / "frozen_scores.parquet", index=False)
    print(f"wrote frozen_scores.parquet rows {len(both):,} days {both.trade_date.nunique()} {both.trade_date.min()}..{both.trade_date.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
