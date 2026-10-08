#!/usr/bin/env python
"""Write the full S20 funnel survivor pool for every shadow signal day (lists only, no outcomes).

The shadow run (run_s20_pure_v1_shadow.py) keeps only the top 20 per rule in daily_lists_raw.csv.
Shadow rules that change the funnel or filter inside it (S0005 cut 60%, S0012 pump ratio) need the
whole stage1 top 100 with each name's NATR rank in the pool. This script scores the same factor
store with the same frozen stage1 and NATR as the shadow run and writes that pool.

Output: output/experiments/s20_pure_v1_shadow/funnel_pool.parquet
  ts_code, trade_date, name, industry, stage1_probability, pool_rank (1..100), natr14,
  natr_pct_in_pool, survives_cut40 (frozen contract), survives_cut60 (S0005)
Nothing after the signal day is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_s20_pure_v1_shadow import EVAL_START, SHADOW, daily_natr, load_factors  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, _frozen_models, stage1_probability  # noqa: E402


def main() -> int:
    cfg = PureConfig()
    _, _, features = _frozen_models()
    f = load_factors(SHADOW / "factor_groups", features, EVAL_START)
    f["stage1_probability"] = stage1_probability(f)
    f = f.merge(daily_natr(EVAL_START), on=["ts_code", "trade_date"], how="left")
    f["pool_rank"] = f.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    pool = f[f.pool_rank <= cfg.pool_size].copy()
    pool["natr_pct_in_pool"] = pool.groupby("trade_date").natr14.rank(pct=True)
    pool["survives_cut40"] = pool.natr_pct_in_pool.isna() | (pool.natr_pct_in_pool <= 0.60)
    pool["survives_cut60"] = pool.natr_pct_in_pool.isna() | (pool.natr_pct_in_pool <= 0.40)
    cols = ["ts_code", "trade_date", "name", "industry", "stage1_probability", "pool_rank", "natr14",
            "natr_pct_in_pool", "survives_cut40", "survives_cut60"]
    pool = pool[cols].sort_values(["trade_date", "pool_rank"]).reset_index(drop=True)
    pool.to_parquet(SHADOW / "funnel_pool.parquet", index=False)

    # the contract list must be reproduced exactly from this pool
    lists = pd.read_csv(SHADOW / "daily_lists_raw.csv", dtype={"trade_date": str})
    lists = lists[lists.rule == "U15D10"]
    mine = pool[pool.survives_cut40].copy()
    mine["rk"] = mine.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    mine = mine[mine.rk <= cfg.top_k]
    same = mine.merge(lists, on=["ts_code", "trade_date"]).groupby("trade_date").size()
    print(f"funnel pool: {pool.trade_date.nunique()} days ({pool.trade_date.min()}..{pool.trade_date.max()}), {len(pool)} rows")
    print(f"contract top-20 reproduced: mean overlap {same.reindex(mine.trade_date.unique()).fillna(0).mean() / cfg.top_k:.3f} "
          f"(days compared {lists.trade_date.nunique()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
