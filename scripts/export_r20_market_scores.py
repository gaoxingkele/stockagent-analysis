#!/usr/bin/env python
"""Full-market R20 scores per day with the production V12 scorer (resumable, one parquet per day).

  python scripts/export_r20_market_scores.py --start 20260127 --end 20260805

Writes output/r20_history/market/<date>.parquet with ts_code, r20_pred, buy_r20_score, pred_max_gain_20,
pred_max_dd_20, pump_score, pump_down_score. Days whose feature groups lack rows for the day itself are
skipped and listed (the loader would otherwise use the latest day of that file). Signal-day scores only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from export_pump_history import stale_groups  # noqa: E402

OUT = ROOT / "output/r20_history/market"
KEEP = ["ts_code", "r20_pred", "buy_r20_score", "pred_max_gain_20", "pred_max_dd_20", "pump_score", "pump_down_score"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    from stockagent_analysis.v12_scoring import V12Scorer
    OUT.mkdir(parents=True, exist_ok=True)
    scorer = V12Scorer.get(ROOT)
    days = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet") if a.start <= p.stem <= a.end)
    skipped = []
    for i, d in enumerate(days, 1):
        path = OUT / f"{d}.parquet"
        if path.exists():
            continue
        stale = stale_groups(scorer, d)
        if stale:
            skipped.append((d, stale))
            print(f"[{i}/{len(days)}] {d} skipped: no rows for {stale}", flush=True)
            continue
        t0 = time.time()
        df = scorer.score_market(d)
        if df["pump_score"].isna().all():
            scorer.predict_one(df, "r5_pump_3way")
            df = scorer.apply_pump_3way(df)
        out = df[[c for c in KEEP if c in df.columns]].copy()
        out.insert(1, "trade_date", d)
        out.to_parquet(path, index=False)
        print(f"[{i}/{len(days)}] {d}: {len(out)} stocks ({time.time() - t0:.0f}s)", flush=True)
    print(f"done; skipped {len(skipped)} day(s): {skipped[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
