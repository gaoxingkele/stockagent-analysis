#!/usr/bin/env python
"""Rank order of the R20 (r20_pred) and S20 (frozen stage1) scores vs first-passage outcomes (U15D10, list basis).

  python scripts/analyze_rank_order.py

Day-by-day rank over the whole market, 8 rank buckets, mean success / stop-first / per trade per bucket; top 20 (and top 10 for R20)
minus rank 21-100, same-day paired, Newey-West(19). Evidence tier reused_holdout, descriptive. Signal days >= 2026-08-06 are not read.
Output: output/experiments/rank_order_20261016/report.txt   (wiki/2026-10-16_r20-s20-rank-order-in-pool.md)
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_r20_s20_nested import tstat  # noqa: E402

OUT = ROOT / "output/experiments/rank_order_20261016"
BINS = [0, 10, 20, 50, 100, 200, 500, 1000, 10**6]
LAB = ["1-10", "11-20", "21-50", "51-100", "101-200", "201-500", "501-1000", ">1000"]
L: list[str] = []


def say(s: str = "") -> None:
    print(s)
    L.append(s)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet",
                        columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    o["succ"] = ((o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))).astype(float)
    o["stop"] = ((o.dn10_day > 0) & ((o.up15_day == 0) | (o.dn10_day <= o.up15_day))).astype(float)
    r = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "r20_pred"]) for p in sorted((ROOT / "output/r20_history/market").glob("*.parquet"))])
    r["trade_date"] = r.trade_date.astype(str)
    fz = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet", columns=["ts_code", "trade_date", "stage1_probability"])
    for name, df, col, segs in (("R20", r, "r20_pred", [("IS", "20260127", "20260413"), ("OOS", "20260414", "20260805")]),
                                ("S20 stage1", fz, "stage1_probability", [("S1", "20250303", "20250829"), ("S2", "20260127", "20260805")])):
        df = df.copy()
        df["rk"] = df.groupby("trade_date")[col].rank(ascending=False, method="first")
        df["b"] = pd.cut(df.rk, BINS, labels=LAB)
        for sn, a, b in segs:
            g = df[df.trade_date.between(a, b)].merge(o, on=["ts_code", "trade_date"])
            t = g.groupby(["trade_date", "b"], observed=True)[["succ", "stop", "ret_base"]].mean().groupby("b", observed=True).mean()
            t[["succ", "stop"]] *= 100
            say(f"\n{name} {sn}: days {g.trade_date.nunique()}")
            say(t.round(2).T.to_string())
            for lab, hi in (("top20", 20), ("top10", 10)):
                for k, mult in (("succ", 100), ("stop", 100), ("ret_base", 1)):
                    x = g[g.rk <= hi].groupby("trade_date")[k].mean()
                    m = g[(g.rk > 20) & (g.rk <= 100)].groupby("trade_date")[k].mean()
                    d = (x - m).dropna()
                    say(f"  {lab} minus rank 21-100, {k}: {d.mean() * mult:+.2f} (t {tstat(d):+.1f})")
            h = g[g.rk <= 100]
            for k in ("succ", "ret_base"):
                ic = h.groupby("trade_date").apply(lambda d: d.rk.rank().corr(d[k].rank()), include_groups=False).dropna()
                say(f"  within top 100, rank-vs-{k} correlation {ic.mean():+.3f} (t {tstat(ic):+.1f}); positive = lower-ranked names do better")
    (OUT / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
