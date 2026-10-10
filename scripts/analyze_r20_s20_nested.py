#!/usr/bin/env python
"""Nested ranking between R20 and S20 (registration: wiki/2026-10-10_r20-s20-nested-ranking.md).

A: R20 top 100 re-ranked by S20 stage1 -> top 10 / 20.   B: S20 top 100 re-ranked by R20 -> top 10 / 20.
Baselines: R20 top N, S20 top N, the S20 contract list, the market. List basis, +15/-10/20 from the next open.
Output: output/experiments/r20_s20_nested_20261010/report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

OUT = ROOT / "output/experiments/r20_s20_nested_20261010"
SEGMENTS = {"R20 in-sample 0127-0413": ("20260127", "20260413"), "R20 out-of-sample 0414-0805": ("20260414", "20260805")}


def tstat(d: pd.Series) -> float:
    d = d.dropna()
    return d.mean() / nw_se(d.to_numpy()) if len(d) > 20 else np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    r20 = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "output/r20_history/market").glob("*.parquet"))], ignore_index=True)
    r20["trade_date"] = r20.trade_date.astype(str)
    fr = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    fr = fr[(fr.trade_date >= "20260127") & (fr.trade_date <= "20260805")]
    contract = select(fr, PureConfig(), "U15D10")[["ts_code", "trade_date"]].assign(contract=True)
    m = fr.merge(r20[["ts_code", "trade_date", "r20_pred"]], on=["ts_code", "trade_date"]).merge(contract, how="left")
    m["contract"] = m.contract.fillna(False).astype(bool)
    out = outcomes()[["ts_code", "trade_date", "ret_v1", "state"]]
    m = m.merge(out, on=["ts_code", "trade_date"])
    m["stop"] = (m.state == "pure_down").astype(float)
    m["up"] = m.state.isin(["pure_up", "dirty_up"]).astype(float)
    g = m.groupby("trade_date")
    m["rk_r"] = g.r20_pred.rank(ascending=False, method="first")
    m["rk_s"] = g.stage1_probability.rank(ascending=False, method="first")
    rows = {}
    for n in (10, 20):
        rows[f"R20 top{n}"] = m[m.rk_r <= n]
        rows[f"S20 top{n}"] = m[m.rk_s <= n]
        a = m[m.rk_r <= 100].copy()
        a["k"] = a.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
        rows[f"A{n}: R20 top100 by S20"] = a[a.k <= n]
        b = m[m.rk_s <= 100].copy()
        b["k"] = b.groupby("trade_date").r20_pred.rank(ascending=False, method="first")
        rows[f"B{n}: S20 top100 by R20"] = b[b.k <= n]
    rows["S20 contract list"] = m[m.contract]
    rows["market"] = m
    lines = [f"days {m.trade_date.nunique()} ({m.trade_date.min()}..{m.trade_date.max()}), stocks per day ~{g.size().median():.0f}",
             f"rank correlation r20 vs stage1 (mean daily Spearman): {g.apply(lambda d: d.r20_pred.corr(d.stage1_probability, method='spearman')).mean():+.2f}",
             f"names in both top-100s per day: median {m[(m.rk_r <= 100) & (m.rk_s <= 100)].groupby('trade_date').size().reindex(sorted(m.trade_date.unique()), fill_value=0).median():.0f}", ""]
    daily = {}
    for seg, (lo, hi) in SEGMENTS.items():
        lines += [f"== {seg} ==", "list                    | names | per trade  stop-first  up    | daily mean (t)   worst month"]
        for name, q in rows.items():
            q = q[(q.trade_date >= lo) & (q.trade_date <= hi)]
            d = q.groupby("trade_date").ret_v1.mean()
            daily[(seg, name)] = d
            mo = d.groupby(d.index.str[:6]).mean()
            lines.append(f"{name:24s}| {len(q):5d} | {q.ret_v1.mean():+6.2f}   {100 * q.stop.mean():5.1f}%   {100 * q.up.mean():5.1f}% | "
                         f"{d.mean():+6.2f} ({tstat(d):+.1f})   {mo.min():+6.2f}")
        lines.append("same-day paired differences:")
        for x, y in (("A10: R20 top100 by S20", "R20 top10"), ("A20: R20 top100 by S20", "R20 top20"),
                     ("B10: S20 top100 by R20", "S20 top10"), ("B20: S20 top100 by R20", "S20 top20"),
                     ("A20: R20 top100 by S20", "S20 contract list"), ("B20: S20 top100 by R20", "S20 contract list"),
                     ("R20 top20", "S20 contract list")):
            dd = (daily[(seg, x)] - daily[(seg, y)]).dropna()
            lines.append(f"    {x:24s} - {y:18s}: {dd.mean():+.2f} (t {tstat(dd):+.1f}, days {len(dd)})")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
