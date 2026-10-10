#!/usr/bin/env python
"""S20 offensive list + R20 pool A: do the two lists combine well? (registration: wiki/2026-10-10_s20-plus-pool-a.md)

Window 2026-04-14 .. 2026-08-05 signal days (pool A is out of sample for R20 from 04-14; the reserved window
from 08-06 is not read). List basis, +15/-10/20-session exit from the next open, cost 0.3%.
Output: output/experiments/s20_plus_pool_a_20261010/report.txt
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

OUT = ROOT / "output/experiments/s20_plus_pool_a_20261010"
START, END = "20260414", "20260805"


def tstat(d: pd.Series) -> float:
    d = d.dropna()
    return d.mean() / nw_se(d.to_numpy()) if len(d) > 20 else np.nan


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    out = outcomes()
    out = out[(out.trade_date >= START) & (out.trade_date <= END)][["ts_code", "trade_date", "ret_v1", "state"]]
    out["stop"] = (out.state == "pure_down").astype(float)
    out["up"] = out.state.isin(["pure_up", "dirty_up"]).astype(float)
    fr = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    fr = fr[(fr.trade_date >= START) & (fr.trade_date <= END)]
    s20 = select(fr, PureConfig(), "U15D10")[["ts_code", "trade_date", "list_rank"]]
    r = pd.read_parquet(ROOT / "output/r20_history/r20_lists_20260414_20260928.parquet")
    r["trade_date"] = r.trade_date.astype(str)
    pa = r[(r["list"] == "pool_a") & (r.trade_date >= START) & (r.trade_date <= END)][["ts_code", "trade_date", "list_rank"]]
    pa20 = pa[pa.list_rank <= 20]
    days = sorted(set(s20.trade_date))
    lists = {"V0 S20 offensive": s20, "V1 pool A top 20": pa20,
             "V2 union": pd.concat([s20, pa20]).drop_duplicates(["ts_code", "trade_date"]),
             "V4 intersection": s20.merge(pa20[["ts_code", "trade_date"]])}
    daily = {}
    lines = [f"window {START}..{END}: S20 days {len(days)}, pool A days {pa.trade_date.nunique()} (top-20 size median {pa20.groupby('trade_date').size().median():.0f})",
             f"overlap: names in both lists {len(lists['V4 intersection'])} over {lists['V4 intersection'].trade_date.nunique()} days", "",
             "list                 | names  days | per trade  stop-first  up    | daily mean (t)   daily sd  worst month"]
    for name, L in lists.items():
        q = L.merge(out, on=["ts_code", "trade_date"])
        if q.empty:
            lines.append(f"{name:20s} |     0    0 | (no names)")
            continue
        d = q.groupby("trade_date").ret_v1.mean()
        daily[name] = d
        mo = d.groupby(d.index.str[:6]).mean()
        lines.append(f"{name:20s} | {len(q):5d} {q.trade_date.nunique():4d} | {q.ret_v1.mean():+6.2f}   {100 * q.stop.mean():5.1f}%   {100 * q.up.mean():5.1f}% | "
                     f"{d.mean():+6.2f} ({tstat(d):+.1f})   {d.std():5.2f}   {mo.min():+6.2f} ({mo.idxmin()})")
    d0, d1 = daily["V0 S20 offensive"], daily["V1 pool A top 20"]
    v3 = pd.Series({t: (0.5 * d0[t] + 0.5 * d1[t]) if t in d1.index else d0[t] for t in d0.index})
    daily["V3 half-half"] = v3
    mo = v3.groupby(v3.index.str[:6]).mean()
    lines.append(f"{'V3 half-half':20s} | {'':5s} {len(v3):4d} | {'':6s}   {'':5s}    {'':5s}  | {v3.mean():+6.2f} ({tstat(v3):+.1f})   {v3.std():5.2f}   {mo.min():+6.2f} ({mo.idxmin()})")
    both = d0.index.intersection(d1.index)
    lines += ["", f"days with both lists: {len(both)}; correlation of daily returns S20 vs pool A {d0[both].corr(d1[both]):+.2f}",
              f"on those days: S20 {d0[both].mean():+.2f}, pool A {d1[both].mean():+.2f}, half-half {v3[both].mean():+.2f}; "
              f"sd S20 {d0[both].std():.2f}, pool A {d1[both].std():.2f}, half-half {v3[both].std():.2f}", ""]
    for name in ("V2 union", "V3 half-half"):
        diff = (daily[name] - d0).dropna()
        lines.append(f"{name} minus V0, same day: {diff.mean():+.2f} (t {tstat(diff):+.1f}, days {len(diff)}, days that differ {(diff.abs() > 1e-9).sum()})")
    mon = pd.DataFrame({k: v.groupby(v.index.str[:6]).mean() for k, v in daily.items()}).round(2)
    lines += ["", "by month (daily mean):", mon.to_string()]
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
