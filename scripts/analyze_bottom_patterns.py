#!/usr/bin/env python
"""Backtest of the bottom patterns inside the S20 and R20 top-100 pools (wiki/2026-10-12_bottom-pattern-lab.md).

  python scripts/analyze_bottom_patterns.py

Pools: S20 = frozen stage1 top 100 (S1 2025-03-03..08-29, S2 2026-01-27..08-05); R20 = production V12 r20_pred
top 100 (in-sample 01-27..04-13, out-of-sample 04-14..08-05); market = every stock, as the reference.
Outcome: list-basis U15D10 (cost 0.3) plus 5/10/20-day returns and stop-first share. For each pattern: hit vs
not-hit inside the pool, and the same-day paired difference (mean of hit names minus the pool mean, days with
a hit), Newey-West(19) t. Composite rule: pool names with bottom score >= 40, top 20 by score, vs the pool's own
top 20. Signal days >= 2026-08-06 are not used. Output: output/experiments/bottom_patterns_20261012/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_r20_s20_nested import tstat  # noqa: E402
from bottom_patterns import FLAGS, pattern_panel  # noqa: E402

OUT = ROOT / "output/experiments/bottom_patterns_20261012"
SEGS = {"S20": {"S1 2025-03..08": ("20250303", "20250829"), "S2 2026-01-27..08-05": ("20260127", "20260805")},
        "R20": {"R20 in-sample 01-27..04-13": ("20260127", "20260413"), "R20 OOS 04-14..08-05": ("20260414", "20260805")},
        "market": {"S1 2025-03..08": ("20250303", "20250829"), "S2 2026-01-27..08-05": ("20260127", "20260805")}}
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def stop_first(q: pd.DataFrame) -> float:
    return 100 * ((q.dn10_day > 0) & ((q.up15_day == 0) | (q.dn10_day <= q.up15_day))).mean()


def pools() -> dict[str, pd.DataFrame]:
    fz = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet",
                         columns=["ts_code", "trade_date", "stage1_probability"])
    fz["own"] = fz.stage1_probability
    fz["rank"] = fz.groupby("trade_date").own.rank(ascending=False, method="first")
    r20 = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "r20_pred"])
                     for p in sorted((ROOT / "output/r20_history/market").glob("*.parquet"))], ignore_index=True)
    r20["trade_date"] = r20.trade_date.astype(str)
    r20["own"] = r20.r20_pred
    r20["rank"] = r20.groupby("trade_date").own.rank(ascending=False, method="first")
    return {"S20": fz[fz["rank"] <= 100][["ts_code", "trade_date", "own", "rank"]],
            "R20": r20[r20["rank"] <= 100][["ts_code", "trade_date", "own", "rank"]]}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "patterns.parquet"
    if cache.exists():
        pp = pd.read_parquet(cache)
    else:
        pp = pattern_panel("20240801", "20261009", out_from="20250101")
        pp.to_parquet(cache, index=False)
    pp = pp[pp.trade_date <= "20260805"]
    pp["score40"] = pp.bottom_score >= 40
    paths = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet",
                            columns=["ts_code", "trade_date", "ret_base", "ret20", "up15_day", "dn10_day"])
    p5 = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/path_panel.parquet", columns=["ts_code", "trade_date", "ret5", "ret10"])
    o = paths.merge(p5, on=["ts_code", "trade_date"], how="left")
    P = pools()
    P["market"] = pp[["ts_code", "trade_date"]].assign(own=np.nan, rank=np.nan)
    flags = FLAGS + ["B2c_first", "score40"]
    say(f"pattern panel {pp.trade_date.min()}..{pp.trade_date.max()}, rows {len(pp):,}")
    say("market hit rates: " + ", ".join(f"{k} {100 * pp[k].mean():.1f}%" for k in flags + ["zone"]))
    for pname, segs in SEGS.items():
        base = P[pname].merge(pp, on=["ts_code", "trade_date"]).merge(o, on=["ts_code", "trade_date"])
        for sname, (a, b) in segs.items():
            g = base[base.trade_date.between(a, b)]
            say(f"\n==== {pname} pool, {sname}: days {g.trade_date.nunique()}, names {len(g):,}, per trade {g.ret_base.mean():+.2f}, "
                f"stop-first {stop_first(g):.1f}%, in bottom zone {100 * g.zone.mean():.0f}%")
            say(f"{'pattern':8s} | hits  share | per trade  stop-first  ret5   ret10  ret20 | not-hit per trade | paired hit-pool (t, days) | in zone: hits, per trade")
            pool_day = g.groupby("trade_date").ret_base.mean()
            for f in flags:
                h, nh = g[g[f]], g[~g[f]]
                if len(h) == 0:
                    say(f"{f:8s} |     0")
                    continue
                d = (h.groupby("trade_date").ret_base.mean() - pool_day).dropna()
                hz = h[h.zone]
                say(f"{f:8s} | {len(h):5d} {100 * len(h) / len(g):4.1f}% | {h.ret_base.mean():+7.2f}  {stop_first(h):6.1f}%  "
                    f"{h.ret5.mean():+5.2f} {h.ret10.mean():+6.2f} {h.ret20.mean():+6.2f} | {nh.ret_base.mean():+7.2f} | "
                    f"{d.mean():+.2f} ({tstat(d):+.1f}, {len(d)}) | {len(hz)}, {hz.ret_base.mean():+.2f}")
            if pname != "market":
                top = g[g["rank"] <= 20]
                sel = g[g.score40].sort_values(["trade_date", "bottom_score", "own"], ascending=[True, False, False])
                sel = sel.groupby("trade_date").head(20)
                d = (sel.groupby("trade_date").ret_base.mean() - top.groupby("trade_date").ret_base.mean()).dropna()
                say(f"composite: pool names with bottom score >= 40, top 20 by score: {len(sel)} names on {sel.trade_date.nunique()} days, "
                    f"per trade {sel.ret_base.mean():+.2f}, stop-first {stop_first(sel):.1f}% | pool own top 20 per trade {top.ret_base.mean():+.2f} "
                    f"| paired {d.mean():+.2f} (t {tstat(d):+.1f}, days {len(d)})")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
