#!/usr/bin/env python
"""Pool S (S20 contract top 20): how did the names with a low pump ratio do?

  python scripts/analyze_s20_low_ratio_names.py

User request 2026-10-30 (idea I0011).  Descriptive, used windows only: signal days 2026-01-27..2026-08-05 (pump scores exist from 2026-01-27;
nothing from 2026-08-06 on is read).  Evidence tier reused_holdout.  Builds on T0044 (ratio bands inside the funnel survivors); this one is on
the final top-20 list and paired within the day.
PRE-REGISTERED:
 1. top-20 names by ratio band (<=1, 1-2, 2-5, >5): n, 5-day start-up, 20d up (+15 first), stop-first (-10 first), per trade (U15D10, cost in ret_v1),
    split at 2026-05-22 (end of the pump model's validation).
 2. same-day pairs inside the top 20: names with ratio < 2 minus names with ratio >= 2 (days with both), per-trade / stop-first / up; NW(19) t.
 3. same with the within-day bottom third of ratio in the top 20 against the rest.
 4. confound check: NATR, industry share, and the same pair after removing the day's mean (within-industry is not possible at n=20; report NATR gap).
 5. stock-clustered view: share of low-ratio rows from the 10 most frequent names; the paired result after dropping those 10 names.
Falsified (I0011) if the paired per-trade difference (ratio<2 minus >=2) is >= 0, or |NW t| < 1.96, or the two segments have opposite signs.
Output: output/experiments/s20_low_ratio_20261030/report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_s20_pump_filter import five_day_labels, pump_scores, PUMP_VALID_END, RESERVED_FROM, TOP_K  # noqa: E402
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

OUT = ROOT / "output/experiments/s20_low_ratio_20261030"
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def row(x: pd.DataFrame, label: str) -> str:
    return (f"{label:34s} n {len(x):5d} | 5d start-up {100 * x.start_up.mean():5.1f}% | 20d up {100 * x.up.mean():5.1f}% "
            f"stop-first {100 * x.stop.mean():5.1f}% per trade {x.ret_v1.mean():+5.2f}")


def paired(a: pd.DataFrame, b: pd.DataFrame, metrics=("ret_v1", "stop", "up")) -> str:
    out = []
    for m in metrics:
        d = (a.groupby("trade_date")[m].mean() - b.groupby("trade_date")[m].mean()).dropna()
        if len(d) > 5:
            out.append(f"{m} {d.mean() * (100 if m != 'ret_v1' else 1):+.2f}{'pp' if m != 'ret_v1' else ''} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, {len(d)} days)")
    return "; ".join(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pump = pump_scores()
    first = pump.trade_date.min()
    frozen = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    frozen = frozen[(frozen.trade_date >= first) & (frozen.trade_date < RESERVED_FROM)]
    surv = select(frozen, PureConfig(top_k=100), "U15D10").merge(pump, on=["ts_code", "trade_date"], how="left")
    out = outcomes()
    out = out[out.trade_date < RESERVED_FROM][["ts_code", "trade_date", "ret_v1", "state"]]
    out["up"] = out.state.isin(["pure_up", "dirty_up"])
    out["stop"] = out.state == "pure_down"
    lab = five_day_labels(set(surv.ts_code), first, RESERVED_FROM)
    s = surv.merge(out, on=["ts_code", "trade_date"]).merge(lab, on=["ts_code", "trade_date"], how="left")
    s = s[s.list_rank <= TOP_K].copy()
    s["seg"] = np.where(s.trade_date <= PUMP_VALID_END, "validation", "out-of-sample")
    natr = [c for c in s.columns if c.lower() in ("natr", "natr14", "natr_14", "t_natr")]
    say(f"pool S top-{TOP_K} rows {len(s)}, days {s.trade_date.nunique()} ({s.trade_date.min()}..{s.trade_date.max()}); with ratio {s.ratio.notna().mean():.1%}; NATR column {natr[:1]}")
    say(f"ratio quantiles in the top 20: {s.ratio.quantile([.1, .25, .5, .75, .9]).round(2).to_dict()}; share < 2: {(s.ratio < 2).mean():.1%}")
    s = s[s.ratio.notna()].copy()
    s["band"] = pd.cut(s.ratio, [-np.inf, 1, 2, 5, np.inf], labels=["<=1", "1-2", "2-5", ">5"])

    say("\n== 1. top-20 names by ratio band ==")
    for seg in ("all", "validation", "out-of-sample"):
        d = s if seg == "all" else s[s.seg == seg]
        say(row(d, f"[{seg}] all top-20"))
        for b, g in d.groupby("band", observed=True):
            say(row(g, f"[{seg}] ratio {b}"))

    s["low"] = s.ratio < 2
    say("\n== 2. same-day pairs inside the top 20: ratio < 2 minus ratio >= 2 ==")
    res = {}
    for seg in ("all", "validation", "out-of-sample"):
        d = s if seg == "all" else s[s.seg == seg]
        diff = (d[d.low].groupby("trade_date").ret_v1.mean() - d[~d.low].groupby("trade_date").ret_v1.mean()).dropna()
        res[seg] = diff.mean()
        say(f"[{seg}] {paired(d[d.low], d[~d.low])}; days with a low name {d[d.low].trade_date.nunique()} of {d.trade_date.nunique()}")

    say("\n== 3. within-day bottom third of ratio in the top 20 vs the rest ==")
    s["bot3"] = s.groupby("trade_date").ratio.rank(pct=True, method="first") <= 1 / 3
    for seg in ("all", "validation", "out-of-sample"):
        d = s if seg == "all" else s[s.seg == seg]
        say(f"[{seg}] {paired(d[d.bot3], d[~d.bot3])}")

    say("\n== 4. confounds ==")
    if natr:
        c = natr[0]
        say(f"NATR mean: ratio<2 {s[s.low][c].mean():.4f}, ratio>=2 {s[~s.low][c].mean():.4f}")
    ind = [c for c in s.columns if c in ("industry", "industry_name")]
    if ind:
        top = s[s.low][ind[0]].value_counts(normalize=True).head(5).round(3).to_dict()
        base = s[~s.low][ind[0]].value_counts(normalize=True).head(5).round(3).to_dict()
        say(f"top industries among ratio<2 rows: {top}; among ratio>=2: {base}")

    say("\n== 5. stock clustering ==")
    freq = s[s.low].ts_code.value_counts()
    say(f"low-ratio rows {int(s.low.sum())} from {freq.size} stocks; the 10 most frequent stocks hold {freq.head(10).sum() / freq.sum():.1%} of them")
    d = s[~s.ts_code.isin(freq.head(10).index)]
    say(f"after dropping those 10 stocks entirely: {paired(d[d.low], d[~d.low])}")
    seg_signs = {k: np.sign(v) for k, v in res.items() if k != "all"}
    say(f"\nfalsification check (I0011): all-sample paired per-trade diff {res['all']:+.2f}; segment signs {seg_signs}")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
