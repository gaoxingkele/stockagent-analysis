#!/usr/bin/env python
"""Round 3: trading-side asymmetry between up and down.

1. Exit-rule economics: take-profit +U, stop -D, else close at H=20.
2. Timing: how fast does each barrier get hit?
3. Clustering: is a pick's pure_down "its own" or "everyone's that day"?
Descriptive; stage1-scored universe; dev and confirm reported separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_pure_r01_states import OUT, load, states  # noqa: E402

COST = 0.3  # round-trip cost + slippage, %


def exit_return(p: pd.DataFrame, U: int, D: int, H: int = 20) -> np.ndarray:
    up = p[f"up{U}_day"].to_numpy().astype(int)
    dn = p[f"dn{D}_day"].to_numpy().astype(int)
    up = np.where((up > 0) & (up <= H), up, 0)
    dn = np.where((dn > 0) & (dn <= H), dn, 0)
    r = p["ret20"].to_numpy().astype(float)
    r = np.where((up > 0) & ((dn == 0) | (up < dn)), U, r)
    r = np.where((dn > 0) & ((up == 0) | (dn <= up)), -D, r)  # same-day -> assume stop first
    return r - COST


def main() -> int:
    p = load()
    p["period"] = np.where(p["trade_date"] <= "20260126", "dev", "confirm")
    p["rk"] = p.groupby("trade_date")["stage1_score"].rank(ascending=False, method="first")
    lines = []

    lines.append("## 1. exit-rule economics (mean % per trade after 0.3% cost; win = >0)")
    rows = []
    for U, D in ((10, 5), (10, 8), (15, 8), (15, 10), (20, 10), (20, 15), (30, 15)):
        r = exit_return(p, U, D)
        for name, m in (("universe", np.ones(len(p), bool)), ("top20", p.rk.to_numpy() <= 20),
                        ("top50", p.rk.to_numpy() <= 50), ("top100", p.rk.to_numpy() <= 100)):
            for per in ("dev", "confirm"):
                mm = m & (p.period.to_numpy() == per)
                rows.append({"U": U, "D": D, "set": name, "period": per,
                             "mean": round(float(np.nanmean(r[mm])), 2), "win": round(100 * (r[mm] > 0).mean(), 1),
                             "breakeven_hit": round(100 * D / (U + D), 1)})
    lines.append(pd.DataFrame(rows).pivot_table(index=["U", "D", "breakeven_hit"], columns=["set", "period"],
                                                 values=["mean", "win"]).round(2).to_string())
    lines.append("hold-to-close reference (ret20 - cost): " + ", ".join(
        f"{n}/{per} mean {(p.loc[m & (p.period == per), 'ret20'] - COST).mean():+.2f} win "
        f"{100*(p.loc[m & (p.period == per), 'ret20'] > COST).mean():.1f}%"
        for n, m in (("universe", p.rk > 0), ("top20", p.rk <= 20)) for per in ("dev", "confirm")))

    lines.append("\n## 2. timing (U15 D10): median/quantile session of first touch, Top100")
    st = states(p, 15, 10)
    q = p[p.rk <= 100]
    s = st[p.rk <= 100]
    for k, col in (("pure_up", "up15_day"), ("pure_down", "dn10_day")):
        d = q.loc[s == k, col].astype(int)
        lines.append(f"{k}: n={len(d):,}  p25/p50/p75 = {d.quantile(.25):.0f}/{d.median():.0f}/{d.quantile(.75):.0f}"
                     f"  within 5 sessions {100*(d<=5).mean():.1f}%")

    lines.append("\n## 3. clustering: universe same-day state share seen by a Top20 pick with that state")
    p["st"] = st
    for k in ("pure_up", "pure_down"):
        day_share = (p["st"] == k).groupby(p["trade_date"]).mean()
        pick = p[(p.rk <= 20) & (p["st"] == k)]
        ds = pick["trade_date"].map(day_share)
        lines.append(f"{k}: universe mean {100*(p['st']==k).mean():.1f}%;  on days a Top20 pick is {k}, "
                     f"universe {k} share is {100*ds.mean():.1f}% (median {100*ds.median():.1f}%)")
        # concentration: fraction of all Top20 {k} events that fall on the worst/best 20% of days
        top_days = day_share[day_share >= day_share.quantile(0.8)].index
        lines.append(f"   share of Top20 {k} events on the 20% of days with most universe {k}: "
                     f"{100*pick.trade_date.isin(top_days).mean():.1f}%")

    text = "\n".join(lines)
    (OUT / "r03_asymmetry.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
