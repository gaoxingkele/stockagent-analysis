#!/usr/bin/env python
"""Round 1: what do "pure up / pure down / chop" look like in the data?

Three-state label, every threshold a parameter:
  pure_up(U, M)   : +U touched within H, and the adverse excursion before that
                    touch never reached -M  (M = "tolerated shake-out")
  pure_down(D)    : -D touched within H before +U  (stop line breached first)
  chop            : neither +U nor -D touched within H
  dirty_up        : +U reached first but only after a >= M shake-out
  ambiguous       : +U and -D on the same session (daily bars cannot order them)
Descriptive only. Uses the stage1-scored universe (ST excluded upstream).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/experiments/s20_pure_20260928"
sys.path.insert(0, str(ROOT / "scripts"))


def states(p: pd.DataFrame, U: int, D: int, M: int | None = None, H: int = 20) -> pd.Series:
    up = p[f"up{U}_day"].to_numpy().astype(int)
    dn = p[f"dn{D}_day"].to_numpy().astype(int)
    up = np.where((up > 0) & (up <= H), up, 0)
    dn = np.where((dn > 0) & (dn <= H), dn, 0)
    s = np.full(len(p), "chop", dtype=object)
    s[(up > 0) & ((dn == 0) | (up < dn))] = "up_first"
    s[(dn > 0) & ((up == 0) | (dn < up))] = "pure_down"
    s[(up > 0) & (up == dn)] = "ambiguous"
    if M is not None:
        mm = p[f"dn{M}_day"].to_numpy().astype(int)
        shaken = (mm > 0) & (mm <= up)          # -M touched on/before the +U session
        s[(s == "up_first") & shaken] = "dirty_up"
    s[s == "up_first"] = "pure_up"
    return pd.Series(s, index=p.index)


def load() -> pd.DataFrame:
    panel = pd.read_parquet(OUT / "path_panel.parquet")
    scored = pd.read_parquet(
        ROOT / "output/experiments/s20_20r_stage1_forward_outcomes/scored_with_outcomes.parquet",
        columns=["ts_code", "trade_date", "stage1_score", "period"],
    )
    return scored.merge(panel, on=["ts_code", "trade_date"], how="inner")


def main() -> int:
    p = load()
    p["period"] = np.where(p["trade_date"] <= "20260126", "dev", "confirm")
    lines = [f"rows={len(p):,} days={p.trade_date.nunique()}"]

    # 1. state shares over a threshold grid
    grid = []
    for U in (10, 15, 20):
        for D in (8, 10, 15):
            for M in (None, 5, 8):
                if M is not None and M >= D:
                    continue
                s = states(p, U, D, M)
                sh = s.value_counts(normalize=True)
                grid.append({"U": U, "D": D, "M": M or "-", **{k: round(100 * sh.get(k, 0), 1) for k in
                             ("pure_up", "dirty_up", "pure_down", "chop", "ambiguous")}})
    g = pd.DataFrame(grid)
    lines.append("\n## state shares (% of stage1 universe, H=20)\n" + g.to_string(index=False))

    # 2. how the reference state (U15/D10/M5) varies by day: share of variance that is day-level
    p["st"] = states(p, 15, 10, 5)
    for k in ("pure_up", "pure_down", "chop"):
        x = (p["st"] == k).astype(float)
        day_mean = x.groupby(p["trade_date"]).transform("mean")
        between = day_mean.var() / x.var()
        dm = x.groupby(p["trade_date"]).mean()
        lines.append(f"{k:9s}: mean {100*x.mean():5.1f}%  daily p10/p50/p90 "
                     f"{100*dm.quantile(.1):5.1f}/{100*dm.quantile(.5):5.1f}/{100*dm.quantile(.9):5.1f}  "
                     f"day-level share of variance {100*between:4.1f}%")

    # 3. stage1 decile x state (reference labels), by period
    p["dec"] = p.groupby("trade_date")["stage1_score"].transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False))
    for per in ("dev", "confirm"):
        q = p[p.period == per]
        t = pd.crosstab(q["dec"], q["st"], normalize="index").mul(100).round(1)
        t["pu/(pu+pd)"] = (100 * t["pure_up"] / (t["pure_up"] + t["pure_down"])).round(1)
        lines.append(f"\n## stage1 decile x state, {per} (U15 D10 M5)\n" + t.to_string())

    # 4. Top20 by stage1: state mix and what an oracle 'remove all pure_down' would leave
    p["rk"] = p.groupby("trade_date")["stage1_score"].rank(ascending=False, method="first")
    for k in (20, 50):
        q = p[p.rk <= k]
        sh = q["st"].value_counts(normalize=True).mul(100).round(1)
        lines.append(f"\nTop{k} state mix: " + ", ".join(f"{a}={b}" for a, b in sh.items()))
        lines.append(f"Top{k} ret20>0: {100*(q.ret20>0).mean():.1f}%  mean ret20 {q.ret20.mean():+.2f}%  "
                     f"entry limit-up (unbuyable) {100*q.entry_limit_up.mean():.1f}%")

    text = "\n".join(lines)
    (OUT / "r01_states.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
