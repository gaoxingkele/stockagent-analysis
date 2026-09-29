#!/usr/bin/env python
"""Round 11: re-target the funnel on "safe and up", with a take-profit band.

Outcome per name (a = lower edge of the profit band, D = crash line, 20 sessions):
  win        +a touched before -D
  small win  neither touched, day-20 close > 0     (counts as success)
  small loss neither touched, day-20 close <= 0
  bad        -D touched before +a                  (the thing that must not happen)
Also: crash15 / crash20 = 20-session max drawdown <= -15% / -20% at any time.
Exit = band rule (half at +a, stop to entry, rest at +b / entry / day-20 close).

Levers (all naive, no new model): pool size, amplitude cap, industry handling.
Selection rule fixed BEFORE running (dev period only):
  minimise bad rate, subject to success >= v1 and band expectancy >= v1;
  ties -> fewer changes from v1. Confirm window is reported, never used to pick.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stockagent_analysis.s20_pure import PureConfig, ExitRule, select  # noqa: E402
from analyze_s20_pure_r09_funnel import natr14  # noqa: E402

OUT = ROOT / "output/experiments/s20_pure_20260928"
A, B, D, COST = 5, 15, 10, 0.3


def load() -> pd.DataFrame:
    s1 = pd.read_parquet(ROOT / "output/experiments/s20_20r_stage1_forward_outcomes/scored_with_outcomes.parquet",
                         columns=["ts_code", "trade_date", "stage1_score"]).rename(columns={"stage1_score": "stage1_probability"})
    band = pd.read_parquet(OUT / "band_panel.parquet")
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "industry"])
    p = s1.merge(band, on=["ts_code", "trade_date"]).merge(natr14(), on=["ts_code", "trade_date"], how="left")
    p = p.rename(columns={"natr14_self": "natr14"}).merge(basic, on="ts_code", how="left")
    p["period"] = np.where(p.trade_date <= "20260126", "dev", "confirm")
    return p


def metrics(q: pd.DataFrame, a=A, b=B, d=D) -> dict:
    cls = q[f"cls_a{a}_d{d}"]
    ret = q[f"ret_a{a}_b{b}_d{d}"] - COST
    month = q.trade_date.str[:6]
    return {"days": q.trade_date.nunique(), "avg_len": round(len(q) / max(q.trade_date.nunique(), 1), 1),
            "success": round(100 * cls.isin([1, 3]).mean(), 1), "win": round(100 * (cls == 1).mean(), 1),
            "bad": round(100 * (cls == 2).mean(), 1),
            "crash15": round(100 * (q.maxdd20 <= -15).mean(), 1), "crash20": round(100 * (q.maxdd20 <= -20).mean(), 1),
            "band_mean": round(float(ret.mean()), 2),
            "worst_month": round(float(ret.groupby(month).mean().min()), 2),
            "max_ind_share": round(100 * float(q.groupby("trade_date")["industry"].agg(
                lambda s: s.value_counts(normalize=True).iloc[0] if len(s) else 0).mean()), 1)}


def main() -> int:
    p = load()
    lines = [f"band a={A} b={B} D={D}, cost {COST}%; rows {len(p):,}"]
    ref = {}
    for per, g in p.groupby("period"):
        ref[per] = metrics(g)
    lines.append("universe: " + str(ref))

    rows = []
    for P, c, ind in itertools.product((50, 100, 200), (0.0, 0.2, 0.4, 0.6, 0.8), ("none", "expand", "replace")):
        cfg = PureConfig(pool_size=P, rules=(ExitRule("X", 15.0, 10.0, c),), primary_rule="X")
        L = select(p, cfg, industry_cap=None if ind == "none" else 4, industry_mode=ind if ind != "none" else "expand",
                   natr_col="natr14")
        for per, g in L.groupby("period"):
            rows.append({"pool": P, "cap": c, "industry": ind, "period": per, **metrics(g)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "r11_grid.csv", index=False)
    dev = t[t.period == "dev"].set_index(["pool", "cap", "industry"])
    v1 = dev.loc[(100, 0.4, "none")]
    ok = dev[(dev.success >= v1.success) & (dev.band_mean >= v1.band_mean)]
    changes = lambda k: (k[0] != 100) + (k[1] != 0.4) + (k[2] != "none")  # noqa: E731
    best = sorted(ok.index, key=lambda k: (ok.loc[k, "bad"], changes(k)))[0]
    lines.append(f"\nv1 on dev: {dict(v1)}")
    lines.append(f"selected on dev: pool={best[0]} cap={best[1]} industry={best[2]} -> {dict(dev.loc[best])}")
    conf = t[t.period == "confirm"].set_index(["pool", "cap", "industry"])
    lines.append(f"same configs on confirm (consumed, descriptive): v1 {dict(conf.loc[(100, 0.4, 'none')])}")
    lines.append(f"                                               sel {dict(conf.loc[best])}")
    cols = ["pool", "cap", "industry", "avg_len", "success", "win", "bad", "crash15", "crash20", "band_mean",
            "worst_month", "max_ind_share"]
    lines.append("\n## dev grid (sorted by bad)\n" + t[t.period == "dev"].sort_values("bad")[cols].head(25).to_string(index=False))
    lines.append("\n## confirm grid, same order keys\n" + t[t.period == "confirm"].sort_values("bad")[cols].head(15).to_string(index=False))

    # band-width sensitivity for the selected config (dev and confirm)
    cfg = PureConfig(pool_size=best[0], rules=(ExitRule("X", 15.0, 10.0, best[1]),), primary_rule="X")
    L = select(p, cfg, industry_cap=None if best[2] == "none" else 4, industry_mode=best[2] if best[2] != "none" else "expand")
    rows = []
    for a, b, d in itertools.product((3, 5, 8), (10, 15, 20), (8, 10, 12, 15)):
        for per, g in L.groupby("period"):
            m = metrics(g, a, b, d)
            rows.append({"a": a, "b": b, "D": d, "period": per, "success": m["success"], "bad": m["bad"],
                         "band_mean": m["band_mean"], "worst_month": m["worst_month"]})
    bw = pd.DataFrame(rows).pivot_table(index=["a", "b", "D"], columns="period",
                                        values=["success", "bad", "band_mean", "worst_month"])
    lines.append("\n## band sensitivity for the selected config\n" + bw.round(2).to_string())
    text = "\n".join(lines)
    (OUT / "r11_safe_up.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
