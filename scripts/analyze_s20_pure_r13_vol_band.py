#!/usr/bin/env python
"""Round 13: safety and upside share one axis (volatility) - scan the band.

Universe-level volatility band, then stage1 ranking inside it:
  keep names whose daily natr14 percentile (whole scored universe) is in [lo, hi]
  rank by stage1 -> Top20 (optionally industry expand, cap 4)
The frontier over (lo, hi) shows how much "success" each unit of "bad" buys.
Selection rule fixed before running (dev only): among bands whose dev bad rate
and crash15 rate are BELOW the universe, maximise band expectancy; ties ->
higher success. Confirm is descriptive.
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
from stockagent_analysis.s20_pure import _industry_fill  # noqa: E402
from analyze_s20_pure_r11_safe_up import load, metrics  # noqa: E402

OUT = ROOT / "output/experiments/s20_pure_20260928"


def band_list(p: pd.DataFrame, lo: float, hi: float, k: int = 20, industry: str = "none") -> pd.DataFrame:
    q = p[(p.natr_pct >= lo) & (p.natr_pct <= hi)].copy()
    q["list_rank"] = q.groupby("trade_date")["stage1_probability"].rank(ascending=False, method="first")
    if industry == "none":
        return q[q.list_rank <= k]
    q = q[q.list_rank <= 200]
    return pd.concat([_industry_fill(g, k, 4, industry) for _, g in q.groupby("trade_date")], ignore_index=True)


def main() -> int:
    p = load()
    p["natr_pct"] = p.groupby("trade_date")["natr14"].rank(pct=True)
    uni = {per: metrics(g) for per, g in p.groupby("period")}
    lines = ["universe " + str(uni)]
    rows = []
    for lo, hi in itertools.product((0.0, 0.1, 0.2, 0.3, 0.4), (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
        if hi - lo < 0.2:
            continue
        L = band_list(p, lo, hi)
        for per, g in L.groupby("period"):
            rows.append({"lo": lo, "hi": hi, "period": per, **metrics(g)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "r13_band_grid.csv", index=False)
    dev = t[t.period == "dev"]
    ok = dev[(dev.bad < uni["dev"]["bad"]) & (dev.crash15 < uni["dev"]["crash15"])]
    best = ok.sort_values(["band_mean", "success"], ascending=False).iloc[0]
    lo, hi = best.lo, best.hi
    cols = ["lo", "hi", "success", "win", "bad", "crash15", "crash20", "band_mean", "worst_month", "max_ind_share"]
    lines.append("\n## dev frontier\n" + dev.sort_values(["hi", "lo"])[cols].to_string(index=False))
    lines.append("\n## confirm frontier (consumed, descriptive)\n" +
                 t[t.period == "confirm"].sort_values(["hi", "lo"])[cols].to_string(index=False))
    lines.append(f"\nselected on dev: natr band [{lo}, {hi}]")
    rows = []
    for ind in ("none", "expand", "replace"):
        L = band_list(p, lo, hi, industry=ind)
        for per, g in L.groupby("period"):
            rows.append({"industry": ind, "period": per, **metrics(g)})
    lines.append(pd.DataFrame(rows).to_string(index=False))
    # monthly for selected, expand
    L = band_list(p, lo, hi, industry="expand")
    ret = L["ret_a5_b15_d10"] - 0.3
    m = pd.DataFrame({"sel": ret.groupby(L.trade_date.str[:6]).mean(),
                      "sel_bad%": (L.cls_a5_d10 == 2).groupby(L.trade_date.str[:6]).mean() * 100,
                      "univ": (p["ret_a5_b15_d10"] - 0.3).groupby(p.trade_date.str[:6]).mean(),
                      "univ_bad%": (p.cls_a5_d10 == 2).groupby(p.trade_date.str[:6]).mean() * 100}).round(2)
    lines.append("\n## monthly (selected band + industry expand vs universe)\n" + m.to_string())
    lines.append(f"months beating universe: {(m.sel > m.univ).sum()}/{len(m)}; lower bad%: {(m['sel_bad%'] < m['univ_bad%']).sum()}/{len(m)}")
    text = "\n".join(lines)
    (OUT / "r13_vol_band.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
