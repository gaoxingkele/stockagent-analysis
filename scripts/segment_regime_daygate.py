#!/usr/bin/env python
"""Follow-up to segment_backtest: the only v1 exclusion candidate is a *day-level* block
(HS300 regime 5 = sideways). Dropping it inside the funnel empties the pool on those days,
so the "gain" is really "do not trade on sideways days". Evaluate it honestly as a day gate:
sleeve = list mean on traded days, 0 on skipped days; dev / confirm / shadow; number of
distinct regime-5 episodes (consecutive runs), worst month, and the same for every regime
so regime 5 is not cherry-picked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s20_pure_history import history_lists, outcomes  # noqa: E402

OUT = ROOT / "output/experiments/segment_backtest"
NAMES = {0: "mixed", 1: "bull_policy", 2: "bull_fast", 3: "bull_slow_diverge", 4: "bear", 5: "sideways"}


def main() -> int:
    L = history_lists()
    L = L[L.rule.isin(["U15D10", "U15D10+B", "safe_v1_1"])]
    P = L.merge(outcomes(), on=["ts_code", "trade_date"])
    rg = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet")
    rg["trade_date"] = rg.trade_date.astype(str)
    regime = rg.set_index("trade_date").regime_id
    day = P.groupby(["rule", "trade_date"]).agg(ret=("ret_safe", "mean"), bad=("cls_a5_d10", lambda s: (s == 2).mean()),
                                                up=("state", lambda s: (s == "pure_up").mean())).reset_index()
    day["regime"] = day.trade_date.map(regime).astype(int)
    day["period"] = np.select([day.trade_date <= "20260126", day.trade_date <= "20260805"], ["dev", "confirm"], "shadow")
    day["month"] = day.trade_date.str[:6]
    lines = []
    # regime calendar: episodes and days
    cal = rg[(rg.trade_date >= "20250303") & (rg.trade_date <= "20260930")].copy()
    cal["period"] = np.select([cal.trade_date <= "20260126", cal.trade_date <= "20260805"], ["dev", "confirm"], "shadow")
    rows = []
    for (per, r), g in cal.groupby(["period", "regime_id"]):
        runs = (g.trade_date.map(lambda d: cal.index[cal.trade_date == d][0]).diff() != 1).sum()
        rows.append({"period": per, "regime": f"{r} {NAMES.get(r, '?')}", "days": len(g), "episodes": int(runs),
                     "months": ", ".join(sorted(g.trade_date.str[:6].unique()))})
    lines.append("## regime calendar (trading days, distinct episodes)\n" + pd.DataFrame(rows).to_string(index=False))

    # per-regime list outcome, every list, every period
    t = day.groupby(["rule", "period", "regime"]).agg(days=("ret", "size"), ret=("ret", "mean"), bad=("bad", "mean"),
                                                     up=("up", "mean")).round(3).reset_index()
    t["regime"] = t.regime.map(lambda r: f"{r} {NAMES.get(r, '?')}")
    lines.append("\n## per-regime sleeve outcome by list and period\n" + t.to_string(index=False))

    # day gate: skip regime k (cash) - every k, so the pick of k=5 is visible among alternatives
    rows = []
    for rule, g in day.groupby("rule"):
        for per, h in g.groupby("period"):
            base = h.ret
            row = {"rule": rule, "period": per, "days": len(h), "none": round(float(base.mean()), 3),
                   "worst_month_none": round(float(base.groupby(h.month).mean().min()), 2)}
            for k in range(6):
                s = base.where(h.regime != k, 0.0)
                row[f"skip{k}"] = round(float(s.mean()), 3)
                if k == 5:
                    row["skip5_days%"] = round(100 * (h.regime == 5).mean(), 1)
                    row["worst_month_skip5"] = round(float(s.groupby(h.month).mean().min()), 2)
                    row["skip5_bad%"] = round(100 * float(h.loc[h.regime != 5, "bad"].mean()), 1)
                    row["none_bad%"] = round(100 * float(h.bad.mean()), 1)
            rows.append(row)
    lines.append("\n## day gate: mean sleeve return (cash on skipped days) when skipping each regime\n"
                 + pd.DataFrame(rows).to_string(index=False))
    text = "\n".join(lines)
    (OUT / "regime_daygate.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
