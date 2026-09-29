#!/usr/bin/env python
"""Round 8: variable-length lists with a failure-rate target (conformal selection).

Jin & Candes (JMLR 2023) conformal selection, adapted to a daily pool:
  - pool       : stage1 daily Top-K (K=100)
  - score      : stage1_score (any score column works)
  - "failure"  : user-chosen event (pure_down, exit<=0, not pure_up)
  - calibration: pool members of the last W signal days whose 20-session
                 outcome had fully matured before today (gap of 21 sessions)
  - conformal p-value of candidate j: (1 + #{calib failures with score >= s_j}) / (n_calib + 1)
  - Benjamini-Hochberg at level q over today's pool -> today's list (can be empty)
Reports realised failure rate among selected, list length, and day coverage.
Exchangeability is violated by day-level shocks (R06), which is the point of the test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_pure_r01_states import OUT, load, states  # noqa: E402
from analyze_s20_pure_r03_asymmetry import exit_return  # noqa: E402

K, W, GAP = 100, 60, 21


def bh_select(p: np.ndarray, q: float) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, m + 1) / m
    k = (np.nonzero(ok)[0].max() + 1) if ok.any() else 0
    sel = np.zeros(m, bool)
    sel[order[:k]] = True
    return sel


def run(pool: pd.DataFrame, fail_col: str, q: float, score: str = "stage1_score") -> pd.DataFrame:
    days = sorted(pool.trade_date.unique())
    by_day = {d: g for d, g in pool.groupby("trade_date")}
    out = []
    for i, d in enumerate(days):
        cal_days = days[max(0, i - GAP - W + 1): max(0, i - GAP + 1)]
        if len(cal_days) < W:
            continue
        cal = pd.concat([by_day[x] for x in cal_days])
        fs = np.sort(cal.loc[cal[fail_col], score].to_numpy())
        n = len(cal)
        g = by_day[d]
        s = g[score].to_numpy()
        n_fail_above = len(fs) - np.searchsorted(fs, s, side="left")
        pv = (1 + n_fail_above) / (n + 1)
        sel = bh_select(pv, q)
        out.append({"trade_date": d, "n_sel": int(sel.sum()),
                    "n_fail": int(g.loc[sel, fail_col].sum()) if sel.any() else 0,
                    "exit_sum": float(g.loc[sel, "exit"].sum()) if sel.any() else 0.0,
                    "top20_fail": float(g.nsmallest(20, "rk")[fail_col].mean()),
                    "top20_exit": float(g.nsmallest(20, "rk")["exit"].mean())})
    return pd.DataFrame(out)


def main() -> int:
    p = load()
    p["rk"] = p.groupby("trade_date")["stage1_score"].rank(ascending=False, method="first")
    pool = p[p.rk <= K].copy()
    pool["exit"] = exit_return(pool, 15, 10)
    st = states(pool, 15, 10).to_numpy()
    pool["f_pure_down"] = st == "pure_down"
    pool["f_exit_loss"] = pool["exit"] <= 0
    pool["f_not_pure_up"] = st != "pure_up"
    lines = [f"pool = stage1 top{K}; calibration = last {W} matured days (gap {GAP}); rule U15/D10"]
    rows = []
    for fail in ("f_pure_down", "f_exit_loss", "f_not_pure_up"):
        base = pool[fail].mean()
        for q in (0.2, 0.3, 0.4, 0.5):
            r = run(pool, fail, q)
            r["period"] = np.where(r.trade_date <= "20260126", "dev", "confirm")
            for per, g in r.groupby("period"):
                sel = g[g.n_sel > 0]
                rows.append({"failure": fail, "pool_base": round(100 * base, 1), "q": q, "period": per,
                             "days": len(g), "days_with_list": f"{100*len(sel)/len(g):.0f}%",
                             "avg_len(when>0)": round(sel.n_sel.mean(), 1) if len(sel) else 0,
                             "realised_fail": round(100 * sel.n_fail.sum() / max(sel.n_sel.sum(), 1), 1),
                             "worst_day_fail": round(100 * (sel.n_fail / sel.n_sel).max(), 0) if len(sel) else np.nan,
                             "days_fail>q": f"{100*((sel.n_fail/sel.n_sel) > q).mean():.0f}%" if len(sel) else "-",
                             "exit_per_trade": round(sel.exit_sum.sum() / max(sel.n_sel.sum(), 1), 2),
                             "fixed_top20_fail": round(100 * g.top20_fail.mean(), 1),
                             "fixed_top20_exit": round(g.top20_exit.mean(), 2)})
    lines.append(pd.DataFrame(rows).to_string(index=False))
    text = "\n".join(lines)
    (OUT / "r08_conformal_selection.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
