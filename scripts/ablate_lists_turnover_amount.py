#!/usr/bin/env python
"""List-level ablation of the turnover / trading-value tilt, plus capacity, no retraining.

Days with stage1 out-of-sample scores: dev folds + confirm (2025-03..2026-08-05).
S20 v1 funnel = stage1 Top100 pool -> drop the pool's highest-natr 40% -> Top20.
Variants (applied inside the same funnel):
  base            the frozen v1 funnel
  turn_cap30      drop the pool's top 30% by turnover before the natr cap
  turn_low50      keep only the pool's lower half by turnover
  turn_balanced   after the natr cap, take the best-ranked names from each turnover tercile (7/7/6)
  amt>=1e8/3e8/5e8 universe filter: 20-session average trading value >= 100m / 300m / 500m CNY
Also the safe list (v1.1) under the same capacity filters.
Every variant reports raw per-trade (band exit), bad rate, success, and the style-matched
excess (same day x size x EP x turnover terciles, as in style_audit.py) with a month-block CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stockagent_analysis.s20_pure import PureConfig, SafeConfig, select_safe  # noqa: E402
from analyze_s20_pure_r11_safe_up import load  # noqa: E402
from style_audit import characteristics  # noqa: E402

OUT = ROOT / "output/experiments/ablation_volume"
COST = 0.3


def funnel(p: pd.DataFrame, variant: str, top_k: int = 20, pool: int = 100, cap: float = 0.4) -> pd.DataFrame:
    f = p[p.stage1_probability.notna()].copy()
    if variant.startswith("amt>="):
        f = f[f.amount20 >= float(variant[5:]) / 1000]             # daily amount is in thousand CNY
    f["pr"] = f.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    f = f[f.pr <= pool].copy()
    f["turn_in_pool"] = f.groupby("trade_date").turnover_rate.rank(pct=True)
    if variant == "turn_cap30":
        f = f[f.turn_in_pool.isna() | (f.turn_in_pool <= 0.70)]
    elif variant == "turn_low50":
        f = f[f.turn_in_pool.isna() | (f.turn_in_pool <= 0.50)]
    f["natr_in_pool"] = f.groupby("trade_date").natr14.rank(pct=True)
    f = f[f.natr_in_pool.isna() | (f.natr_in_pool <= 1 - cap)].copy()
    f["lr"] = f.groupby("trade_date").stage1_probability.rank(ascending=False, method="first")
    if variant == "turn_balanced":
        f["t3"] = pd.cut(f.turn_in_pool, [0, 1 / 3, 2 / 3, 1], labels=False, include_lowest=True)
        f["r3"] = f.groupby(["trade_date", "t3"]).stage1_probability.rank(ascending=False, method="first")
        quota = {0: 7, 1: 7, 2: 6}
        return f[f.r3 <= f.t3.map(quota)]
    return f[f.lr <= top_k]


def score(q: pd.DataFrame) -> dict:
    day = q.groupby("trade_date")
    ex = day.excess_matched.mean()
    months = ex.index.str[:6]
    uniq = np.unique(months)
    rng = np.random.default_rng(7)
    boots = [pd.concat([ex[months == m] for m in rng.choice(uniq, len(uniq))]).mean() for _ in range(1000)]
    return {"days": q.trade_date.nunique(), "avg_len": round(len(q) / q.trade_date.nunique(), 1),
            "per_trade%": round(float(day.ret.mean().mean()), 2),
            "success%": round(100 * q.cls_a5_d10.isin([1, 3]).mean(), 1),
            "bad%": round(100 * (q.cls_a5_d10 == 2).mean(), 1),
            "mv_pct": round(float(q.mv_pct.mean()), 2), "turn_pct": round(float(q.turn_pct.mean()), 2),
            "amount_pct": round(float(q.amount_pct.mean()), 2),
            "style_matched_pp": round(float(ex.mean()), 2),
            "CI95": (round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)),
            "worst_month%": round(float(day.ret.mean().groupby(ex.index.str[:6]).mean().min()), 2)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p = load()                                   # stage1 scores + band outcomes + natr14 + industry
    p = p[p.trade_date <= "20260805"]
    C = characteristics(set(p.trade_date))
    p = p.merge(C[["ts_code", "trade_date", "turnover_rate", "amount20", "mv_pct", "turn_pct", "amount_pct",
                   "size3", "ep3", "turn3"]], on=["ts_code", "trade_date"], how="left")
    p["ret"] = p.ret_a5_b15_d10 - COST
    bench = p.groupby(["trade_date", "size3", "ep3", "turn3"]).ret.transform("mean")
    p["excess_matched"] = p.ret - bench
    rows = []
    for v in ("base", "turn_cap30", "turn_low50", "turn_balanced", "amt>=1e8", "amt>=3e8", "amt>=5e8"):
        L = funnel(p, v)
        for per, g in (("all", L), ("dev", L[L.trade_date <= "20260126"]), ("confirm", L[L.trade_date > "20260126"])):
            rows.append({"list": "S20 进攻版", "variant": v, "period": per, **score(g)})
    for v in ("base", "amt>=1e8", "amt>=3e8", "amt>=5e8"):
        q = p if v == "base" else p[p.amount20 >= float(v[5:]) / 1000]
        L = select_safe(q, SafeConfig())
        rows.append({"list": "S20 稳健版", "variant": v, "period": "all", **score(L)})
    univ = p.groupby("trade_date").ret.mean()
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "list_ablation.csv", index=False, encoding="utf-8-sig")
    text = (f"days {p.trade_date.nunique()} ({p.trade_date.min()}..{p.trade_date.max()}), universe per-trade "
            f"{univ.mean():.2f}%\n\n" + T.to_string(index=False))
    (OUT / "list_ablation.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
