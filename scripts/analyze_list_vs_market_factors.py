#!/usr/bin/env python
"""Do index, sector-index and ETF factors tell good list days (or good names) from bad ones?

Two units of analysis, both on the days outside the frozen model's fit and tuning windows
(2025-03-03..08-31 and 2026-01-27..08-05, 251 days), for the frozen (trend-chasing) and v2
(non-chasing) offensive lists built with the frozen funnel:

  A. day level   one row per signal day: the list's mean +15%/-10% exit return against market
                 factors known at that day's close (CSI 300 / 500 / 1000 / ChiNext bars, Shenwan
                 level-1 industry indices, ETF turnover). Reported as rank correlation and
                 tercile means, for every factor, without picking thresholds.
  B. name level  one row per listed name: its own industry's strength and trend at the signal day
                 (industry = Tushare industry label, equal-weight index built from the stock cache),
                 against its exit return and whether it hit -10% first.

The factor list is fixed before running; nothing is tuned. Descriptive: these days have been used
many times (in-list questions about 150 rules). The reserved window is not read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_s20_trend_filters import trend_states  # noqa: E402
from experiment_cross_filter_v2 import RESERVED_FROM, nw_se  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

OUT = ROOT / "output/experiments/list_vs_market_factors_20261006"
CACHE = ROOT / "output/tushare_cache"
SCORES = {"frozen": ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet",
          "v2": ROOT / "output/experiments/s20_unified_v2_20261005/predictions.parquet"}
OOS = lambda d: (d < "20250901") | (d > "20260126")  # noqa: E731


def index_close(code: str) -> pd.Series:
    f = pd.read_parquet(CACHE / "index_daily" / f"{code}.parquet", columns=["trade_date", "close"])
    return f.set_index("trade_date").close.sort_index()


def market_factors() -> pd.DataFrame:
    hs, zz500, zz1000, cyb = (index_close(c) for c in ("000300.SH", "000905.SH", "000852.SH", "399006.SZ"))
    r = lambda s, n: s / s.shift(n) - 1  # noqa: E731
    f = pd.DataFrame(index=hs.index)
    f["hs300_r5"], f["hs300_r20"], f["hs300_r60"] = r(hs, 5), r(hs, 20), r(hs, 60)
    f["hs300_vol20"] = np.log(hs).diff().rolling(20).std() * np.sqrt(252)
    f["hs300_dd60"] = hs / hs.rolling(60).max() - 1
    ma20 = hs.rolling(20).mean()
    f["hs300_ma20_slope5"] = ma20 / ma20.shift(5) - 1
    f["hs300_vs_ma60"] = hs / hs.rolling(60).mean() - 1
    f["small_minus_large_20"] = r(zz1000, 20) - r(hs, 20)
    f["growth_minus_large_20"] = r(cyb, 20) - r(hs, 20)
    f["zz500_r20"] = r(zz500, 20)
    # Shenwan level-1 industries
    l1 = pd.read_parquet(CACHE / "sw_index_classify.parquet")
    l1 = set(l1[l1.level == "L1"].index_code)
    sw = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "close"]) for p in sorted((CACHE / "sw_daily").glob("*.parquet"))])
    sw = sw[sw.ts_code.isin(l1)].pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    s20 = sw / sw.shift(20) - 1
    s10 = sw / sw.shift(10) - 1
    f["sw_l1_up_share_20"] = (s20 > 0).mean(axis=1)
    f["sw_l1_dispersion_20"] = s20.std(axis=1)
    f["sw_l1_persist"] = pd.Series({d: s10.loc[d].corr(s10.shift(10).loc[d], method="spearman") for d in s10.index})
    f["sw_l1_ma20_up_share"] = (sw.rolling(20).mean() > sw.rolling(20).mean().shift(5)).mean(axis=1)
    # ETF turnover
    etf = pd.concat([pd.read_parquet(p, columns=["trade_date", "amount"]).groupby("trade_date").amount.sum()
                     for p in sorted((CACHE / "etf_daily").glob("*.parquet"))])
    etf = etf.sort_index()
    f["etf_amount_z20"] = (etf - etf.rolling(20).mean()) / etf.rolling(20).std()
    f["etf_amount_vs_60"] = etf / etf.rolling(60).mean() - 1
    f["stock_breadth_ma20"] = trend_states().groupby("trade_date").ma20_up.mean()
    return f


def industry_context() -> pd.DataFrame:
    """Equal-weight industry indices from the stock cache: 20-day return rank and MA20 direction per industry-day."""
    px = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "pct_chg"])
                    for p in sorted((CACHE / "daily").glob("*.parquet")) if "20241001" <= p.stem < RESERVED_FROM])
    px["trade_date"] = px["trade_date"].astype(str)
    basic = pd.read_parquet(CACHE / "stock_basic.parquet", columns=["ts_code", "industry"])
    px = px.merge(basic, on="ts_code").dropna(subset=["industry"])
    ind = px.groupby(["trade_date", "industry"]).pct_chg.mean().unstack("industry").sort_index() / 100.0
    level = np.exp(np.log1p(ind.fillna(0.0)).cumsum())
    r20 = level / level.shift(20) - 1
    ma20 = level.rolling(20).mean()
    out = pd.DataFrame({"ind_r20_rank": r20.rank(axis=1, pct=True).stack(), "ind_ma20_up": (ma20 > ma20.shift(5)).stack().astype(float),
                        "ind_r20": r20.stack()})
    out.index.names = ["trade_date", "industry"]
    return out.reset_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    out = outcomes()[["ts_code", "trade_date", "ret_v1", "state"]]
    basic = pd.read_parquet(CACHE / "stock_basic.parquet", columns=["ts_code", "industry"])
    lists = {}
    for name, path in SCORES.items():
        picked = select(pd.read_parquet(path), PureConfig(), "U15D10")[["ts_code", "trade_date"]].merge(out).merge(basic, on="ts_code", how="left")
        lists[name] = picked[OOS(picked.trade_date)]
    factors = market_factors()
    lines = ["Index, sector-index and ETF factors against the S20 offensive lists. 251 days outside the frozen model's fit and tuning windows.",
             "ret = +15%/-10% exit return per listed name (list basis). Factors fixed before running; no thresholds tuned.", ""]

    lines.append("== A. day level: rank correlation of each factor with the day's list return, and tercile means (low / mid / high) ==")
    lines.append(f"{'factor':24s} | {'frozen rho':>10s} {'t':>6s} | {'frozen terciles':>22s} | {'v2 rho':>8s} {'t':>6s} | {'v2 terciles':>22s}")
    table_a = []
    for col in factors.columns:
        row = f"{col:24s}"
        for name in ("frozen", "v2"):
            day = lists[name].groupby("trade_date").ret_v1.mean().to_frame("ret").join(factors[col].rename("x")).dropna()
            rho = day.ret.corr(day.x, method="spearman")
            # t for the rank correlation from the slope of ret on rank(x), day-clustered via NW
            xr = day.x.rank(pct=True) - 0.5
            beta = (day.ret * xr).sum() / (xr ** 2).sum()
            resid = day.ret - beta * xr
            se = np.sqrt(nw_se((resid * xr).to_numpy()) ** 2 * len(day) ** 2 / ((xr ** 2).sum() ** 2)) if len(day) > 10 else float("nan")
            terc = day.groupby(pd.qcut(day.x, 3, labels=["low", "mid", "high"]), observed=True).ret.mean()
            row += f" | {rho:+10.2f} {beta / se if se else float('nan'):+6.2f} | {terc['low']:+6.2f} {terc['mid']:+6.2f} {terc['high']:+6.2f}      " if name == "frozen" else \
                   f" | {rho:+8.2f} {beta / se if se else float('nan'):+6.2f} | {terc['low']:+6.2f} {terc['mid']:+6.2f} {terc['high']:+6.2f}"
            table_a.append({"factor": col, "model": name, "rho": rho, "t": beta / se if se else np.nan,
                            "low": terc["low"], "mid": terc["mid"], "high": terc["high"]})
        lines.append(row)
    lines.append("")
    strong = [r for r in table_a if abs(r["t"]) >= 2]
    lines.append(f"factors with |t| >= 2: {len(strong)} of {len(table_a)} cells: " + "; ".join(f"{r['factor']}/{r['model']} rho {r['rho']:+.2f} t {r['t']:+.2f}" for r in strong))
    both = {}
    for r in table_a:
        both.setdefault(r["factor"], {})[r["model"]] = r["rho"]
    same = [f"{k} (frozen {v['frozen']:+.2f}, v2 {v['v2']:+.2f})" for k, v in both.items() if v["frozen"] * v["v2"] > 0 and min(abs(v["frozen"]), abs(v["v2"])) >= 0.15]
    lines.append("factors with the same sign and |rho| >= 0.15 for both lists: " + ("; ".join(same) if same else "none"))
    lines.append("")

    lines.append("== B. name level: a listed name's own industry at the signal day ==")
    ind = industry_context()
    for name in ("frozen", "v2"):
        q = lists[name].merge(ind, on=["trade_date", "industry"], how="left").dropna(subset=["ind_r20_rank"])
        q["ind_strength"] = pd.cut(q.ind_r20_rank, [0, 1 / 3, 2 / 3, 1.0], labels=["weak third", "middle", "strong third"])
        lines.append(f"[{name}] names with an industry label: {len(q):,} of {len(lists[name]):,}")
        for grp, g in q.groupby("ind_strength", observed=True):
            lines.append(f"    industry 20d return in the {grp:12s}: n {len(g):5d} ret {g.ret_v1.mean():+5.2f} pure_up {100 * (g.state == 'pure_up').mean():4.1f} pure_down {100 * (g.state == 'pure_down').mean():4.1f}")
        for grp, g in q.groupby("ind_ma20_up"):
            lines.append(f"    industry MA20 {'rising ' if grp == 1 else 'falling'}                 : n {len(g):5d} ret {g.ret_v1.mean():+5.2f} pure_up {100 * (g.state == 'pure_up').mean():4.1f} pure_down {100 * (g.state == 'pure_down').mean():4.1f}")
        hi, lo = q[q.ind_strength == "strong third"], q[q.ind_strength == "weak third"]
        d = (hi.groupby("trade_date").ret_v1.mean() - lo.groupby("trade_date").ret_v1.mean()).dropna()
        lines.append(f"    same-day strong-industry minus weak-industry names: {d.mean():+.2f} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, days {len(d)})")
        up, dn = q[q.ind_ma20_up == 1], q[q.ind_ma20_up == 0]
        d = (up.groupby("trade_date").ret_v1.mean() - dn.groupby("trade_date").ret_v1.mean()).dropna()
        lines.append(f"    same-day industry-MA20-rising minus falling names : {d.mean():+.2f} (t {d.mean() / nw_se(d.to_numpy()):+.2f}, days {len(d)})")
        conc = q.groupby("trade_date").ind_r20_rank.mean()
        day = q.groupby("trade_date").ret_v1.mean().to_frame("ret").join(conc.rename("x"))
        lines.append(f"    day level: mean industry-strength rank of the list vs the day's return, rho {day.ret.corr(day.x, method='spearman'):+.2f}")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    pd.DataFrame(table_a).to_csv(OUT / "day_level_factors.csv", index=False)
    factors.to_csv(OUT / "market_factors_daily.csv")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
