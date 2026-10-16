#!/usr/bin/env python
"""Which signal-day technical indicators separate winners from losers inside the S20 / R20 top-100 pools?

  python scripts/analyze_list_indicators.py

User question 2026-10-15. Descriptive on already-used periods (evidence tier reused_holdout): no indicator
found here is a validated rule. Safeguards: (1) within-day percentile rank inside the pool, so the day
mix cannot drive the result; (2) top tercile minus bottom tercile of the indicator, success rate
(+15% before -10% within 20 sessions, list basis U15D10 from the next open) and per-trade return, same-day
paired, Newey-West(19) t; (3) pick-then-confirm: an indicator is picked on the first period of a pool
(S1 / R20 in-sample, |t| >= 2) and confirmed on the second (same sign, |t| >= 2); (4) the multiple-testing
bar is Bonferroni over all indicators x periods x 2 metrics.
Indicators (all use data up to the signal day): returns 1/5/10/20/60, RSI 6/14, MACD hist and DIF (/close),
KDJ J, Bollinger %b and bandwidth, deviation from MA5/10/20/60, MA5/MA20 slope, distance to the 60-day
high / low, 20-day range position, NATR(14), volume vs previous 5 / 20 days, turnover rate, volume ratio
(daily_basic), today's amplitude, body, upper / lower shadow, gap, up days in the last 5.
Signal days >= 2026-08-06 are not read. Output: output/experiments/list_indicators_20261015/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from analyze_bottom_patterns import pools  # noqa: E402
from analyze_r20_s20_nested import tstat  # noqa: E402

OUT = ROOT / "output/experiments/list_indicators_20261015"
SEGS = {"S20": [("S1", "20250303", "20250829"), ("S2", "20260127", "20260805")],
        "R20": [("IS", "20260127", "20260413"), ("OOS", "20260414", "20260805")]}
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def feats(g: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (g[k].astype(float) for k in ("open", "high", "low", "close", "vol"))
    pc = c.shift()
    f = pd.DataFrame(index=g.index)
    for n in (1, 5, 10, 20, 60):
        f[f"ret{n}"] = c / c.shift(n) - 1
    d = c.diff()
    for n in (6, 14):
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        f[f"rsi{n}"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = 2 * (dif - dea) / c
    f["macd_dif"] = dif / c
    lo9, hi9 = l.rolling(9).min(), h.rolling(9).max()
    rsv = ((c - lo9) / (hi9 - lo9).replace(0, np.nan) * 100).fillna(50)
    K = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    D = K.ewm(alpha=1 / 3, adjust=False).mean()
    f["kdj_j"] = 3 * K - 2 * D
    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60)}
    sd20 = c.rolling(20).std(ddof=0)
    f["boll_pctb"] = (c - (ma[20] - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    f["boll_bw"] = 4 * sd20 / ma[20]
    for n in (5, 10, 20, 60):
        f[f"dev_ma{n}"] = c / ma[n] - 1
    f["ma5_slope"] = ma[5] / ma[5].shift(3) - 1
    f["ma20_slope"] = ma[20] / ma[20].shift(5) - 1
    f["dist_high60"] = c / h.rolling(60).max()
    f["dist_low60"] = c / l.rolling(60).min()
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    f["range_pos20"] = (c - lo20) / (hi20 - lo20).replace(0, np.nan)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    f["natr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c
    f["vol_vs5"] = v / v.shift().rolling(5).mean()
    f["vol_vs20"] = v / v.shift().rolling(20).mean()
    f["amp"] = (h - l) / pc
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["upper_shadow"] = (h - np.maximum(o, c)) / rng
    f["lower_shadow"] = (np.minimum(o, c) - l) / rng
    f["gap"] = o / pc - 1
    f["up_days5"] = (d > 0).rolling(5).sum()
    f["ts_code"], f["trade_date"] = g.ts_code.to_numpy(), g.trade_date.to_numpy()
    return f


def panel(codes: set[str]) -> pd.DataFrame:
    from price_panel import adjusted_daily
    px = adjusted_daily("20240801", "20260805", codes)
    f = pd.concat([feats(g) for _, g in px.groupby("ts_code", sort=False)], ignore_index=True)
    db = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "turnover_rate", "volume_ratio"])
                    for p in sorted((ROOT / "output/tushare_cache/daily_basic").glob("*.parquet"))
                    if "20250101" <= p.stem <= "20260805"], ignore_index=True)
    db["trade_date"] = db.trade_date.astype(str)
    return f.merge(db, on=["ts_code", "trade_date"], how="left")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    P = pools()
    codes = set().union(*(set(p.ts_code) for p in P.values()))
    cache = OUT / "indicators.parquet"
    if cache.exists():
        ind = pd.read_parquet(cache)
    else:
        ind = panel(codes)
        ind.to_parquet(cache, index=False)
    names = [c for c in ind.columns if c not in ("ts_code", "trade_date")]
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet",
                        columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    o["success"] = ((o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))).astype(float)
    n_tests = len(names) * 4 * 2
    say(f"indicators {len(names)}; Bonferroni over {n_tests} tests: |t| >= {2.0 + 0.0:.1f} nominal, "
        f"{__import__('scipy.stats', fromlist=['norm']).norm.isf(0.025 / n_tests):.2f} after correction")
    res: dict[tuple[str, str], pd.DataFrame] = {}
    for pname, segs in SEGS.items():
        base = P[pname].merge(o, on=["ts_code", "trade_date"]).merge(ind, on=["ts_code", "trade_date"], how="left")
        for sname, a, b in segs:
            g = base[base.trade_date.between(a, b)].copy()
            rows = []
            for k in names:
                pct = g.groupby("trade_date")[k].rank(pct=True)
                top, bot = g[pct > 2 / 3], g[pct <= 1 / 3]
                r = {"ind": k}
                for m in ("success", "ret_base"):
                    dd = (top.groupby("trade_date")[m].mean() - bot.groupby("trade_date")[m].mean()).dropna()
                    r[f"{m}_d"] = dd.mean() * (100 if m == "success" else 1)
                    r[f"{m}_t"] = tstat(dd)
                rows.append(r)
            res[(pname, sname)] = pd.DataFrame(rows).set_index("ind")
            say(f"\n==== {pname} top 100, {sname}: {len(g):,} cases, {g.trade_date.nunique()} days; success {100 * g.success.mean():.1f}%, "
                f"per trade {g.ret_base.mean():+.2f}")
            x = res[(pname, sname)].sort_values("success_t")
            say("top tercile minus bottom tercile (success: pct points; per trade: points); strongest 6 each way by success t")
            say(f"{'indicator':14s} | success diff (t) | per-trade diff (t)")
            for k, r in pd.concat([x.head(6), x.tail(6)]).iterrows():
                say(f"{k:14s} | {r.success_d:+6.1f} ({r.success_t:+5.1f}) | {r.ret_base_d:+6.2f} ({r.ret_base_t:+5.1f})")
    say("\n==== pick on period 1 (|t| >= 2 on success), confirm on period 2 (same sign, |t| >= 2) ====")
    from scipy.stats import norm
    bar = norm.isf(0.025 / n_tests)
    for pname, segs in SEGS.items():
        r1, r2 = res[(pname, segs[0][0])], res[(pname, segs[1][0])]
        pick = r1[r1.success_t.abs() >= 2].index
        conf = [k for k in pick if np.sign(r2.loc[k, "success_d"]) == np.sign(r1.loc[k, "success_d"]) and abs(r2.loc[k, "success_t"]) >= 2]
        say(f"\n{pname}: picked {len(pick)} on {segs[0][0]}, confirmed {len(conf)} on {segs[1][0]} "
            f"(by chance about {0.05 * len(pick):.1f} would pass the |t|>=2 test with the right sign even with no signal: ~{0.025 * len(pick):.1f})")
        for k in conf:
            say(f"  {k:14s} {segs[0][0]} {r1.loc[k, 'success_d']:+.1f} ({r1.loc[k, 'success_t']:+.1f}) -> "
                f"{segs[1][0]} {r2.loc[k, 'success_d']:+.1f} ({r2.loc[k, 'success_t']:+.1f}); per trade "
                f"{r1.loc[k, 'ret_base_d']:+.2f} / {r2.loc[k, 'ret_base_d']:+.2f}")
    say("\n==== same sign in all four periods, success difference ====")
    allp = pd.concat({f"{p}-{s}": res[(p, s)].success_d for p, segs in SEGS.items() for s, _, _ in segs}, axis=1)
    allt = pd.concat({f"{p}-{s}": res[(p, s)].success_t for p, segs in SEGS.items() for s, _, _ in segs}, axis=1)
    same = allp[(np.sign(allp).nunique(axis=1) == 1)]
    for k, r in same.iterrows():
        say(f"  {k:14s} " + "  ".join(f"{c} {r[c]:+.1f} ({allt.loc[k, c]:+.1f})" for c in allp.columns))
    say(f"\nlargest |t| anywhere: {allt.abs().max().max():.1f}; corrected bar {bar:.2f}")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
