#!/usr/bin/env python
"""Announced events as flags inside the S20 / R20 top-100 pools: unlocks, earnings window, earnings forecasts.

  python scripts/analyze_event_flags_in_pools.py

Idea I0006 (2026-10-16), designed before looking at any outcome. Evidence tier reused_holdout: every period was already used.
Signal days >= 2026-08-06 are not read.

Flags (all known on the signal day t; t+20 = the 20th session after t):
  unlock_any    announced unlock (share_float: ann_date <= t) with float_date in (t, t+20 sessions]
  unlock_0.2    same, unlocking shares >= 0.2 x average daily volume of the last 20 sessions ("days to absorb" >= 0.2)
  unlock_1      same, days to absorb >= 1   (the dimensionless number of the idea: supply released / daily absorption)
  earn_window   disclosure_date scheduled date (pre_date) in (t, t+20 sessions]; actual_date used when it exists and is <= t+20 sessions
  fc_pos        earnings forecast announced in the 30 calendar days up to t, type in {预增, 略增, 扭亏, 续盈}
  fc_neg        same, type in {预减, 略减, 首亏, 续亏, 增亏}
Measure: flagged minus unflagged inside the pool on the same day, success rate (pp), per trade (U15D10, list basis) and
stop-first share (pp); mean over days with both groups, Newey-West(19) t. Bonferroni over 6 flags x 4 periods x 3 metrics = 72 tests.
Caveats: disclosure pre_date is the schedule in the cache snapshot (not a point-in-time archive); share_float rows are aggregated per
(stock, float_date).
Output: output/experiments/event_flags_20261016/{report.txt, flags.parquet, results.csv}
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

OUT = ROOT / "output/experiments/event_flags_20261016"
EV = ROOT / "output/tushare_cache/events"
SEGS = {"S20": [("S1", "20250303", "20250829"), ("S2", "20260127", "20260805")],
        "R20": [("IS", "20260127", "20260413"), ("OOS", "20260414", "20260805")]}
FLAGS = ["unlock_any", "unlock_0.2", "unlock_1", "earn_window", "fc_pos", "fc_neg"]
POS, NEG = {"预增", "略增", "扭亏", "续盈"}, {"预减", "略减", "首亏", "续亏", "增亏"}
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def sessions() -> list[str]:
    return sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))


def build_flags(cases: pd.DataFrame) -> pd.DataFrame:
    """cases: ts_code, trade_date. Returns the same keys plus the flags."""
    days = sessions()
    pos = {d: i for i, d in enumerate(days)}
    plus20 = {d: (days[pos[d] + 20] if pos[d] + 20 < len(days) else "20991231") for d in pos}
    c = cases[["ts_code", "trade_date"]].drop_duplicates().copy()
    codes = set(c.ts_code)
    # 20-day average volume (shares) at the signal day
    vol = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "vol"]) for p in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet"))
                     if "20250101" <= p.stem <= "20260805"], ignore_index=True)
    vol = vol[vol.ts_code.isin(codes)].sort_values(["ts_code", "trade_date"])
    vol["trade_date"] = vol.trade_date.astype(str)
    vol["adv"] = vol.groupby("ts_code").vol.transform(lambda s: s.rolling(20, min_periods=10).mean()) * 100.0
    c = c.merge(vol[["ts_code", "trade_date", "adv"]], on=["ts_code", "trade_date"], how="left")
    c["t20"] = c.trade_date.map(plus20)

    sf = pd.concat([pd.read_parquet(p, columns=["ts_code", "ann_date", "float_date", "float_share"]) for p in sorted((EV / "share_float").glob("*.parquet"))],
                   ignore_index=True)
    sf = sf[sf.ts_code.isin(codes)]
    sf["float_share"] = pd.to_numeric(sf.float_share, errors="coerce")
    sf = sf.groupby(["ts_code", "float_date"], as_index=False).agg(ann_date=("ann_date", "min"), shares=("float_share", "sum"))
    m = c[["ts_code", "trade_date", "t20", "adv"]].merge(sf, on="ts_code", how="left")
    m = m[(m.ann_date <= m.trade_date) & (m.float_date > m.trade_date) & (m.float_date <= m.t20)]
    u = m.groupby(["ts_code", "trade_date"], as_index=False).shares.sum().rename(columns={"shares": "unlock_shares"})
    c = c.merge(u, on=["ts_code", "trade_date"], how="left")
    c["absorb"] = c.unlock_shares / c.adv
    c["unlock_any"] = c.unlock_shares.fillna(0) > 0
    c["unlock_0.2"] = c.absorb >= 0.2
    c["unlock_1"] = c.absorb >= 1.0

    dd = pd.concat([pd.read_parquet(p, columns=["ts_code", "end_date", "pre_date", "actual_date"]) for p in sorted((EV / "disclosure_date").glob("*.parquet"))],
                   ignore_index=True)
    dd = dd[dd.ts_code.isin(codes)].copy()
    dd["when"] = dd.actual_date.fillna(dd.pre_date)
    m = c[["ts_code", "trade_date", "t20"]].merge(dd[["ts_code", "when"]].dropna(), on="ts_code", how="left")
    m = m[(m.when > m.trade_date) & (m.when <= m.t20)]
    c = c.merge(m[["ts_code", "trade_date"]].drop_duplicates().assign(earn_window=True), on=["ts_code", "trade_date"], how="left")
    c["earn_window"] = c.earn_window.fillna(False).astype(bool)

    fc = pd.concat([pd.read_parquet(p, columns=["ts_code", "ann_date", "type"]) for p in sorted((EV / "forecast").glob("*.parquet"))], ignore_index=True)
    fc = fc[fc.ts_code.isin(codes)].copy()
    fc["d"] = pd.to_datetime(fc.ann_date)
    c["d"] = pd.to_datetime(c.trade_date)
    for name, types in (("fc_pos", POS), ("fc_neg", NEG)):
        f = fc[fc.type.isin(types)][["ts_code", "d"]].rename(columns={"d": "fd"})
        m = c[["ts_code", "trade_date", "d"]].merge(f, on="ts_code", how="inner")
        m = m[(m.fd <= m.d) & (m.fd > m.d - pd.Timedelta(days=30))]
        c = c.merge(m[["ts_code", "trade_date"]].drop_duplicates().assign(**{name: True}), on=["ts_code", "trade_date"], how="left")
        c[name] = c[name].fillna(False).astype(bool)
    return c.drop(columns=["d", "t20"])


def main() -> int:
    from scipy.stats import norm
    OUT.mkdir(parents=True, exist_ok=True)
    P = pools()
    for k in P:
        P[k] = P[k][P[k].trade_date.between("20250303", "20260805")]
    allc = pd.concat(P.values())[["ts_code", "trade_date"]].drop_duplicates()
    cache = OUT / "flags.parquet"
    fl = pd.read_parquet(cache) if cache.exists() else build_flags(allc)
    if not cache.exists():
        fl.to_parquet(cache, index=False)
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet",
                        columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    o["success"] = ((o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))).astype(float)
    o["stopfirst"] = ((o.dn10_day > 0) & ((o.up15_day == 0) | (o.dn10_day <= o.up15_day))).astype(float)
    n_tests = len(FLAGS) * 4 * 3
    bar = norm.isf(0.025 / n_tests)
    say(f"flags {len(FLAGS)} x 4 periods x 3 metrics = {n_tests} tests; Bonferroni |t| >= {bar:.2f}")
    rows = []
    for pname, segs in SEGS.items():
        base = P[pname].merge(o, on=["ts_code", "trade_date"]).merge(fl, on=["ts_code", "trade_date"], how="left")
        for sname, a, b in segs:
            g = base[base.trade_date.between(a, b)]
            say(f"\n==== {pname} top 100, {sname}: {len(g):,} cases, {g.trade_date.nunique()} days; success {100 * g.success.mean():.1f}%, "
                f"per trade {g.ret_base.mean():+.2f}, stop-first {100 * g.stopfirst.mean():.1f}%")
            say(f"{'flag':12s} | share | success diff (t) | per trade diff (t) | stop-first diff (t) | days")
            for k in FLAGS:
                x = g[g[k].fillna(False).astype(bool)]
                y = g[~g[k].fillna(False).astype(bool)]
                r = {"pool": pname, "seg": sname, "flag": k, "share": len(x) / len(g)}
                dd = {}
                for m_, lab, mult in (("success", "succ", 100), ("ret_base", "ret", 1), ("stopfirst", "stop", 100)):
                    d = (x.groupby("trade_date")[m_].mean() - y.groupby("trade_date")[m_].mean()).dropna()
                    dd[lab] = d
                    r[f"{lab}_d"], r[f"{lab}_t"], r["days"] = d.mean() * mult, (tstat(d) if len(d) > 20 else np.nan), len(d)
                rows.append(r)
                say(f"{k:12s} | {100 * r['share']:4.1f}% | {r['succ_d']:+6.1f} ({r['succ_t']:+4.1f}) | {r['ret_d']:+6.2f} ({r['ret_t']:+4.1f}) | "
                    f"{r['stop_d']:+6.1f} ({r['stop_t']:+4.1f}) | {r['days']}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    say(f"\nlargest |t|: success {res.succ_t.abs().max():.1f}, per trade {res.ret_t.abs().max():.1f}, stop-first {res.stop_t.abs().max():.1f}; bar {bar:.2f}")
    say("flags with |t| >= 2 same sign in both periods of a pool (success or stop-first):")
    for pname in SEGS:
        for k in FLAGS:
            r = res[(res.pool == pname) & (res.flag == k)]
            for m_ in ("succ", "stop", "ret"):
                if len(r) == 2 and (r[f"{m_}_t"].abs() >= 2).all() and r[f"{m_}_d"].gt(0).all() | r[f"{m_}_d"].lt(0).all() and (r[f"{m_}_t"].abs() >= 2).all():
                    say(f"  {pname} {k} {m_}: {r[f'{m_}_d'].round(2).tolist()} t {r[f'{m_}_t'].round(1).tolist()}")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
