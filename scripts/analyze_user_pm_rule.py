#!/usr/bin/env python
"""User's afternoon rule (2026-10-26): after 14:30 open the gain ranking, find the stock on its first red day whose short
intraday MA5 has just turned up, buy at once, sell the next morning at a high, repeat.   Idea I0007.

  python scripts/analyze_user_pm_rule.py

PRE-REGISTERED before any result was looked at (definitions are fixed here and not tuned afterwards).
Evidence tier: reused_holdout / descriptive. Both periods were already used by earlier S20/R20 work. Data end 2026-08-05; nothing from
the reserved window (signal days >= 2026-08-06) exists in the cache and none is read.

Data: output/tushare_cache/min15 (Tushare stk_mins, 15-minute bars, bar labelled by its END time, 09:30 = opening auction), the 1,976
stocks that ever entered the S20 stage1 top 100. Tushare has no 10-minute bars, 15 minutes is the nearest. This universe is NOT the
whole market (it is momentum-selected by construction), so "top 50 of the gain ranking" below is top 50 among these 1,976.
Decision bars: the 14:30 bar and the 14:45 bar (bar close = price), the rule takes the earliest bar of the day that satisfies it.
  gain       close of the decision bar / pre_close - 1 (daily cache); rank among the cached stocks that are not at the up-limit
  gainer50   gain rank <= 50 and the price is not within 0.1% of the up-limit (cannot be bought)
  first_red  gain > 0 and the previous session's pct_chg <= 0   (variant first_red2: also the session before that <= 0)
  ma_turn    MA5 of the 15-minute closes at the decision bar > MA5 one bar earlier, and MA5 one bar earlier <= MA5 two bars earlier
  RULE       gainer50 & first_red & ma_turn (the user's rule). Also reported alone: gainer50, gainer50&first_red, gainer50&ma_turn,
             first_red&ma_turn (no ranking requirement), and the baseline "every cached stock at 14:30".
Sell (all next session, return measured from the decision-bar close):
  r_open   next-session open;  r_1000  close of the 10:00 bar (PRIMARY, a time rule that can be executed);  r_close  next close;
  r_hi     highest price of the next morning 09:45..11:30 bars (ORACLE: cannot be sold at the high, shown only as a ceiling).
Net = gross - 0.3% round trip (the project's cost). List-basis, equal weight per trade, per-day mean then Newey-West(19) t.
Periods: dev 2025-03-03..2026-01-26, conf 2026-01-27..2026-08-05.
PRIMARY: RULE net r_1000 mean (daily-mean) and paired difference to "gainer50 minus RULE members" on the same day.
Falsified (idea I0007) if either period has net r_1000 mean <= 0, or the paired difference has t < 1.96, or (part B) flagged minus unflagged
success rate in the S20 top 100 is <= 0.
Part B: the same flags (evaluated on the signal day t, known by the close) inside the S20 stage1 top 100 (list-making is post-close, so the
flag can be used as a next-open filter): success (+15% before -10% within 20 sessions, U15D10 list basis) flagged minus unflagged, paired
by day.
Output: output/experiments/user_pm_rule_20261026/{report.txt, trades.parquet, groups.csv}
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

OUT = ROOT / "output/experiments/user_pm_rule_20261026"
CACHE = ROOT / "output/tushare_cache"
START, END = "20250203", "20260805"
PERIODS = [("dev", "20250303", "20260126"), ("conf", "20260127", "20260805")]
COST = 0.003
LINES: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LINES.append(s)


def build() -> pd.DataFrame:
    days = sorted(p.stem for p in (CACHE / "daily").glob("*.parquet") if START <= p.stem <= END)
    nxt = dict(zip(days[:-1], days[1:]))
    d = pd.concat([pd.read_parquet(CACHE / "daily" / f"{x}.parquet", columns=["ts_code", "trade_date", "open", "close", "pre_close", "pct_chg"]) for x in days])
    d["trade_date"] = d.trade_date.astype(str)
    lim = pd.concat([pd.read_parquet(CACHE / "stk_limit" / f"{x}.parquet", columns=["ts_code", "trade_date", "up_limit"]) for x in days if (CACHE / "stk_limit" / f"{x}.parquet").exists()])
    lim["trade_date"] = lim.trade_date.astype(str)
    d = d.merge(lim, on=["ts_code", "trade_date"], how="left").sort_values(["ts_code", "trade_date"])
    g = d.groupby("ts_code")
    d["prev_red"] = g.pct_chg.shift(1) > 0
    d["prev2_red"] = g.pct_chg.shift(2) > 0
    d["next_date"] = d.trade_date.map(nxt)
    dn = d[["ts_code", "trade_date", "open", "close"]].rename(columns={"trade_date": "next_date", "open": "next_open", "close": "next_close"})
    d = d.merge(dn, on=["ts_code", "next_date"], how="left")

    rows = []
    for p in sorted((CACHE / "min15").glob("*.parquet")):
        m = pd.read_parquet(p)
        t = m.trade_time.astype(str)
        m = m.assign(trade_date=t.str[:4] + t.str[5:7] + t.str[8:10], hm=t.str[11:16])
        m = m[(m.trade_date >= START) & (m.trade_date <= END) & (m.hm != "09:30")].sort_values(["trade_date", "hm"]).reset_index(drop=True)
        if m.empty:
            continue
        m["ma"] = m.close.rolling(5).mean()
        m["ma1"], m["ma2"] = m.ma.shift(1), m.ma.shift(2)
        dec = m[m.hm.isin(["14:30", "14:45"])][["trade_date", "hm", "close", "ma", "ma1", "ma2"]].rename(columns={"close": "p"})
        c1000 = m[m.hm == "10:00"].set_index("trade_date").close.rename("c1000")
        hi = m[(m.hm >= "09:45") & (m.hm <= "11:30")].groupby("trade_date").high.max().rename("hi_am")
        nd = pd.concat([c1000, hi], axis=1).reset_index().rename(columns={"trade_date": "next_date"})
        dec = dec.assign(ts_code=p.stem, next_date=dec.trade_date.map(nxt)).merge(nd, on="next_date", how="left")
        rows.append(dec)
    x = pd.concat(rows, ignore_index=True).merge(d, on=["ts_code", "trade_date", "next_date"], how="left")
    x = x[x.pre_close.notna() & x.c1000.notna()].copy()
    x["gain"] = x.p / x.pre_close - 1
    x["buyable"] = ~(x.p >= x.up_limit * 0.999) if x.up_limit.notna().any() else True
    x["buyable"] = x.buyable.fillna(True)
    ok = x[x.buyable]
    x["rank"] = ok.groupby(["trade_date", "hm"]).gain.rank(ascending=False, method="first")
    x["gainer50"] = x["rank"] <= 50
    x["first_red"] = (x.gain > 0) & (x.prev_red == False)  # noqa: E712
    x["first_red2"] = x.first_red & (x.prev2_red == False)  # noqa: E712
    x["ma_turn"] = (x.ma > x.ma1) & (x.ma1 <= x.ma2)
    for a, k in (("r_open", "next_open"), ("r_1000", "c1000"), ("r_close", "next_close"), ("r_hi", "hi_am")):
        x[a] = x[k] / x.p - 1
    return x


def groups(x: pd.DataFrame) -> dict[str, pd.DataFrame]:
    g = {"all@14:30": x[x.hm == "14:30"], "gainer50@14:30": x[(x.hm == "14:30") & x.gainer50]}
    for name, m in (("gainer50", x.gainer50), ("gainer50&first_red", x.gainer50 & x.first_red), ("gainer50&ma_turn", x.gainer50 & x.ma_turn),
                    ("first_red&ma_turn", x.first_red & x.ma_turn), ("RULE", x.gainer50 & x.first_red & x.ma_turn),
                    ("RULE(first_red2)", x.gainer50 & x.first_red2 & x.ma_turn)):
        g[name] = x[m].sort_values("hm").drop_duplicates(["ts_code", "trade_date"])
    return g


def summarize(g: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for gname, df in g.items():
        for pname, a, b in PERIODS:
            s = df[df.trade_date.between(a, b)]
            if s.empty:
                continue
            r = {"group": gname, "period": pname, "trades": len(s), "days": s.trade_date.nunique()}
            for k in ("r_open", "r_1000", "r_close", "r_hi"):
                dm = s.groupby("trade_date")[k].mean()
                r[k] = 100 * dm.mean()
                if k == "r_1000":
                    r["r_1000_net"], r["t_net"] = 100 * (dm.mean() - COST), tstat(dm - COST) if len(dm) > 20 else np.nan
                    r["win_net"] = 100 * (s[k] - COST > 0).mean()
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "trades.parquet"
    x = pd.read_parquet(cache) if cache.exists() else build()
    if not cache.exists():
        x.to_parquet(cache, index=False)
    say(f"decision rows {len(x):,}; stocks {x.ts_code.nunique()}; days {x.trade_date.nunique()}; flags share: gainer50 {100 * x.gainer50.mean():.1f}%, "
        f"first_red {100 * x.first_red.mean():.1f}%, ma_turn {100 * x.ma_turn.mean():.1f}%")
    g = groups(x)
    res = summarize(g)
    res.to_csv(OUT / "groups.csv", index=False, encoding="utf-8-sig")
    say("\nPart A: next-morning returns from the decision-bar close (per-day mean, %, gross unless noted). r_hi is an ORACLE ceiling.")
    say(res.round(3).to_string(index=False))
    say("\nPaired: RULE net r_1000 minus the other top-50 gainers on the same day (percentage points)")
    rule, top = g["RULE"], g["gainer50"]
    for pname, a, b in PERIODS:
        r = rule[rule.trade_date.between(a, b)]
        o = top[top.trade_date.between(a, b)]
        o = o.merge(r[["ts_code", "trade_date"]].assign(inr=1), on=["ts_code", "trade_date"], how="left")
        o = o[o.inr.isna()]
        d = (r.groupby("trade_date").r_1000.mean() - o.groupby("trade_date").r_1000.mean()).dropna()
        say(f"  {pname}: days {len(d)}, diff {100 * d.mean():+.3f}pp, t {tstat(d) if len(d) > 20 else float('nan'):+.2f}")
        a1 = r.groupby("trade_date").r_1000.mean()
        say(f"  {pname}: RULE gross r_1000 mean {100 * a1.mean():+.3f}% over {len(a1)} days with a trigger; days with no trigger skipped (cash)")

    say("\nPart B: S20 stage1 top 100, flags on the signal day, U15D10 list-basis success (+15% before -10%); flagged minus unflagged")
    P = pools()["S20"]
    P = P[P.trade_date.between("20250303", "20260805")]
    fl = x.sort_values("hm").drop_duplicates(["ts_code", "trade_date"])[["ts_code", "trade_date"]].copy()
    flags = {}
    for name, m in (("gainer50", x.gainer50), ("first_red", x.first_red), ("ma_turn", x.ma_turn), ("RULE", x.gainer50 & x.first_red & x.ma_turn)):
        flags[name] = x[m][["ts_code", "trade_date"]].drop_duplicates().assign(**{name: True})
    base = P.merge(fl.assign(has_min=True), on=["ts_code", "trade_date"], how="left")
    for k, v in flags.items():
        base = base.merge(v, on=["ts_code", "trade_date"], how="left")
        base[k] = base[k].fillna(False).astype(bool)
    o = pd.read_parquet(ROOT / "output/experiments/s20_alt_designs_20261011/paths.parquet", columns=["ts_code", "trade_date", "ret_base", "up15_day", "dn10_day"])
    o["success"] = ((o.up15_day > 0) & ((o.dn10_day == 0) | (o.up15_day < o.dn10_day))).astype(float)
    o["stopfirst"] = ((o.dn10_day > 0) & ((o.up15_day == 0) | (o.dn10_day <= o.up15_day))).astype(float)
    base = base.merge(o, on=["ts_code", "trade_date"])
    say(f"  pool cases {len(base):,}; with minute data on the day {100 * base.has_min.fillna(False).mean():.1f}%")
    say(f"  {'period':5s} {'flag':9s} | share | success diff pp (t) | per-trade diff (t) | stop-first diff pp (t) | flagged n")
    for pname, a, b in PERIODS:
        s = base[base.trade_date.between(a, b)]
        for k in flags:
            xs, ys = s[s[k]], s[~s[k]]
            out = []
            for m_, mult in (("success", 100), ("ret_base", 1), ("stopfirst", 100)):
                d = (xs.groupby("trade_date")[m_].mean() - ys.groupby("trade_date")[m_].mean()).dropna()
                out.append(f"{d.mean() * mult:+6.2f} ({tstat(d) if len(d) > 20 else float('nan'):+4.1f})")
            say(f"  {pname:5s} {k:9s} | {100 * s[k].mean():4.1f}% | {out[0]} | {out[1]} | {out[2]} | {int(s[k].sum())}")
    (OUT / "report.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
