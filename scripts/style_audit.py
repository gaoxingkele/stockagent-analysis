#!/usr/bin/env python
"""Week-1 style audit: what do our lists load on, and how much survives style matching?

Lists: S20 v1, v1 + valve B, v1.1 safe (2025-03..2026-08, s20_pure_history) and the
replayed R20 Pool A (2026-04-14..), plus the universe.

1. Exposure: per day, each pick's cross-sectional percentile (universe = SH/SZ names with
   a daily_basic row) of total market cap, EP (1/PE_ttm), BP, turnover, amount, natr14,
   MAX20 (largest daily return over 20 sessions); list mean by month.
2. Characteristic-matched excess (Daniel-Grinblatt-Titman-Wermers style): benchmark for a
   pick = mean outcome of all universe names in the same day x size-tercile x EP-tercile x
   turnover-tercile cell. Outcome = band exit +5/+15/-10 per trade (same yardstick as
   everywhere else). Raw minus matched = selection skill net of size/value/turnover.
3. Micro-cap share: fraction of picks in the bottom 30% of market cap (CH-3 shell zone),
   and results with those picks removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from s20_pure_history import history_lists, outcomes  # noqa: E402

OUT = ROOT / "output/experiments/style_audit"
CHAR = ["mv_pct", "ep_pct", "bp_pct", "turn_pct", "amount_pct", "natr_pct", "max20_pct"]


def characteristics(dates: set[str]) -> pd.DataFrame:
    db_dir = ROOT / "output/tushare_cache/daily_basic"
    daily_dir = ROOT / "output/tushare_cache/daily"
    parts = []
    all_days = sorted(p.stem for p in daily_dir.glob("*.parquet"))
    need = sorted(d for d in all_days if d >= "20250101")
    px = pd.concat([pd.read_parquet(daily_dir / f"{d}.parquet",
                                    columns=["ts_code", "trade_date", "high", "low", "close", "pre_close", "pct_chg", "amount"])
                    for d in all_days if d >= "20241101"], ignore_index=True)
    px["trade_date"] = px.trade_date.astype(str)
    px = px[px.ts_code.str.endswith((".SH", ".SZ"))].sort_values(["ts_code", "trade_date"])
    tr = np.maximum(px.high - px.low, np.maximum((px.high - px.pre_close).abs(), (px.low - px.pre_close).abs()))
    g = px.groupby("ts_code")
    px["natr14"] = tr.groupby(px.ts_code).transform(lambda s: s.rolling(14, min_periods=10).mean()) / px.close
    px["max20"] = g.pct_chg.transform(lambda s: s.rolling(20, min_periods=15).max())
    px["amount20"] = g.amount.transform(lambda s: s.rolling(20, min_periods=15).mean())
    px = px[px.trade_date.isin(dates)]
    for d in sorted(dates):
        f = db_dir / f"{d}.parquet"
        if not f.exists():
            continue
        b = pd.read_parquet(f, columns=["ts_code", "total_mv", "pe_ttm", "pb", "turnover_rate"])
        b["trade_date"] = d
        parts.append(b)
    db = pd.concat(parts, ignore_index=True)
    c = db.merge(px[["ts_code", "trade_date", "natr14", "max20", "amount20"]], on=["ts_code", "trade_date"], how="inner")
    c["ep"] = 1 / c.pe_ttm.where(c.pe_ttm > 0)          # loss makers -> NaN EP (ranked lowest below)
    c["bp"] = 1 / c.pb.where(c.pb > 0)
    grp = c.groupby("trade_date")
    for col, name in (("total_mv", "mv_pct"), ("ep", "ep_pct"), ("bp", "bp_pct"), ("turnover_rate", "turn_pct"),
                      ("amount20", "amount_pct"), ("natr14", "natr_pct"), ("max20", "max20_pct")):
        c[name] = grp[col].rank(pct=True)
    c["ep_pct"] = c.ep_pct.fillna(0.0)                    # negative earnings = cheapest bucket is wrong, put at bottom
    c["size3"] = pd.cut(c.mv_pct, [0, 1 / 3, 2 / 3, 1], labels=False, include_lowest=True)
    c["ep3"] = pd.cut(c.ep_pct, [-0.01, 1 / 3, 2 / 3, 1], labels=False, include_lowest=True)
    c["turn3"] = pd.cut(c.turn_pct, [0, 1 / 3, 2 / 3, 1], labels=False, include_lowest=True)
    return c


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L = history_lists()
    names = {"U15D10": "S20 进攻版", "U15D10+B": "S20 进攻版+B", "safe_v1_1": "S20 稳健版"}
    L = L[L.rule.isin(names)].assign(list=lambda x: x.rule.map(names))[["trade_date", "ts_code", "list"]]
    r20f = sorted((ROOT / "output/r20_history").glob("r20_lists_2026041*_*.parquet"))
    if r20f:
        R = pd.read_parquet(r20f[0])
        R = R[R["list"] == "pool_a"].assign(list="R20 池A")[["trade_date", "ts_code", "list"]]
        L = pd.concat([L, R], ignore_index=True)
    o = outcomes()[["ts_code", "trade_date", "ret_safe", "cls_a5_d10"]]
    dates = set(L.trade_date) & set(o.trade_date)
    C = characteristics(dates)
    U = C.merge(o, on=["ts_code", "trade_date"])                    # universe with outcomes
    cell = U.groupby(["trade_date", "size3", "ep3", "turn3"]).ret_safe.mean().rename("bench")
    U = U.join(cell, on=["trade_date", "size3", "ep3", "turn3"])
    P = L.merge(U, on=["ts_code", "trade_date"])                     # picks with characteristics + outcomes
    P["excess_raw"] = P.ret_safe - P.trade_date.map(U.groupby("trade_date").ret_safe.mean())
    P["excess_matched"] = P.ret_safe - P.bench
    P["micro"] = P.mv_pct <= 0.30

    rows = []
    for nm, g in P.groupby("list"):
        day = g.groupby("trade_date")
        rows.append({"list": nm, "days": g.trade_date.nunique(), "picks": len(g),
                     **{c: round(float(g[c].mean()), 2) for c in CHAR},
                     "micro_share%": round(100 * g.micro.mean(), 1),
                     "per_trade%": round(float(day.ret_safe.mean().mean()), 2),
                     "vs_universe_pp": round(float(day.excess_raw.mean().mean()), 2),
                     "vs_style_match_pp": round(float(day.excess_matched.mean().mean()), 2),
                     "ex_micro_per_trade%": round(float(g[~g.micro].groupby("trade_date").ret_safe.mean().mean()), 2),
                     "ex_micro_vs_match_pp": round(float(g[~g.micro].groupby("trade_date").excess_matched.mean().mean()), 2)})
    T = pd.DataFrame(rows)
    # month-block bootstrap for the style-matched excess
    rng = np.random.default_rng(20261001)
    ci = {}
    for nm, g in P.groupby("list"):
        s = g.groupby("trade_date").excess_matched.mean()
        months = s.index.str[:6]
        uniq = np.unique(months)
        b = [pd.concat([s[months == m] for m in rng.choice(uniq, len(uniq))]).mean() for _ in range(2000)]
        ci[nm] = (round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2))
    T["match_CI95"] = T.list.map(ci)
    # what does the universe's own size gradient look like? (outcome by size tercile)
    grad = U.groupby("size3").ret_safe.mean().round(2).to_dict()
    mon = P.assign(month=P.trade_date.str[:6]).groupby(["month", "list"])[["mv_pct", "excess_matched"]].mean().round(2)
    text = ("## exposures (mean cross-sectional percentile, 0.5 = market median) and style-matched excess\n"
            + T.to_string(index=False)
            + f"\n\nuniverse per-trade return by size tercile (0 = smallest): {grad}"
            + "\n\n## monthly market-cap percentile and style-matched excess\n" + mon.unstack("list").to_string())
    (OUT / "style_audit.txt").write_text(text, encoding="utf-8")
    T.to_csv(OUT / "style_audit.csv", index=False, encoding="utf-8-sig")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
