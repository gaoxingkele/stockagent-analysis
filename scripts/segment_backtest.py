#!/usr/bin/env python
"""Segmented backtest of the S20-Pure lists: where do the winners and the stop-outs live?

Picks of S20 v1 (U15D10) and v1.1 safe over every scorable day (dev 2025-03..2026-01
test folds, confirm 2026-01-27..08-05, shadow >= 2026-08-06 matured) are cut by one
dimension at a time - industry, board, market-cap / EP / turnover / trading-value /
volatility / MAX20 quintiles or terciles (universe percentiles on the signal day),
price level, listing age, 5- and 20-day past return, list rank, valve level, regime,
month, weekday - and each block is scored on the two objectives:
  up   = pure_up share (+15% first, no -5% shake-out)      -> maximise
  bad  = -10% touched before +5% (band-exit stop-out)      -> minimise
plus band-exit per-trade return and the same block's universe rates (so a block that
merely rides a sector beta is visible).

Pre-registered selection (dev only; confirm and shadow only confirm the sign):
  exclusion candidate: n >= 200 picks, bad >= list bad + 4pp, up <= list up,
                       month-block bootstrap 95% CI of (block bad - list bad) > 0,
                       and confirm (block bad - list bad) > 0
  overweight candidate: n >= 200, bad <= list bad - 4pp, up >= list up + 4pp, CI < 0,
                       and confirm sign agrees
Exclusion candidates are then simulated as a filter on the v1 funnel (drop + refill
from the pool) against random removal of the same count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stockagent_analysis.s20_pure import SafeConfig, select_safe  # noqa: E402
from analyze_s20_pure_r11_safe_up import load  # noqa: E402
from eval_risk_gates_and_revisions import stats, v1_funnel  # noqa: E402
from s20_pure_history import history_lists, outcomes, universe_rows, valve_all  # noqa: E402
from style_audit import characteristics  # noqa: E402

OUT = ROOT / "output/experiments/segment_backtest"
MIN_N, GAP_PP = 200, 4.0


def past_returns(dates: set[str]) -> pd.DataFrame:
    files = [p for p in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")) if p.stem >= "20241101"]
    px = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "close"]) for p in files], ignore_index=True)
    px["trade_date"] = px.trade_date.astype(str)
    px = px[px.ts_code.str.endswith((".SH", ".SZ"))].sort_values(["ts_code", "trade_date"])
    g = px.groupby("ts_code").close
    px["past_r5"], px["past_r20"] = g.pct_change(5) * 100, g.pct_change(20) * 100
    return px[px.trade_date.isin(dates)][["ts_code", "trade_date", "close", "past_r5", "past_r20"]]


def add_segments(df: pd.DataFrame, basic: pd.DataFrame, valve: pd.Series, regime: pd.Series) -> pd.DataFrame:
    d = df.merge(basic[["ts_code", "industry", "list_date"]], on="ts_code", how="left", suffixes=("", "_b"))
    if "industry" not in df.columns:
        pass
    elif d.industry.isna().all():
        d["industry"] = d.industry_b
    code = d.ts_code.str[:3]
    d["board"] = np.select([code.isin(["688", "689"]), code.isin(["300", "301"])], ["科创板", "创业板"], "主板")
    age = (pd.to_datetime(d.trade_date) - pd.to_datetime(d.list_date, errors="coerce")).dt.days / 365
    d["list_age"] = pd.cut(age, [-1, 1, 3, 100], labels=["<1年", "1-3年", ">3年"])
    d["price"] = pd.cut(d.close, [0, 10, 30, 1e9], labels=["<10元", "10-30元", ">30元"])
    for col, name in (("mv_pct", "市值"), ("ep_pct", "估值EP"), ("turn_pct", "换手"), ("amount_pct", "成交额"),
                      ("natr_pct", "波动natr"), ("max20_pct", "MAX20")):
        d[name] = pd.cut(d[col], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["Q1低", "Q2", "Q3", "Q4", "Q5高"])
    d["past_r20"] = pd.cut(d.past_r20, [-1e9, -5, 5, 15, 1e9], labels=["<-5%", "-5~5%", "5~15%", ">15%"])
    d["past_r5"] = pd.cut(d.past_r5, [-1e9, -3, 3, 1e9], labels=["<-3%", "-3~3%", ">3%"])
    d["valve"] = d.trade_date.map(valve).fillna("unknown")
    d["regime"] = d.trade_date.map(regime).fillna(-1).astype(int).astype(str)
    d["month"] = d.trade_date.str[:6]
    d["weekday"] = pd.to_datetime(d.trade_date).dt.dayofweek.map({0: "一", 1: "二", 2: "三", 3: "四", 4: "五"})
    if "list_rank" in d:
        d["名次"] = pd.cut(d.list_rank, [0, 5, 10, 20, 100], labels=["1-5", "6-10", "11-20", ">20"])
    return d


DIMS = ["industry", "board", "list_age", "price", "市值", "估值EP", "换手", "成交额", "波动natr", "MAX20",
        "past_r20", "past_r5", "valve", "regime", "month", "weekday", "名次"]


def block_table(P: pd.DataFrame, U: pd.DataFrame, dim: str, rng) -> pd.DataFrame:
    rows = []
    for per in ("dev", "confirm", "shadow"):
        L = P[P.period == per]
        if L.empty:
            continue
        base_bad, base_up = (L.cls_a5_d10 == 2).mean(), (L.state == "pure_up").mean()
        Uu = U[U.period == per]
        for blk, g in L.groupby(dim, observed=True):
            if len(g) < 30:
                continue
            ug = Uu[Uu[dim] == blk] if dim in Uu.columns else Uu.iloc[0:0]   # list-only dims (rank) have no universe twin
            row = {"dim": dim, "block": str(blk), "period": per, "n": len(g), "share%": round(100 * len(g) / len(L), 1),
                   "up%": round(100 * (g.state == "pure_up").mean(), 1), "bad%": round(100 * (g.cls_a5_d10 == 2).mean(), 1),
                   "success%": round(100 * g.cls_a5_d10.isin([1, 3]).mean(), 1),
                   "ret_band": round(float(g.ret_safe.mean()), 2), "ret_v1": round(float(g.ret_v1.mean()), 2),
                   "up_vs_list": round(100 * ((g.state == "pure_up").mean() - base_up), 1),
                   "bad_vs_list": round(100 * ((g.cls_a5_d10 == 2).mean() - base_bad), 1),
                   "univ_up%": round(100 * (ug.state == "pure_up").mean(), 1) if len(ug) else np.nan,
                   "univ_bad%": round(100 * (ug.cls_a5_d10 == 2).mean(), 1) if len(ug) else np.nan}
            if per == "dev" and len(g) >= MIN_N:
                mb = (g.cls_a5_d10 == 2).groupby(g.month).agg(["sum", "size"])
                lb = (L.cls_a5_d10 == 2).groupby(L.month).agg(["sum", "size"])
                months = mb.index.intersection(lb.index)
                diffs = []
                for _ in range(500):
                    pick = rng.choice(months, len(months), replace=True)
                    a, b = mb.loc[pick].sum(), lb.loc[pick].sum()
                    diffs.append(a["sum"] / a["size"] - b["sum"] / b["size"])
                row["bad_diff_CI95"] = (round(100 * np.percentile(diffs, 2.5), 1), round(100 * np.percentile(diffs, 97.5), 1))
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261002)
    L = history_lists()
    L = L[L.rule.isin(["U15D10", "safe_v1_1"])]
    o = outcomes()
    P = L.merge(o, on=["ts_code", "trade_date"])
    dates = set(P.trade_date)
    C = characteristics(dates)
    pr = past_returns(dates)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[["ts_code", "industry", "list_date"]]
    vt = valve_all().set_index("date").level
    rg = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet")
    regime = rg.assign(trade_date=rg.trade_date.astype(str)).set_index("trade_date").regime_id
    feats = C[["ts_code", "trade_date", "mv_pct", "ep_pct", "turn_pct", "amount_pct", "natr_pct", "max20_pct"]]
    P = P.merge(feats, on=["ts_code", "trade_date"], how="left").merge(pr, on=["ts_code", "trade_date"], how="left")
    P = add_segments(P.drop(columns=["industry"], errors="ignore"), basic, vt, regime)
    U = universe_rows(dates).merge(o, on=["ts_code", "trade_date"]).merge(feats, on=["ts_code", "trade_date"], how="left") \
        .merge(pr, on=["ts_code", "trade_date"], how="left")
    U["period"] = np.select([U.trade_date <= "20260126", U.trade_date <= "20260805"], ["dev", "confirm"], "shadow")
    U = add_segments(U, basic, vt, regime)

    lines, tables, cands = [], [], {}
    for rule, name in (("U15D10", "进攻版 v1"), ("safe_v1_1", "稳健版 v1.1")):
        Pr = P[P.rule == rule]
        base = Pr.groupby("period").apply(lambda g: pd.Series({"n": len(g), "up%": 100 * (g.state == "pure_up").mean(),
                                                               "bad%": 100 * (g.cls_a5_d10 == 2).mean(),
                                                               "ret_band": g.ret_safe.mean()})).round(2)
        lines.append(f"\n# {name}: list averages\n" + base.to_string())
        T = pd.concat([block_table(Pr, U, d, rng) for d in DIMS], ignore_index=True)
        T.insert(0, "list", name)
        tables.append(T)
        dev = T[(T.period == "dev") & T.bad_diff_CI95.notna()].copy()
        conf = T[T.period == "confirm"].set_index(["dim", "block"])
        dev["conf_bad_vs_list"] = [conf.bad_vs_list.get((a, b), np.nan) for a, b in zip(dev.dim, dev.block)]
        dev["conf_up_vs_list"] = [conf.up_vs_list.get((a, b), np.nan) for a, b in zip(dev.dim, dev.block)]
        lo, hi = zip(*dev.bad_diff_CI95)
        dev["ci_lo"], dev["ci_hi"] = lo, hi
        excl = dev[(dev.bad_vs_list >= GAP_PP) & (dev.up_vs_list <= 0) & (dev.ci_lo > 0) & (dev.conf_bad_vs_list > 0)]
        over = dev[(dev.bad_vs_list <= -GAP_PP) & (dev.up_vs_list >= GAP_PP) & (dev.ci_hi < 0) & (dev.conf_bad_vs_list < 0)]
        cands[rule] = (excl, over)
        cols = ["dim", "block", "n", "share%", "up%", "bad%", "ret_band", "up_vs_list", "bad_vs_list", "univ_up%", "univ_bad%",
                "bad_diff_CI95", "conf_bad_vs_list", "conf_up_vs_list"]
        lines.append(f"\n## {name}: dev blocks ranked by bad_vs_list (n >= {MIN_N})\n"
                     + dev.sort_values("bad_vs_list", ascending=False)[cols].head(25).to_string(index=False))
        lines.append(f"\n## {name}: dev blocks ranked by up_vs_list\n"
                     + dev.sort_values("up_vs_list", ascending=False)[cols].head(15).to_string(index=False))
        lines.append(f"\n## {name}: EXCLUSION candidates (pre-registered rule)\n" + (excl[cols].to_string(index=False) if len(excl) else "none"))
        lines.append(f"\n## {name}: OVERWEIGHT candidates\n" + (over[cols].to_string(index=False) if len(over) else "none"))
    pd.concat(tables, ignore_index=True).to_csv(OUT / "blocks.csv", index=False, encoding="utf-8-sig")

    # simulate exclusion candidates on the v1 funnel (dev + confirm), random removal as control
    excl, _ = cands["U15D10"]
    if len(excl):
        p = load()
        p = p[p.trade_date <= "20260805"].copy()
        path = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/path_panel.parquet",
                               columns=["ts_code", "trade_date", "up15_day", "dn10_day"])
        p = p.merge(path, on=["ts_code", "trade_date"], how="left").merge(feats, on=["ts_code", "trade_date"], how="left") \
            .merge(pr, on=["ts_code", "trade_date"], how="left")
        p["ret"] = p.ret_a5_b15_d10 - 0.3
        p = add_segments(p.drop(columns=["industry"], errors="ignore"), basic, vt, regime)
        flag = np.zeros(len(p), bool)
        for d_, b_ in zip(excl.dim, excl.block):
            flag |= (p[d_].astype(str) == b_).to_numpy()
        p["seg_excl"] = flag
        p["period"] = np.where(p.trade_date <= "20260126", "dev", "confirm")
        p = p.reset_index(drop=True)
        base = v1_funnel(p)
        gated = v1_funnel(p, drop_col="seg_excl")
        removed = base[base.seg_excl]
        ctrl = []
        for _ in range(100):
            drop = []
            for d, n in removed.groupby("trade_date").size().items():
                cand = base.index[base.trade_date == d]
                drop += list(rng.choice(cand, min(n, len(cand)), replace=False))
            ctrl.append(v1_funnel(p, drop_idx=set(drop)))
        rows = []
        for per in ("dev", "confirm"):
            s0, s1 = stats(base[base.period == per]), stats(gated[gated.period == per])
            sc = [stats(c[c.period == per]) for c in ctrl]
            rows.append({"period": per, "removed_per_day": round(len(removed[removed.period == per]) / max(1, base[base.period == per].trade_date.nunique()), 2),
                         "ret_base": s0["per_trade%"], "ret_gated": s1["per_trade%"], "ret_random": round(float(np.mean([x["per_trade%"] for x in sc])), 3),
                         "bad_base": s0["bad%"], "bad_gated": s1["bad%"], "bad_random": round(float(np.mean([x["bad%"] for x in sc])), 2),
                         "success_base": s0["success%"], "success_gated": s1["success%"]})
        lines.append("\n# v1 funnel with the exclusion candidates applied (refill from pool; random removal = control)\n"
                     + pd.DataFrame(rows).to_string(index=False))
    text = "\n".join(lines)
    (OUT / "segment_backtest.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
