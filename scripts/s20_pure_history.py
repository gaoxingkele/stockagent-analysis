#!/usr/bin/env python
"""S20-Pure lists and outcomes over every day we can score, on one yardstick.

Coverage (stage1 has out-of-sample scores only on these days):
  dev      2025-03..04, 2025-07..08, 2025-11..2026-01   walk-forward test folds;
           the funnel parameters (cap, band, valve cut-offs) were CHOSEN here
  confirm  2026-01-27..2026-08-05                        seen once, descriptive
  shadow   2026-08-06..                                   after every freeze, true OOS
Missing 2025 months (01-02, 05-06, 09-10) were stage1 training/tuning segments.

Lists: v1 aggressive (U15D10), v1.1 safe, v1 as served with valve action B,
and the same-day universe. Every list is scored twice:
  safe-up view   band exit +5%/+15%, crash line -10% (success / bad / 15% drawdown)
  v1 view        take-profit +15% / stop -10% / 20 sessions
Book: one sleeve per signal day at 1/20 capital (cash when a list is empty).

Import `history_lists()` / `score()` from other scripts (the R20 comparison).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from stockagent_analysis.market_valve import ValveConfig, apply_actions, daily_breadth, valve_levels  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, SafeConfig, exit_return, select, select_safe, three_state  # noqa: E402

EXP = ROOT / "output/experiments/s20_pure_20260928"
SHADOW = ROOT / "output/experiments/s20_pure_v1_shadow"
OUT = ROOT / "output/experiments/s20_pure_history"


def period_of(d: pd.Series) -> pd.Series:
    return pd.Series(np.select([d <= "20260126", d <= "20260805"], ["dev", "confirm"], "shadow"), index=d.index)


def valve_all() -> pd.DataFrame:
    files = sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    b = daily_breadth(pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "pct_chg"]) for p in files]))
    live = ROOT / "output/jev_market/live_market_days.csv"
    if live.exists():
        lv = pd.read_csv(live, dtype={"date": str})[["date", "limit_down"]].rename(columns={"date": "trade_date"})
        b = pd.concat([b, lv[~lv.trade_date.isin(b.trade_date)]], ignore_index=True)
    return valve_levels(b, ValveConfig())


def history_lists() -> pd.DataFrame:
    """Raw v1 (U15D10) + safe lists for every scorable day, then B-served v1."""
    from analyze_s20_pure_r11_safe_up import load
    p = load()                                                     # dev + confirm, stage1 OOF scores
    v1 = select(p, PureConfig(), "U15D10")
    sf = select_safe(p, SafeConfig())
    cols = ["trade_date", "rule", "list_rank", "ts_code", "industry"]
    raw = pd.concat([v1[cols], sf[cols]], ignore_index=True)
    sh = pd.read_csv(SHADOW / "daily_lists_raw.csv", dtype={"trade_date": str, "ts_code": str})
    sh = sh[sh.rule.isin(["U15D10", "safe_v1_1"]) & (sh.trade_date > raw.trade_date.max())]
    raw = pd.concat([raw, sh[cols]], ignore_index=True)
    served = apply_actions(raw, valve_all().rename(columns={"trade_date": "date"}), ValveConfig(),
                           aggressive_rules=("U15D10",))
    b = served[served.rule == "U15D10"].assign(rule="U15D10+B")
    out = pd.concat([raw.assign(served_by=raw.rule), b], ignore_index=True)
    out["period"] = period_of(out.trade_date)
    return out


def outcomes() -> pd.DataFrame:
    band = pd.read_parquet(EXP / "band_panel.parquet",
                           columns=["ts_code", "trade_date", "maxdd20", "ret20", "cls_a5_d10", "ret_a5_b15_d10"])
    path = pd.read_parquet(EXP / "path_panel.parquet", columns=["ts_code", "trade_date", "up15_day", "dn10_day", "dn5_day"])
    o = band.merge(path, on=["ts_code", "trade_date"])
    r = PureConfig().rule("U15D10")
    o["ret_v1"] = exit_return(o.up15_day, o.dn10_day, o.ret20, r)
    o["ret_safe"] = o.ret_a5_b15_d10 - SafeConfig().cost_pct
    o["state"] = three_state(o.up15_day, o.dn10_day, o.dn5_day)
    return o


def score(q: pd.DataFrame) -> dict:
    """q: list rows merged with outcomes() (one row per pick)."""
    if q.empty:
        return {"days": 0}
    day_safe = q.groupby("trade_date").ret_safe.mean()
    day_v1 = q.groupby("trade_date").ret_v1.mean()
    book = day_safe.sort_index().cumsum() / 20
    return {"days": q.trade_date.nunique(), "avg_len": round(len(q) / q.trade_date.nunique(), 1),
            "success%": round(100 * q.cls_a5_d10.isin([1, 3]).mean(), 1),
            "bad%": round(100 * (q.cls_a5_d10 == 2).mean(), 1),
            "drawdown15%": round(100 * (q.maxdd20 <= -15).mean(), 1),
            "band_mean%": round(float(day_safe.mean()), 2),
            "v1rule_mean%": round(float(day_v1.mean()), 2),
            "pure_up%": round(100 * (q.state == "pure_up").mean(), 1),
            "pos_days%": round(100 * (day_safe > 0).mean(), 1),
            "worst_month%": round(float(day_safe.groupby(day_safe.index.str[:6]).mean().min()), 2),
            "book_maxDD%": round(float((book - book.cummax()).min()), 2)}


def universe_rows(dates) -> pd.DataFrame:
    s1 = pd.read_parquet(ROOT / "output/experiments/s20_20r_stage1_forward_outcomes/scored_with_outcomes.parquet",
                         columns=["ts_code", "trade_date"])
    u = s1[s1.trade_date.isin(set(dates))]
    extra = sorted(set(dates) - set(u.trade_date))
    if extra:   # shadow days: whole SH/SZ market from the band panel
        b = pd.read_parquet(EXP / "band_panel.parquet", columns=["ts_code", "trade_date"])
        u = pd.concat([u, b[b.trade_date.isin(extra)]], ignore_index=True)
    return u.assign(rule="universe")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    L = history_lists()
    o = outcomes()
    M = L.merge(o, on=["ts_code", "trade_date"])            # matured picks only
    U = universe_rows(M.trade_date.unique()).merge(o, on=["ts_code", "trade_date"])
    U["period"] = period_of(U.trade_date)
    M = pd.concat([M, U], ignore_index=True)
    names = {"U15D10": "进攻版 v1", "U15D10+B": "进攻版 v1 + 阀门B", "safe_v1_1": "稳健版 v1.1", "universe": "全市场"}
    rows = []
    for per in ("dev", "confirm", "shadow", "all"):
        g = M if per == "all" else M[M.period == per]
        for rule, nm in names.items():
            rows.append({"period": per, "list": nm, **score(g[g.rule == rule])})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    mon = []
    for (mo, rule), g in M.assign(month=M.trade_date.str[:6]).groupby(["month", "rule"]):
        s = score(g)
        mon.append({"month": mo, "list": names[rule], "band_mean%": s["band_mean%"], "bad%": s["bad%"],
                    "success%": s["success%"]})
    Mo = pd.DataFrame(mon).pivot(index="month", columns="list", values="band_mean%")
    Mo.to_csv(OUT / "monthly_band_mean.csv", encoding="utf-8-sig")
    text = (f"matured signal days {M.trade_date.min()}..{M.trade_date.max()}\n\n" + T.to_string(index=False)
            + "\n\n## monthly mean per-trade return, safe-up band exit (%)\n" + Mo.round(2).to_string())
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
