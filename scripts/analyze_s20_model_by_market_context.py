#!/usr/bin/env python
"""When does the trend-chasing model earn, and when does the non-chasing one: market and sector context.

The user's hypothesis (2026-10-05): chase when the broad index is in an up-swing; do not chase, or let
sector strength decide, when it is sideways or falling. This script lines up the daily result of two
offensive lists with the context known at that day's close:

  frozen  the frozen stage1 model (picks names that already rose; 98% have a rising MA20)
  v2      the unified day-ranking model (does not chase)

Context:
  index phase   up = index 20-day and 60-day returns both positive; down = both negative; turn = mixed
  regime        the six states in output/regimes/daily_regime.parquet
  breadth       share of all stocks whose MA20 is rising
  sector layer  equal-weight industry indices built from the stock cache (a stand-in for sector ETFs):
                share of industries up over 20 days; whether the leaders of the previous 10 days kept
                leading over the last 10 (rank correlation across industries)

Three switching rules are written down before anything is computed. Days inside the frozen model's
fit and tuning windows (2025-09-01 .. 2026-01-26) are left out, so both lists are out of sample.
Descriptive only: these days have been used many times. The reserved window is not read.
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

OUT = ROOT / "output/experiments/s20_market_context_20261005"
SCORES = {
    "frozen": ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet",
    "v2": ROOT / "output/experiments/s20_unified_v2_20261005/predictions.parquet",
}


def daily_lists() -> pd.DataFrame:
    out = outcomes()[["ts_code", "trade_date", "ret_v1"]]
    days = {}
    for name, path in SCORES.items():
        scores = pd.read_parquet(path)
        picked = select(scores, PureConfig(), "U15D10")[["ts_code", "trade_date"]].merge(out, on=["ts_code", "trade_date"])
        days[name] = picked.groupby("trade_date").ret_v1.mean()
    return pd.DataFrame(days).dropna()


def sector_context() -> pd.DataFrame:
    daily = ROOT / "output/tushare_cache/daily"
    px = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "pct_chg"])
                    for p in sorted(daily.glob("*.parquet")) if "20241001" <= p.stem < RESERVED_FROM])
    px["trade_date"] = px["trade_date"].astype(str)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "industry"])
    px = px.merge(basic, on="ts_code").dropna(subset=["industry"])
    ind = px.groupby(["trade_date", "industry"]).pct_chg.mean().unstack("industry").sort_index() / 100.0
    log = np.log1p(ind.fillna(0.0))
    r20 = np.expm1(log.rolling(20).sum())
    last10 = np.expm1(log.rolling(10).sum())
    prev10 = last10.shift(10)
    ctx = pd.DataFrame(index=ind.index)
    ctx["sector_up_share"] = (r20 > 0).mean(axis=1)
    ctx["leader_persist"] = [last10.loc[d].corr(prev10.loc[d], method="spearman") for d in ind.index]
    return ctx


def table(frame: pd.DataFrame, by: pd.Series, title: str) -> list[str]:
    lines = [f"-- {title} --", f"{'state':22s} {'days':>4s} {'frozen':>7s} {'v2':>7s} {'frozen-v2':>10s} {'t':>6s}"]
    for state, g in frame.groupby(by, observed=True):
        d = g.frozen - g.v2
        se = nw_se(d.to_numpy(), lag=min(19, max(len(d) // 4, 1)))
        lines.append(f"{str(state):22s} {len(g):4d} {g.frozen.mean():+7.2f} {g.v2.mean():+7.2f} {d.mean():+10.2f} {d.mean() / se if se > 0 else float('nan'):+6.2f}")
    lines.append("")
    return lines


def book(series: pd.Series) -> str:
    month = series.groupby(series.index.str[:6]).mean()
    return (f"per trade {series.mean():+5.2f} | 2025 part {series[series.index < '20260101'].mean():+5.2f} | 2026 part {series[series.index >= '20260101'].mean():+5.2f} "
            f"| worst month {month.min():+5.2f} | months up {int((month > 0).sum())}/{len(month)}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = daily_lists()
    regime = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet")
    regime["trade_date"] = regime["trade_date"].astype(str)
    f = f.join(regime.set_index("trade_date")[["regime", "ret_20d", "ret_60d"]])
    breadth = trend_states().groupby("trade_date").ma20_up.mean().rename("breadth")
    f = f.join(breadth).join(sector_context())
    f["phase"] = np.select([(f.ret_20d > 0) & (f.ret_60d > 0), (f.ret_20d < 0) & (f.ret_60d < 0)], ["up", "down"], "turn")
    oos = f[(f.index < "20250901") | (f.index > "20260126")].copy()

    lines = [
        "Offensive-list result per day (+15%/-10% exit, list basis) for the trend-chasing frozen model and the",
        f"non-chasing v2, by the context known at the signal-day close. {len(oos)} days outside the frozen model's fit and tuning windows.",
        "",
        f"all days: frozen {book(oos.frozen)}",
        f"all days: v2     {book(oos.v2)}",
        "",
    ]
    lines += table(oos, oos.phase, "index phase (20-day and 60-day index return)")
    lines += table(oos, oos.regime, "regime label")
    lines += table(oos, pd.cut(oos.breadth, [0, 0.35, 0.5, 0.65, 1.0]), "breadth: share of stocks with a rising MA20")
    lines += table(oos, pd.cut(oos.sector_up_share, [0, 0.35, 0.65, 1.0]), "sector layer: share of industries up over 20 days")
    lines += table(oos, pd.cut(oos.leader_persist, [-1, -0.1, 0.1, 1.0]), "sector layer: did the last 10 days' leaders also lead the 10 days before")

    lines.append("-- switching rules written down before running (signal-day choice of list) --")
    books = {
        "frozen only": oos.frozen,
        "v2 only": oos.v2,
        "half and half": (oos.frozen + oos.v2) / 2,
        "R1 index up-phase -> frozen, else v2": pd.Series(np.where(oos.phase == "up", oos.frozen, oos.v2), index=oos.index),
        "R2 breadth >= 50% -> frozen, else v2": pd.Series(np.where(oos.breadth >= 0.5, oos.frozen, oos.v2), index=oos.index),
        "R3 leaders persist (>0) -> frozen, else v2": pd.Series(np.where(oos.leader_persist > 0, oos.frozen, oos.v2), index=oos.index),
    }
    for name, series in books.items():
        extra = ""
        if name.startswith("R"):
            d = series - books["half and half"]
            extra = f" | vs half-and-half {d.mean():+.2f} (t {d.mean() / nw_se(d.to_numpy()):+.2f})"
        lines.append(f"{name:44s}: {book(series)}{extra}")
    lines.append(f"share of days sent to frozen: R1 {100 * (oos.phase == 'up').mean():.0f}%  R2 {100 * (oos.breadth >= 0.5).mean():.0f}%  R3 {100 * (oos.leader_persist > 0).mean():.0f}%")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    f.to_csv(OUT / "daily_context.csv")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
