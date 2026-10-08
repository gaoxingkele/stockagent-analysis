#!/usr/bin/env python
"""Overextension and mean reversion around the 120-day line (user question 2026-10-07).

Two mirror hypotheses, full market, split-adjusted prices, signal days 2024-07..2026-07-28:
  H-down (found 2026-10-07): far ABOVE MA120 and RSI overbought -> clean start-down in the next 5 days
  H-up   (user):             far BELOW MA120, MA120 still RISING, RSI oversold -> rebound / mean reversion
Cells are fixed before running (no threshold search):
  distance  close/MA120 - 1:  <= -15%, (-15%,-5%], (-5%,+5%), [+5%,+15%), >= +15%
  MA120 slope: MA120 today vs 20 sessions ago (rising / falling)
  RSI14 (Wilder): <= 30 oversold, >= 70 overbought
Outcomes: method C clean start-up / start-down over 5 sessions, 5- and 10-session return from the next
open, and the S20 exit (+15/-10, 20 sessions) from the next open. The falling-MA120 twin of each cell is
the control ("falling knife"). Each cell is also reported per half-year, and as an excess over the same
day's market average, so a cell that only fires on one market day cannot pass for a pattern.
Descriptive; the reserved window is not read.
Output: output/experiments/ma120_reversion_20261007/report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from price_panel import adjusted_daily, wilder_rsi  # noqa: E402
from stockagent_analysis.pump_labels import clean_start_labels  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, exit_return  # noqa: E402

OUT = ROOT / "output/experiments/ma120_reversion_20261007"
START, END, LAST_SIGNAL = "20240102", "20260805", "20260708"


def s20_exit(px: pd.DataFrame) -> pd.Series:
    """+15% / -10% / 20 sessions from the next open, first touch on highs and lows, cost 0.3%."""
    g = px.groupby("ts_code", sort=False)
    entry = g.open.shift(-1)
    up_day = pd.Series(0, index=px.index)
    dn_day = pd.Series(0, index=px.index)
    for k in range(20, 0, -1):          # walk backwards so the earliest touch wins
        hi, lo = g.high.shift(-k), g.low.shift(-k)
        up_day = up_day.mask(hi >= entry * 1.15, k)
        dn_day = dn_day.mask(lo <= entry * 0.90, k)
    ret20 = (g.close.shift(-20) / entry - 1) * 100
    r = PureConfig().rule("U15D10")
    out = pd.Series(exit_return(up_day, dn_day, ret20, r), index=px.index)
    return out.where(g.close.shift(-20).notna())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    px = adjusted_daily(START, END)
    g = px.groupby("ts_code", sort=False)
    ma120 = g.close.transform(lambda s: s.rolling(120, min_periods=120).mean())
    px["dist"] = px.close / ma120 - 1
    px["ma120_up"] = ma120 > ma120.groupby(px.ts_code, sort=False).shift(20)
    px["rsi"] = wilder_rsi(px.close, px.ts_code)
    entry = g.open.shift(-1)
    px["r5"] = (g.close.shift(-5) / entry - 1) * 100
    px["r10"] = (g.close.shift(-10) / entry - 1) * 100
    px["s20"] = s20_exit(px)
    lab = clean_start_labels(px[["ts_code", "trade_date", "high", "low", "close", "pre_close"]])
    px = px.merge(lab[["ts_code", "trade_date", "label"]], on=["ts_code", "trade_date"])
    px = px[(px.trade_date <= LAST_SIGNAL) & px.dist.notna() & px.rsi.notna() & px.label.notna()].copy()
    px["c_up"], px["c_dn"] = (px.label == 2).astype(float), (px.label == 1).astype(float)
    for col in ("r5", "r10", "s20"):                       # same-day market excess
        px[f"x_{col}"] = px[col] - px.groupby("trade_date")[col].transform("mean")
    px["dist_band"] = pd.cut(px.dist, [-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf],
                             labels=["<=-15%", "-15..-5%", "+-5%", "+5..+15%", ">=+15%"], right=False)
    px["rsi_band"] = np.where(px.rsi <= 30, "oversold", np.where(px.rsi >= 70, "overbought", "mid"))
    px["half"] = px.trade_date.str[:4] + np.where(px.trade_date.str[4:6] <= "06", "H1", "H2")

    def row(x: pd.DataFrame, label: str) -> str:
        days = x.trade_date.nunique()
        d = x.groupby("trade_date").x_r5.mean()
        t5 = d.mean() / nw_se(d.to_numpy()) if len(d) > 20 else float("nan")
        return (f"{label:52s} n {len(x):7d} days {days:3d} | clean-up {100 * x.c_up.mean():5.1f}% clean-down {100 * x.c_dn.mean():5.1f}% "
                f"| r5 {x.r5.mean():+5.2f} (excess {x.x_r5.mean():+5.2f}, t {t5:+5.2f}) r10 {x.r10.mean():+5.2f} (ex {x.x_r10.mean():+5.2f}) "
                f"| S20 exit {x.s20.mean():+5.2f} (ex {x.x_s20.mean():+5.2f})")

    lines = [f"full market, split-adjusted, signal days {px.trade_date.min()}..{px.trade_date.max()}, {px.trade_date.nunique()} days, {len(px):,} rows",
             row(px, "all stock-days"), "", "== cells: distance x MA120 slope x RSI =="]
    cells = [("<=-15%", True, "oversold", "H-up: far below, MA120 rising, oversold"),
             ("<=-15%", False, "oversold", "  control: far below, MA120 falling, oversold"),
             ("-15..-5%", True, "oversold", "near below, MA120 rising, oversold"),
             ("-15..-5%", False, "oversold", "  control: near below, MA120 falling, oversold"),
             ("<=-15%", True, "mid", "far below, MA120 rising, RSI mid"),
             (">=+15%", True, "overbought", "H-down: far above, MA120 rising, overbought"),
             (">=+15%", False, "overbought", "  far above, MA120 falling, overbought"),
             ("+5..+15%", True, "overbought", "near above, MA120 rising, overbought")]
    picked = {}
    for band, up, rb, name in cells:
        x = px[(px.dist_band == band) & (px.ma120_up == up) & (px.rsi_band == rb)]
        picked[name] = x
        lines.append(row(x, name))
    lines += ["", "== stability: H-up and H-down by half-year (excess over the same day's market) =="]
    for name in ("H-up: far below, MA120 rising, oversold", "  control: far below, MA120 falling, oversold",
                 "H-down: far above, MA120 rising, overbought"):
        x = picked[name]
        lines.append(name.strip())
        for h, g2 in x.groupby("half"):
            lines.append(f"    {h}: n {len(g2):6d} days {g2.trade_date.nunique():3d} | clean-up {100 * g2.c_up.mean():5.1f}% clean-down {100 * g2.c_dn.mean():5.1f}% "
                         f"| excess r5 {g2.x_r5.mean():+5.2f} r10 {g2.x_r10.mean():+5.2f} S20 {g2.x_s20.mean():+5.2f}")
    lines += ["", "== dose: distance below MA120 with MA120 rising and RSI <= 30 =="]
    sub = px[px.ma120_up & (px.rsi <= 30) & (px.dist < 0)]
    for lo, hi in ((-1, -0.25), (-0.25, -0.15), (-0.15, -0.10), (-0.10, -0.05), (-0.05, 0)):
        lines.append(row(sub[(sub.dist >= lo) & (sub.dist < hi)], f"dist [{lo:+.2f},{hi:+.2f})"))
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
