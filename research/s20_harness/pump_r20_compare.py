"""Realized comparison: pump 3-way vs R20 on the same mature daily signals.

Both signals exist only in the production daily score files (2026-07 onwards),
so this is the one substrate where they can be compared on identical
next-20-session outcomes. Outcomes are computed from the local daily cache;
signals whose 20-session window has not closed are excluded.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "output/tushare_cache/daily"
SCORES_DIR = ROOT / "output/daily_pick"
HORIZON = 20


def calendar() -> list[str]:
    days = sorted(path.stem for path in DAILY.glob("*.parquet")
                  if len(path.stem) == 8 and path.stem.isdigit())
    return days


def mature_signal_dates(days: list[str]) -> list[str]:
    available = sorted(path.stem.replace("scores_", "")
                       for path in SCORES_DIR.glob("scores_*.parquet"))
    cutoff = days[-HORIZON]  # D+20 must exist in the calendar
    return [day for day in available if day <= cutoff]


def realized_outcomes(days: list[str]) -> pd.DataFrame:
    """entry = next-session open; close20/maxgain20/maxdd20 over the next 20 sessions."""
    columns = ["ts_code", "open", "high", "low", "close"]
    frames = []
    for day in days:
        path = DAILY / (day + ".parquet")
        if path.exists():
            part = pd.read_parquet(path, columns=columns)
            part["trade_date"] = day
            frames.append(part)
    daily = pd.concat(frames, ignore_index=True)
    stocks = sorted(daily.ts_code.unique())
    position = {day: index for index, day in enumerate(days)}
    column = {stock: index for index, stock in enumerate(stocks)}
    opens = np.full((len(days), len(stocks)), np.nan)
    highs = np.full((len(days), len(stocks)), np.nan)
    lows = np.full((len(days), len(stocks)), np.nan)
    closes = np.full((len(days), len(stocks)), np.nan)
    for row in daily.itertuples(index=False):
        day_index, stock_index = position[row.trade_date], column[row.ts_code]
        opens[day_index, stock_index] = row.open
        highs[day_index, stock_index] = row.high
        lows[day_index, stock_index] = row.low
        closes[day_index, stock_index] = row.close

    table_rows = []
    for day_index in range(len(days)):
        entry_index = day_index + 1
        exit_index = day_index + HORIZON
        if exit_index >= len(days):
            break
        entry = opens[entry_index]
        with np.errstate(all="ignore"):
            window_high = np.nanmax(highs[entry_index:exit_index + 1], axis=0)
            window_low = np.nanmin(lows[entry_index:exit_index + 1], axis=0)
        usable = np.isfinite(entry) & (entry > 0) & np.isfinite(closes[exit_index]) \
            & np.isfinite(window_high) & np.isfinite(window_low)
        for stock_index in np.flatnonzero(usable):
            table_rows.append({
                "ts_code": stocks[stock_index], "trade_date": days[day_index],
                "close20_ret": float(closes[exit_index, stock_index] / entry[stock_index] - 1),
                "maxgain20": float(window_high[stock_index] / entry[stock_index] - 1),
                "maxdd20": float(window_low[stock_index] / entry[stock_index] - 1)})
    table = pd.DataFrame(table_rows)
    return table[["ts_code", "trade_date", "close20_ret", "maxgain20", "maxdd20"]]


def evaluate(scores: pd.DataFrame, outcomes: pd.DataFrame, signal: str,
             event: str, k: int = 20, *, gate: bool = False) -> dict:
    merged = scores.merge(outcomes, on=["ts_code", "trade_date"], how="inner")
    merged = merged.dropna(subset=["close20_ret", "maxgain20", "maxdd20"])
    merged["event"] = _event_column(merged, event)
    if gate:
        merged = merged.dropna(subset=["r20_pred", "pred_max_gain_20", "pred_max_dd_20"])
        picked = merged[(merged.r20_pred.ge(25.0) | merged.pred_max_gain_20.ge(25.0))
                        & merged.pred_max_dd_20.ge(-15.0)]
    else:
        merged = merged.dropna(subset=[signal])
        picked = merged.sort_values(["trade_date", signal], ascending=[True, False]) \
            .groupby("trade_date", sort=False).head(k)
    per_day = picked.groupby("trade_date").event.agg(["mean", "size"])
    return {"signal": signal, "event": event, "gate": gate, "days": int(per_day.shape[0]),
            "selected": int(len(picked)),
            "pooled_precision": float(picked.event.mean()) if len(picked) else None,
            "mean_daily_precision": float(per_day["mean"].mean()) if len(per_day) else None,
            "median_daily_precision": float(per_day["mean"].median()) if len(per_day) else None,
            "daily_base_rate_mean": float(merged.groupby("trade_date").event.mean().mean()),
            "lift": (float(picked.event.mean()) / float(merged.event.mean()))
            if len(picked) and merged.event.mean() > 0 else None}


def _event_column(merged: pd.DataFrame, event: str) -> pd.Series:
    if event == "event25_safe":
        return ((merged.close20_ret.ge(0.25) | merged.maxgain20.ge(0.25))
                & merged.maxdd20.ge(-0.15))
    if event == "event20_safe":
        return ((merged.close20_ret.ge(0.20) | merged.maxgain20.ge(0.20))
                & merged.maxdd20.ge(-0.15))
    if event == "up20":
        return merged.close20_ret.ge(0.20) | merged.maxgain20.ge(0.20)
    raise ValueError("unknown event " + event)


def run(output_dir: Path) -> dict:
    days = calendar()
    signals = mature_signal_dates(days)
    if not signals:
        raise ValueError("no mature signal dates in the daily score files")
    outcomes = realized_outcomes(days)
    frames = []
    for day in signals:
        part = pd.read_parquet(SCORES_DIR / f"scores_{day}.parquet")
        part["trade_date"] = day
        frames.append(part)
    scores = pd.concat(frames, ignore_index=True)
    scores["pump_ratio"] = scores.pump_score / (scores.pump_down_score + 0.01)
    rows = []
    for event in ("event25_safe", "event20_safe", "up20"):
        rows.append(evaluate(scores, outcomes, "buy_r20_score", event))
        rows.append(evaluate(scores, outcomes, "r20_pred", event))
        rows.append(evaluate(scores, outcomes, "pump_score", event))
        rows.append(evaluate(scores, outcomes, "pump_ratio", event))
    rows.append(evaluate(scores, outcomes, "r20_pred", "event25_safe", gate=True))
    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "comparison.csv", index=False)
    return {"mature_signal_dates": signals, "signal_days": len(signals),
            "calendar_last": days[-1], "rows": int(len(scores)), "comparison": rows,
            "note": "production score files start 2026-07-01; outcomes recomputed "
                    "from the local daily cache; 20-session windows that are not "
                    "closed are excluded"}
