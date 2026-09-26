"""Rank-consensus composition experiments for the R20/S20/pump family.

Signals live on different scales (probabilities, ranks, scores), so the
composition is a mean of within-day percentile ranks. This is a hypothesis to
test on clean data later; here it is measured on the two available substrates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def rank_percentile(frame: pd.DataFrame, signal: str) -> pd.Series:
    return frame.groupby("trade_date")[signal].rank(pct=True)


def daily_topk_precision(frame: pd.DataFrame, score: pd.Series, k: int = 20) -> pd.DataFrame:
    ranked = frame.assign(_score=score).sort_values(["trade_date", "_score"],
                                                     ascending=[True, False])
    top = ranked.groupby("trade_date", sort=False).head(k)
    per_day = top.groupby("trade_date").positive20.agg(["mean", "size"])
    monthly = top.assign(month=top.trade_date.str[:6]).groupby("month").positive20.mean()
    return pd.DataFrame({
        "pooled_precision": [float(top.positive20.mean())],
        "mean_daily_precision": [float(per_day["mean"].mean())],
        "days": [int(per_day.shape[0])],
        "base_rate": [float(frame.positive20.mean())],
        "worst_month": [float(monthly.min())],
        "best_month": [float(monthly.max())],
        "monthly_precision": [monthly.round(4).to_dict()]})


def confirmation_experiment(output_dir: Path) -> dict:
    frame = pd.read_parquet(ROOT / "output/experiments/s20_20r_confirmation"
                            / "run_v1/predictions.parquet")
    signals = ("s20_20r_rank", "stage1_probability", "lambdarank_score", "r20_p20_reference")
    rows = []
    for name in signals:
        rows.append(daily_topk_precision(frame, rank_percentile(frame, name))
                    .assign(signal=name))
    composites = {
        "r20+stage1": ("r20_p20_reference", "stage1_probability"),
        "r20+stage1+rank": ("r20_p20_reference", "stage1_probability", "s20_20r_rank"),
        "all_four": signals,
    }
    for name, members in composites.items():
        score = sum(rank_percentile(frame, member) for member in members) / len(members)
        rows.append(daily_topk_precision(frame, score).assign(signal=name))
    table = pd.concat(rows, ignore_index=True)
    table["lift"] = table.pooled_precision / table.base_rate
    table.to_csv(output_dir / "confirmation_consensus.csv", index=False)
    return {"substrate": "s20_20r_confirmation (2026-01-27..08-05)",
            "rows": table.to_dict("records")}


def daily_experiment(output_dir: Path) -> dict:
    from research.s20_harness.pump_r20_compare import calendar, realized_outcomes

    all_days = calendar()
    outcomes = realized_outcomes(all_days)
    mature = sorted(path.stem.replace("scores_", "")
                    for path in (ROOT / "output/daily_pick").glob("scores_*.parquet"))
    cutoff = all_days[-20]
    mature = [day for day in mature if day <= cutoff]
    frames = []
    for day in mature:
        part = pd.read_parquet(ROOT / "output/daily_pick" / f"scores_{day}.parquet")
        part["trade_date"] = day
        frames.append(part)
    scores = pd.concat(frames, ignore_index=True)
    merged = scores.merge(outcomes, on=["ts_code", "trade_date"], how="inner")
    merged = merged.dropna(subset=["close20_ret", "maxgain20", "maxdd20"])

    def topk(event_values: pd.Series, score: pd.Series, k: int = 20) -> dict:
        ranked = merged.assign(_score=score).sort_values(["trade_date", "_score"],
                                                          ascending=[True, False])
        top = ranked.groupby("trade_date", sort=False).head(k)
        return {"pooled_precision": float(top.event.mean()) if len(top) else None,
                "days": int(top.trade_date.nunique()), "selected": int(len(top))}

    rows = []
    consensus = (rank_percentile(merged, "buy_r20_score")
                 + rank_percentile(merged, "pump_score")) / 2
    consensus_down_excluded = merged.assign(
        _down_rank=merged.groupby("trade_date").pump_down_score.rank(pct=True))
    for event in ("event25_safe", "event20_safe", "up20"):
        values = _event_column(merged, event)
        merged["event"] = values
        for name, score in (("r20_buy", rank_percentile(merged, "buy_r20_score")),
                            ("pump", rank_percentile(merged, "pump_score")),
                            ("r20+pump", consensus),
                            ("r20+pump,down-gated",
                             consensus.where(consensus_down_excluded._down_rank.le(0.8)))):
            row = topk(values, score)
            row.update(signal=name, event=event)
            rows.append(row)
    table = pd.DataFrame(rows)
    table["base_rate"] = [float(_event_column(merged, row.event).mean())
                          for row in table.itertuples()]
    table["lift"] = table.pooled_precision / table.base_rate
    table.to_csv(output_dir / "daily_consensus.csv", index=False)
    correlation = merged[["buy_r20_score", "pump_score", "pump_down_score"]].corr() \
        .round(3).to_dict()
    return {"substrate": "daily_pick mature days", "days": len(mature),
            "correlation": correlation, "rows": table.to_dict("records")}


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
