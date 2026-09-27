"""Descriptive forward-outcome stats for S20-20R Stage 1 daily Top-K picks.

Answers: from the earliest available Stage 1 scores (2025-03-03) through the
closed confirmation window (2026-08-05), how many Top10/20/50 picks had
negative subsequent movement (forward close return < 0 at +5/+10/+20
sessions; max drawdown over 20 sessions <= -5% / -10%).

Read-only descriptive analysis. Does NOT consume the reserved prospective
window (2026-08-06+), does NOT select/retune anything, and is not evidence
for advancement. Entry convention matches the frozen label pipeline:
next available session open after the signal date.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV_PRED = ROOT / "output/experiments/s20_20r_residual_portable/predictions.parquet"
CONF_PRED = ROOT / "output/experiments/s20_20r_confirmation/run_v1/predictions.parquet"
DAILY = ROOT / "output/tushare_cache/daily"
OUT = ROOT / "output/experiments/s20_20r_stage1_forward_outcomes"

KS = (10, 20, 50)
HORIZONS = (5, 10, 20)
DD_LEVELS = (-5.0, -10.0)


def load_scores() -> pd.DataFrame:
    dev = pd.read_parquet(DEV_PRED, columns=["ts_code", "trade_date", "s20_20r_platt"])
    dev = dev.rename(columns={"s20_20r_platt": "stage1_score"})
    dev["period"] = "dev_20250303_20260126"
    conf = pd.read_parquet(CONF_PRED, columns=["ts_code", "trade_date", "stage1_probability", "class20"])
    conf = conf.rename(columns={"stage1_probability": "stage1_score"})
    conf["period"] = "confirm_20260127_20260805"
    df = pd.concat([dev, conf], ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def load_bars(dates_min: str) -> pd.DataFrame:
    files = sorted(p for p in DAILY.glob("*.parquet") if p.stem >= dates_min)
    frames = [pd.read_parquet(p, columns=["ts_code", "trade_date", "open", "high", "low", "close"]) for p in files]
    bars = pd.concat(frames, ignore_index=True)
    bars["trade_date"] = bars["trade_date"].astype(str)
    bars = bars.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return bars


class Panel:
    """Per-code chronological arrays for O(1) forward-path slicing."""

    def __init__(self, bars: pd.DataFrame):
        self.by_code = {}
        for code, grp in bars.groupby("ts_code", sort=False):
            self.by_code[code] = (
                grp["trade_date"].to_numpy(),
                grp["open"].to_numpy(),
                grp["low"].to_numpy(),
                grp["close"].to_numpy(),
            )

    def forward(self, code: str, date: str):
        """Return (entry_open, closes, lows) for up to 20 sessions starting at the
        first available session strictly after the signal date, or None."""
        rec = self.by_code.get(code)
        if rec is None:
            return None
        dates, opens, lows, closes = rec
        i = np.searchsorted(dates, date, side="right")
        if i >= len(dates):
            return None
        j = min(i + 20, len(dates))
        return float(opens[i]), closes[i:j], lows[i:j]


def outcomes(panel: Panel, df: pd.DataFrame) -> pd.DataFrame:
    """Attach forward-outcome columns to each scored row."""
    n = len(df)
    rets = {k: np.full(n, np.nan) for k in HORIZONS}
    maxdd = np.full(n, np.nan)
    codes = df["ts_code"].to_numpy()
    dates = df["trade_date"].to_numpy()
    for idx in range(n):
        fwd = panel.forward(codes[idx], dates[idx])
        if fwd is None:
            continue
        entry, closes, lows = fwd
        if entry <= 0:
            continue
        for k in HORIZONS:
            if len(closes) >= k:
                rets[k][idx] = (closes[k - 1] / entry - 1.0) * 100.0
        maxdd[idx] = (lows.min() / entry - 1.0) * 100.0
    out = df.copy()
    for k in HORIZONS:
        out[f"ret{k}"] = rets[k]
    out["maxdd20"] = maxdd
    return out


def summarize(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for period in ["dev_20250303_20260126", "confirm_20260127_20260805", "all"]:
        sub = scored if period == "all" else scored[scored["period"] == period]
        uni = {"period": period, "bucket": "universe(全部打分股)", "n": len(sub)}
        for k in HORIZONS:
            uni[f"ret{k}<0%"] = float((sub[f"ret{k}"] < 0).mean() * 100)
        for lv in DD_LEVELS:
            uni[f"maxdd20<={lv:g}%"] = float((sub["maxdd20"] <= lv).mean() * 100)
        uni["ret20均值%"] = float(sub["ret20"].mean())
        rows.append(uni)
        for k in KS:
            top = sub[sub["rank"] <= k]
            row = {"period": period, "bucket": f"Top{k}", "n": len(top)}
            for h in HORIZONS:
                row[f"ret{h}<0%"] = float((top[f"ret{h}"] < 0).mean() * 100)
            for lv in DD_LEVELS:
                row[f"maxdd20<={lv:g}%"] = float((top["maxdd20"] <= lv).mean() * 100)
            row["ret20均值%"] = float(top["ret20"].mean())
            rows.append(row)
    summary = pd.DataFrame(rows).round(2)

    conf = scored[scored["period"].str.startswith("confirm")]
    strict_rows = []
    for k in KS:
        top = conf[conf["rank"] <= k]
        vc = top["class20"].value_counts()
        n = len(top)
        strict_rows.append({
            "bucket": f"Top{k}", "n": n,
            "up_first(安全+20)": int(vc.get(0, 0)), "up_first%": round(vc.get(0, 0) / n * 100, 2),
            "down_first(先触下轨)": int(vc.get(2, 0)), "down_first%": round(vc.get(2, 0) / n * 100, 2),
            "censored(两未触)": int(vc.get(1, 0)), "ambiguous(同日双触)": int(vc.get(3, 0)),
        })
    strict = pd.DataFrame(strict_rows)

    top20 = scored[scored["rank"] <= 20].copy()
    top20["month"] = top20["trade_date"].str[:6]
    monthly = (
        top20.groupby("month")
        .agg(n=("ret20", "size"), neg20_pct=("ret20", lambda s: round(float((s < 0).mean() * 100), 1)),
             mean_ret20=("ret20", lambda s: round(float(s.mean()), 2)),
             deep_dd_pct=("maxdd20", lambda s: round(float((s <= -10).mean() * 100), 1)))
        .reset_index()
    )
    return summary, strict, monthly


def main() -> int:
    scores = load_scores()
    print(f"scored rows: {len(scores):,}, dates {scores['trade_date'].min()} -> {scores['trade_date'].max()}, "
          f"{scores['trade_date'].nunique()} dates")

    scores["rank"] = scores.groupby("trade_date")["stage1_score"].rank(ascending=False, method="first")
    bars = load_bars(scores["trade_date"].min())
    panel = Panel(bars)
    scored = outcomes(panel, scores)
    na = scored["ret20"].isna().sum()
    print(f"rows without full 20-session forward path (kept, excluded from that metric): {na:,}")

    summary, strict, monthly = summarize(scored)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    strict.to_csv(OUT / "strict_path_confirm_only.csv", index=False)
    monthly.to_csv(OUT / "monthly_top20.csv", index=False)
    scored.to_parquet(OUT / "scored_with_outcomes.parquet", index=False)

    pd.set_option("display.width", 200)
    print("\n== 负样本占比（%）；n=该桶样本数 ==\n")
    print(summary.to_string(index=False))
    print("\n== 严格路径（仅确认期有 class20；先触下轨=最负面的走势） ==\n")
    print(strict.to_string(index=False))
    print("\n== Top20 逐月 ==\n")
    print(monthly.to_string(index=False))
    print(f"\nartifacts -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
