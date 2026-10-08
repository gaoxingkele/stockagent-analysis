"""Split-adjusted daily price panel from the Tushare daily cache, for path and trend studies.

The cache is unadjusted. Ex-rights days show up as a gap between close[t-1] and pre_close[t]; chaining
close / pre_close gives a continuous total-return index, and every price of a day is rescaled with the
same factor. Moving averages, RSI, ATR and path labels computed on this panel are free of split gaps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def adjusted_daily(start: str, end: str, codes: set[str] | None = None) -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")):
        if start <= f.stem <= end:
            x = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol"])
            x = x[x.ts_code.str.endswith((".SH", ".SZ"))]
            parts.append(x if codes is None else x[x.ts_code.isin(codes)])
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px.trade_date.astype(str)
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    step = (px.close / px.pre_close).where(px.pre_close > 0, 1.0)
    first = px.ts_code.ne(px.ts_code.shift())
    step[first] = 1.0
    idx = step.groupby(px.ts_code, sort=False).cumprod()
    first_close = px.close.where(first).groupby(px.ts_code, sort=False).transform("first")
    factor = idx * first_close / px.close          # adjusted = raw * factor, equal to raw on the first day
    for c in ("open", "high", "low", "close"):
        px[c] = px[c] * factor
    px["pre_close"] = px.groupby("ts_code", sort=False).close.shift().fillna(px.pre_close * factor)
    return px


def wilder_rsi(close: pd.Series, codes: pd.Series, n: int = 14) -> pd.Series:
    d = close.groupby(codes, sort=False).diff()
    up = d.clip(lower=0).groupby(codes, sort=False).transform(lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean())
    dn = (-d.clip(upper=0)).groupby(codes, sort=False).transform(lambda s: s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean())
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
