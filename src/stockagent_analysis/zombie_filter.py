"""僵尸区过滤器 (来自量化访谈文章: 88% 交易员共识).

僵尸区定义:
  1. 过去 N 日中, |close - MA60| / MA60 < threshold_pct 的天数比例 >= ratio_threshold
  2. MA60 短期斜率 <= slope_threshold (走平或微跌)

实战含义:
  价格在 MA60 ±5% 区间内横盘 N≥20 日 + 趋势缺失 = 流动性弱 + 信号弱
  追入大概率被反复打脸 (止损 → 反弹 → 再追 → 再止损)

接入方式:
  - V12 V7c 5 铁律之外的第 6 条过滤器: NOT is_zombie
  - 或加进 factor_lab 232 因子池让 LightGBM 自动学
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


def compute_zombie_factors(
    daily_df: pd.DataFrame,
    lookback: int = 20,
    deviation_pct: float = 0.05,
    slope_window: int = 10,
    ratio_threshold: float = 0.9,
    slope_threshold: float = 0.003,
) -> pd.DataFrame:
    """
    输入: daily_df 含列 [trade_date, close]
    输出: 每行加 4 个新列:
      - ma60: 60日均线
      - zombie_days_pct: 过去 lookback 天处于 zombie 区的比例
      - ma60_slope_short: MA60 的 slope_window 天斜率
      - is_zombie: bool, 满足 ratio_threshold + slope_threshold
    """
    df = daily_df.copy().sort_values("trade_date").reset_index(drop=True)
    close = df["close"]
    ma60 = close.rolling(60, min_periods=30).mean()
    df["ma60"] = ma60

    deviation = (close - ma60).abs() / ma60
    in_zone = (deviation < deviation_pct).astype(float)
    df["zombie_days_pct"] = in_zone.rolling(lookback, min_periods=10).mean()

    df["ma60_slope_short"] = ma60.pct_change(slope_window)

    df["is_zombie"] = (
        (df["zombie_days_pct"] >= ratio_threshold)
        & (df["ma60_slope_short"] <= slope_threshold)
        & ma60.notna()
    )
    return df


def evaluate_zombie_on_stocks(
    daily_cache_dir: str,
    ts_codes: list[str],
    end_date: str,
    **kwargs,
) -> pd.DataFrame:
    """对一批 ts_code 在 end_date 时点判断是否为 zombie."""
    from pathlib import Path

    cache = Path(daily_cache_dir)
    files = sorted(cache.glob("*.parquet"))
    end_int = int(end_date)
    parts = [pd.read_parquet(f) for f in files if int(f.stem) <= end_int]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    results = []
    for ts in ts_codes:
        d = big[big["ts_code"] == ts]
        if d.empty or len(d) < 80:
            results.append({
                "ts_code": ts, "is_zombie": None,
                "zombie_days_pct": None, "ma60_slope_short": None,
                "ma60": None, "close": None, "deviation_to_ma60_pct": None,
            })
            continue
        z = compute_zombie_factors(d, **kwargs)
        last = z[z["trade_date"] == end_date]
        if last.empty:
            last = z.tail(1)
        r = last.iloc[0]
        results.append({
            "ts_code": ts,
            "close": float(r["close"]),
            "ma60": float(r["ma60"]) if pd.notna(r["ma60"]) else None,
            "deviation_to_ma60_pct": float(
                (r["close"] - r["ma60"]) / r["ma60"] * 100
            ) if pd.notna(r["ma60"]) else None,
            "zombie_days_pct": float(r["zombie_days_pct"]) if pd.notna(r["zombie_days_pct"]) else None,
            "ma60_slope_short_pct": float(r["ma60_slope_short"] * 100) if pd.notna(r["ma60_slope_short"]) else None,
            "is_zombie": bool(r["is_zombie"]),
        })
    return pd.DataFrame(results)
