"""1H 向下钻取二次验证 (Sprint 4.5, V12.17).

核心思想 (用户 0517 创新):
  - 日线 V12 选股做"宽度筛选"(全市场 5000+)
  - 1H 信号做"深度二次验证"(仅对推荐池 50-100 只)
  - 实战意义: 日线选股 + 1H 选时

精简实施 (不训练 1H 模型, 用规则信号):
  - fetch_1h_data(ts_code, lookback_days): 拉 1H OHLCV
  - compute_1h_signals(df): 1H_MA20/RSI/MACD/趋势方向/量能
  - score_intraday(signals): 输出 1H 综合分 [-1, +1]
  - dual_consensus(v12_rec, intraday): 日线 + 1H 共识

输出:
  - intraday_signal: 1H 综合分
  - intraday_trend: up/down/sideways
  - intraday_rsi: 1H RSI(14)
  - dual_consensus: strong_buy / hold / strong_avoid
"""
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def fetch_1h_data(ts_code: str, lookback_days: int = 10, end_date: Optional[str] = None) -> pd.DataFrame:
    """拉 1H 数据 (Tushare pro_bar)."""
    import tushare as ts
    from datetime import datetime, timedelta
    if "TUSHARE_TOKEN" not in os.environ:
        from dotenv import load_dotenv
        load_dotenv()
    ts.set_token(os.environ["TUSHARE_TOKEN"])

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    # 多取一些, 防止节假日
    start_dt = end_dt - timedelta(days=lookback_days + 14)
    start_str = start_dt.strftime("%Y%m%d") + " 09:00:00"
    end_str = end_dt.strftime("%Y%m%d") + " 15:00:00"

    try:
        df = ts.pro_bar(ts_code=ts_code, freq="60min",
                         start_date=start_str, end_date=end_str)
        if df is None or df.empty: return pd.DataFrame()
        df = df.sort_values("trade_time").reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def compute_1h_signals(df: pd.DataFrame) -> dict:
    """从 1H OHLC 计算信号."""
    if df.empty or len(df) < 20:
        return {"valid": False, "reason": "数据不足"}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["vol"].values

    # MA
    ma5 = pd.Series(close).rolling(5).mean().iloc[-1]
    ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    ma_slope_20 = (pd.Series(close).rolling(20).mean().iloc[-1] /
                    pd.Series(close).rolling(20).mean().iloc[-5] - 1) * 100 \
                   if len(close) >= 25 else 0

    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
    rs = gain / loss if loss > 0 else (10 if gain > 0 else 0)
    rsi14 = 100 - 100 / (1 + rs)

    # 量能放大 (最近 3 bar avg vs 过去 20 bar avg)
    vol_recent = float(np.mean(vol[-3:]))
    vol_baseline = float(np.mean(vol[-23:-3])) if len(vol) >= 23 else float(np.mean(vol[:-3]))
    vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1.0

    # 趋势方向
    cur_close = float(close[-1])
    if cur_close > ma20 * 1.01 and ma_slope_20 > 0.3:
        trend = "up"
    elif cur_close < ma20 * 0.99 and ma_slope_20 < -0.3:
        trend = "down"
    else:
        trend = "sideways"

    return {
        "valid": True,
        "n_bars": len(df),
        "close": cur_close,
        "ma5": float(ma5), "ma20": float(ma20),
        "ma_slope_20_pct": float(ma_slope_20),
        "rsi14": float(rsi14),
        "vol_ratio_3v20": float(vol_ratio),
        "trend": trend,
        "last_bar_time": str(df["trade_time"].iloc[-1]),
    }


def score_intraday(sig: dict) -> float:
    """1H 综合分 [-1, +1]."""
    if not sig.get("valid"):
        return 0.0
    score = 0.0
    # 趋势
    if sig["trend"] == "up": score += 0.4
    elif sig["trend"] == "down": score -= 0.4
    # RSI
    rsi = sig["rsi14"]
    if rsi < 30: score += 0.3   # 超卖反转机会
    elif rsi > 75: score -= 0.3   # 超买
    elif 50 < rsi < 65: score += 0.2   # 健康上行
    # MA 关系
    if sig["close"] > sig["ma20"]: score += 0.2
    else: score -= 0.2
    # MA20 斜率
    slope = sig["ma_slope_20_pct"]
    if slope > 1.0: score += 0.2
    elif slope < -1.0: score -= 0.2
    # 量能
    if sig["vol_ratio_3v20"] > 1.5 and sig["trend"] == "up":
        score += 0.2   # 上涨放量
    elif sig["vol_ratio_3v20"] > 1.5 and sig["trend"] == "down":
        score -= 0.2   # 下跌放量
    return float(np.clip(score, -1.0, 1.0))


def dual_consensus(v12_buy_r5_score: float, intraday_score: float) -> str:
    """日线 r5 buy_score 与 1H 综合分双引擎共识."""
    # 日线 r5 buy_score [0, 100], 1H 分 [-1, +1]
    daily_strong = v12_buy_r5_score >= 65
    daily_weak = v12_buy_r5_score < 35
    intraday_strong = intraday_score >= 0.4
    intraday_weak = intraday_score <= -0.3

    if daily_strong and intraday_strong:
        return "strong_buy"
    if daily_weak and intraday_weak:
        return "strong_avoid"
    if daily_strong and intraday_weak:
        return "wait_pullback"   # 日线看好但 1H 弱, 等回调
    if daily_weak and intraday_strong:
        return "short_pulse"     # 日线弱但 1H 强, 短线脉冲
    return "neutral"


def drill_down_analyze(ts_codes: list[str], end_date: str,
                        v12_scores: Optional[dict] = None,
                        lookback_days: int = 10,
                        progress_cb=None) -> pd.DataFrame:
    """对一批股票做 1H 二次验证.

    Args:
      ts_codes: V12 推荐池股票列表
      end_date: 评估日期 YYYYMMDD
      v12_scores: {ts_code: buy_r5_score} 日线 V12 分数
      lookback_days: 1H 历史回看天数 (默认 10 ≈ 40 bar)

    Returns:
      DataFrame: ts_code, intraday_score, trend, rsi14, dual_consensus
    """
    rows = []
    n = len(ts_codes)
    t0 = time.time()
    for i, ts in enumerate(ts_codes, 1):
        try:
            df = fetch_1h_data(ts, lookback_days=lookback_days, end_date=end_date)
            sig = compute_1h_signals(df)
            score = score_intraday(sig)
            daily_r5 = (v12_scores or {}).get(ts, 50.0)
            consensus = dual_consensus(daily_r5, score)
            rows.append({
                "ts_code": ts,
                "daily_buy_r5": daily_r5,
                "intraday_score": round(score, 3),
                "intraday_trend": sig.get("trend"),
                "intraday_rsi14": round(sig.get("rsi14", 0), 1),
                "intraday_ma_slope_pct": round(sig.get("ma_slope_20_pct", 0), 2),
                "intraday_vol_ratio": round(sig.get("vol_ratio_3v20", 0), 2),
                "dual_consensus": consensus,
                "n_bars": sig.get("n_bars", 0),
            })
        except Exception as e:
            rows.append({"ts_code": ts, "error": str(e)[:80]})
        if progress_cb and i % 5 == 0:
            progress_cb(i, n, time.time() - t0)
        # 限速
        time.sleep(0.3)   # ~200/min 限速
    return pd.DataFrame(rows)
