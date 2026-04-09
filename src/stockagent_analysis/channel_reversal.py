# -*- coding: utf-8 -*-
"""通道反转指标 — 基于平滑Donchian通道 + RSI + 量能的阶段判定与评分。

通道定义:
  upper = WMA(rolling_max(high, 100), 60)
  lower = WMA(rolling_min(low, 100), 60)
  middle = (upper + lower) / 2

阶段状态机:
  Phase 0U/0D — 通道内上/下半区
  Phase 1    — 破下轨 (超卖触发)
  Phase 2    — 回到轨道 (反弹开始)
  Phase 3A   — 弱反弹 (量弱+RSI未脱超卖, 大概率破新低)
  Phase 3B   — 过中轨 (量增+RSI回升, 看多初步确认)
  Phase 4A   — 有效反转 (连续3日站上上轨+放量)
  Phase 4B   — 上轨整理 (触及上轨未能有效突破, 小中枢)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 通道计算 ────────────────────────────────────────────────

def wma(series: pd.Series, period: int) -> pd.Series:
    """加权移动平均 (Weighted Moving Average)。"""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def compute_channel(df: pd.DataFrame, lookback: int = 100, smooth: int = 60) -> pd.DataFrame:
    """计算通道上/中/下轨。

    Parameters
    ----------
    df : DataFrame with columns: high, low, close, volume
    lookback : rolling max/min 窗口 (default 100)
    smooth : WMA 平滑周期 (default 60)

    Returns
    -------
    DataFrame with added columns: ch_upper, ch_lower, ch_middle
    """
    df = df.copy()
    df["_roll_max"] = df["high"].rolling(lookback, min_periods=lookback).max()
    df["_roll_min"] = df["low"].rolling(lookback, min_periods=lookback).min()
    df["ch_upper"] = wma(df["_roll_max"], smooth)
    df["ch_lower"] = wma(df["_roll_min"], smooth)
    df["ch_middle"] = (df["ch_upper"] + df["ch_lower"]) / 2
    df.drop(columns=["_roll_max", "_roll_min"], inplace=True)
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI(14)。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ── 阶段判定 ────────────────────────────────────────────────

# Phase 常量
P_0U = "0U"      # 上半区 (close > middle)
P_0D = "0D"      # 下半区 (close < middle)
P_1 = "1"        # 破下轨
P_2 = "2"        # 回到轨道
P_3A = "3A"      # 弱反弹
P_3B = "3B"      # 过中轨
P_4A = "4A"      # 有效反转
P_4B = "4B"      # 上轨整理

# 超时阈值
TIMEOUT_P1 = 10   # Phase 1 持续 > 10日不回轨 → 退化0D
TIMEOUT_P2 = 20   # Phase 2 持续 > 20日未过中轨 → 3A
TIMEOUT_P3B = 15  # Phase 3B 持续 > 15日未触上轨 → 4B
TIMEOUT_P4B = 20  # Phase 4B 整理 > 20日 → 0U


def detect_phases(df: pd.DataFrame) -> pd.DataFrame:
    """逐行判定通道反转阶段。

    输入 df 需已包含: close, high, low, volume, ch_upper, ch_lower, ch_middle
    输出新增列: phase, phase_days, rsi, rsi_slope, vol_ratio, cr_score
    """
    df = df.copy()
    n = len(df)

    # 预计算
    df["rsi"] = compute_rsi(df["close"])
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=10).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    # RSI 5日斜率 (线性回归斜率)
    rsi_vals = df["rsi"].values
    rsi_slope = np.full(n, np.nan)
    for i in range(4, n):
        window = rsi_vals[i - 4:i + 1]
        if not np.any(np.isnan(window)):
            x = np.arange(5, dtype=float)
            rsi_slope[i] = np.polyfit(x, window, 1)[0]
    df["rsi_slope"] = rsi_slope

    # 阶段判定
    phases = [None] * n
    phase_days = np.zeros(n, dtype=int)
    scores = np.zeros(n, dtype=float)

    prev_phase = None
    prev_days = 0
    prev_above_middle = None  # 上一次在middle哪侧

    for i in range(n):
        close = df.iloc[i]["close"]
        upper = df.iloc[i]["ch_upper"]
        lower = df.iloc[i]["ch_lower"]
        middle = df.iloc[i]["ch_middle"]
        rsi = df.iloc[i]["rsi"]
        slope = df.iloc[i]["rsi_slope"]
        vol_r = df.iloc[i]["vol_ratio"]

        if pd.isna(upper) or pd.isna(lower) or pd.isna(middle):
            phases[i] = None
            scores[i] = 50
            continue

        above_middle = close > middle

        # ── 状态转移逻辑 ──
        phase = prev_phase
        days = prev_days + 1

        # Phase 1: 破下轨
        if close < lower:
            if prev_phase != P_1:
                phase = P_1
                days = 1
            else:
                # 超时检查
                if days > TIMEOUT_P1:
                    phase = P_0D
                    days = 1

        # Phase 2: 从Phase 1回到轨道
        elif prev_phase == P_1 and close >= lower:
            phase = P_2
            days = 1

        # Phase 2 → 3A / 3B
        elif prev_phase == P_2:
            if close > middle:
                phase = P_3B
                days = 1
            elif days > TIMEOUT_P2:
                phase = P_3A
                days = 1
            elif vol_r < 0.8 and (pd.isna(rsi) or rsi < 35) and days > 5:
                # 量弱 + RSI仍低迷 + 已经反弹5日以上还没过中轨
                phase = P_3A
                days = 1

        # Phase 3A: 弱反弹 (终态，除非重新破下轨触发新Phase 1)
        elif prev_phase == P_3A:
            if close < lower:
                phase = P_1
                days = 1
            elif close > middle:
                phase = P_3B  # 意外走强
                days = 1

        # Phase 3B → 4A / 4B
        elif prev_phase == P_3B:
            if close > upper:
                # 检查是否连续3日站上upper
                if i >= 2:
                    c1 = df.iloc[i - 1]["close"] > df.iloc[i - 1]["ch_upper"]
                    c2 = df.iloc[i - 2]["close"] > df.iloc[i - 2]["ch_upper"]
                    if c1 and c2:
                        phase = P_4A
                        days = 1
            if phase == P_3B and days > TIMEOUT_P3B:
                phase = P_4B
                days = 1

        # Phase 4A: 有效反转 (终态，保持)
        elif prev_phase == P_4A:
            if close < middle:
                phase = P_0D
                days = 1
            # 保持4A

        # Phase 4B: 上轨整理
        elif prev_phase == P_4B:
            # 如果站上upper 3日→升级4A
            if close > upper and i >= 2:
                c1 = df.iloc[i - 1]["close"] > df.iloc[i - 1]["ch_upper"]
                c2 = df.iloc[i - 2]["close"] > df.iloc[i - 2]["ch_upper"]
                if c1 and c2:
                    phase = P_4A
                    days = 1
            elif days > TIMEOUT_P4B:
                phase = P_0U
                days = 1
            elif close < middle:
                phase = P_0D
                days = 1

        # Phase 0: 通道内方向判定
        elif phase is None or prev_phase in (P_0U, P_0D):
            if close < lower:
                phase = P_1
                days = 1
            elif above_middle:
                if prev_above_middle is not None and not prev_above_middle:
                    # 从下半区进入上半区 → 转多头
                    phase = P_0U
                    days = 1
                elif prev_phase == P_0D:
                    phase = P_0U
                    days = 1
                else:
                    phase = P_0U
            else:
                if prev_above_middle is not None and prev_above_middle:
                    # 从上半区跌入下半区 → 空头形成
                    phase = P_0D
                    days = 1
                elif prev_phase == P_0U:
                    phase = P_0D
                    days = 1
                else:
                    phase = P_0D

        # 兜底
        if phase is None:
            phase = P_0U if above_middle else P_0D
            days = 1

        phases[i] = phase
        phase_days[i] = days
        prev_phase = phase
        prev_days = days
        prev_above_middle = above_middle

        # ── 通道位置比例: 中轨=0, 上轨=+1, 下轨=-1 ──
        ch_width_up = upper - middle if upper > middle else 1e-9
        ch_width_dn = middle - lower if middle > lower else 1e-9
        if close >= middle:
            ch_pos = min(1.0, (close - middle) / ch_width_up)
        else:
            ch_pos = max(-1.0, -(middle - close) / ch_width_dn)

        # ── 评分 ──
        scores[i] = _calc_score(phase, rsi, slope, vol_r, days, ch_pos)

    df["phase"] = phases
    df["phase_days"] = phase_days
    df["cr_score"] = scores
    return df


def _calc_score(phase: str, rsi: float, rsi_slope: float, vol_ratio: float, days: int, ch_pos: float = 0.0) -> float:
    """根据阶段 + RSI + 量能 + 通道位置计算评分。

    ch_pos: 通道位置比例, 中轨=0, 上轨=+1, 下轨=-1
    P_0U/P_0D: 按通道位置线性插值, 中轨50 → 上轨80 / 下轨20
    其他阶段: 保持原有逻辑(特殊事件信号)
    """
    rsi = rsi if not pd.isna(rsi) else 50
    rsi_slope = rsi_slope if not pd.isna(rsi_slope) else 0
    vol_ratio = vol_ratio if not pd.isna(vol_ratio) else 1.0

    score = 50.0  # 默认

    # 评分基于146只股票48039样本回测校准(2026-04-05):
    #   4B胜率69%/20日+12.1%(最佳) > 3B胜率56%/+7.9% > 1破下轨胜率61.5%
    #   4A胜率仅50%(站上上轨后短期超买) > 3A胜率56%(实际不弱)
    if phase in (P_0U, P_0D):
        # 通道内均值回归: 靠近下轨→超跌反弹机会(高分), 靠近上轨→回调风险(低分)
        # 下轨(-1)=80分, 中轨(0)=50分, 上轨(+1)=20分
        score = 50 - ch_pos * 30
    elif phase == P_1:
        # 破下轨: 回测胜率61.5%，均值回归效应强，是反转买点
        # 越远离下轨(ch_pos越负)→反弹空间越大→分数越高
        score = 50 - ch_pos * 30
        if rsi < 20:
            score += 8   # 深度超卖，反转潜力大
        elif rsi < 30:
            score += 5   # 超卖
        if vol_ratio > 1.3:
            score += 5   # 破下轨+放量=恐慌释放
    elif phase == P_2:
        # 从破下轨回到轨道内, 仍在下半区, 均值回归看多
        score = 50 - ch_pos * 30
        # RSI从超卖回升额外加分
        if rsi < 30 and rsi_slope > 0:
            score += 8
        elif rsi >= 30 and rsi_slope > 1:
            score += 6
        if vol_ratio > 1.2:
            score += 4
    elif phase == P_3A:
        # 弱反弹: 回测胜率56%/20日+5.36%
        score = 50 - ch_pos * 30 - 5  # 位置基础上略扣分
        if rsi < 30:
            score -= 3
    elif phase == P_3B:
        # 过中轨: 回测胜率56%/20日+7.91%，可靠的看多确认信号
        score = 68
        if rsi > 40 and rsi_slope > 0.5:
            score += 8
        elif rsi > 35 and rsi_slope > 0:
            score += 4
        if vol_ratio > 1.2:
            score += 5
        if rsi > 40 and rsi_slope > 1.0 and vol_ratio > 1.2:
            score = max(score, 82)
    elif phase == P_4A:
        # 有效反转: 站上上轨
        score = 72
        if rsi > 50 and rsi_slope > 0:
            score += 3
        if vol_ratio > 1.5:
            score += 3
        if days > 5:
            score -= 5
        score = min(80, score)
    elif phase == P_4B:
        # 上轨整理: 回测最佳信号
        score = 78
        if vol_ratio > 1.0:
            score += 4
        if rsi > 45 and rsi_slope > 0:
            score += 4
        if days > 15:
            score -= 5
        score = min(88, score)

    return max(5, min(95, score))


# ── 回测接口 ────────────────────────────────────────────────

def analyze_symbol(symbol: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """对单只股票计算通道反转指标。

    Parameters
    ----------
    symbol : 6位股票代码
    df : 可选，已有日线DataFrame (需含 open/high/low/close/volume)
         如果不传，从TDX本地数据读取

    Returns
    -------
    DataFrame with channel + phase + score columns
    """
    if df is None:
        df = _load_daily(symbol)
    if df is None or len(df) < 160:
        return pd.DataFrame()

    # 标准化列名
    col_map = {
        "收盘": "close", "开盘": "open", "最高": "high",
        "最低": "low", "成交量": "volume", "日期": "date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = compute_channel(df)
    df = detect_phases(df)
    return df


def backtest_phases(df: pd.DataFrame, forward_days: list[int] | None = None) -> dict:
    """统计各Phase的后续N日收益率。

    Returns
    -------
    {phase: {count, avg_ret_5d, avg_ret_10d, win_rate_5d, ...}}
    """
    if forward_days is None:
        forward_days = [5, 10, 20]

    df = df.copy()
    close_arr = df["close"].values

    # 预计算 forward returns
    for n in forward_days:
        ret = np.full(len(df), np.nan)
        for i in range(len(df) - n):
            if close_arr[i] > 0:
                ret[i] = (close_arr[i + n] / close_arr[i] - 1) * 100
        df[f"fwd_ret_{n}d"] = ret

    results = {}
    for phase_val in [P_0U, P_0D, P_1, P_2, P_3A, P_3B, P_4A, P_4B]:
        mask = df["phase"] == phase_val
        subset = df[mask]
        if len(subset) == 0:
            continue
        stats = {"count": len(subset)}
        for n in forward_days:
            col = f"fwd_ret_{n}d"
            valid = subset[col].dropna()
            if len(valid) > 0:
                stats[f"avg_ret_{n}d"] = round(float(valid.mean()), 2)
                stats[f"med_ret_{n}d"] = round(float(valid.median()), 2)
                stats[f"win_rate_{n}d"] = round(float((valid > 0).mean()) * 100, 1)
            else:
                stats[f"avg_ret_{n}d"] = None
                stats[f"win_rate_{n}d"] = None
        # avg score
        stats["avg_score"] = round(float(subset["cr_score"].mean()), 1)
        results[phase_val] = stats

    return results


def batch_backtest(symbols: list[str]) -> dict:
    """批量回测多只股票，汇总各Phase统计。"""
    all_phase_data = {p: [] for p in [P_0U, P_0D, P_1, P_2, P_3A, P_3B, P_4A, P_4B]}

    for symbol in symbols:
        try:
            df = analyze_symbol(symbol)
            if df.empty:
                continue
            close_arr = df["close"].values
            for n in [5, 10, 20]:
                ret = np.full(len(df), np.nan)
                for i in range(len(df) - n):
                    if close_arr[i] > 0:
                        ret[i] = (close_arr[i + n] / close_arr[i] - 1) * 100
                df[f"fwd_ret_{n}d"] = ret

            for _, row in df.dropna(subset=["phase", "fwd_ret_5d"]).iterrows():
                phase = row["phase"]
                if phase in all_phase_data:
                    all_phase_data[phase].append({
                        "symbol": symbol,
                        "ret_5d": row["fwd_ret_5d"],
                        "ret_10d": row.get("fwd_ret_10d"),
                        "ret_20d": row.get("fwd_ret_20d"),
                        "score": row["cr_score"],
                        "rsi": row.get("rsi"),
                        "vol_ratio": row.get("vol_ratio"),
                    })
        except Exception as e:
            print(f"  [通道] {symbol} 失败: {e}")

    # 汇总
    summary = {}
    for phase, records in all_phase_data.items():
        if not records:
            continue
        pdf = pd.DataFrame(records)
        stats = {"count": len(pdf)}
        for n in [5, 10, 20]:
            col = f"ret_{n}d"
            valid = pdf[col].dropna()
            if len(valid) > 0:
                stats[f"avg_ret_{n}d"] = round(float(valid.mean()), 2)
                stats[f"med_ret_{n}d"] = round(float(valid.median()), 2)
                stats[f"win_rate_{n}d"] = round(float((valid > 0).mean()) * 100, 1)
        stats["avg_score"] = round(float(pdf["score"].mean()), 1)
        summary[phase] = stats

    return summary


def print_backtest(results: dict) -> None:
    """打印回测结果表格。"""
    print()
    print("=" * 100)
    print("通道反转指标回测")
    print("=" * 100)
    print(f"{'Phase':>6} | {'含义':>12} | {'样本':>5} | {'均分':>5} | "
          f"{'5日均涨':>8} | {'5日胜率':>7} | {'10日均涨':>8} | {'10日胜率':>8} | {'20日均涨':>8}")
    print("-" * 100)

    labels = {
        P_0U: "上半区(多)",
        P_0D: "下半区(空)",
        P_1: "破下轨",
        P_2: "回到轨道",
        P_3A: "弱反弹",
        P_3B: "过中轨",
        P_4A: "有效反转",
        P_4B: "上轨整理",
    }
    order = [P_4A, P_3B, P_4B, P_0U, P_2, P_1, P_0D, P_3A]
    for p in order:
        s = results.get(p)
        if not s:
            continue
        def _f(key):
            v = s.get(key)
            return f"{v:+7.2f}%" if v is not None else "    N/A"
        def _w(key):
            v = s.get(key)
            return f"{v:>6.1f}%" if v is not None else "    N/A"
        print(f"{p:>6} | {labels.get(p, '?'):>12} | {s['count']:>5} | {s['avg_score']:>5.1f} | "
              f"{_f('avg_ret_5d')} | {_w('win_rate_5d')} | {_f('avg_ret_10d')} | {_w('win_rate_10d')} | {_f('avg_ret_20d')}")
    print("=" * 100)


# ── 数据加载 ────────────────────────────────────────────────

def _load_daily(symbol: str) -> pd.DataFrame | None:
    """从TDX本地加载日线数据。"""
    try:
        from .data_backend import DataBackend
        backend = DataBackend(mode="combined", default_sources=["tdx", "akshare"])
        df = backend._fetch_kline_tdx(symbol, "day", limit=500)
        if df is not None and not df.empty:
            # TDX返回的列: ts, open, high, low, close, volume, amount, pct_chg
            df = df.rename(columns={"ts": "date"})
            df["date"] = pd.to_datetime(df["date"])
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.reset_index(drop=True)
    except Exception:
        pass
    return None
