"""v3 K 走势专家 - 多模态视觉版绘图 (6 子图分层设计)。

架构: 单张 matplotlib 大图(18×21 in, dpi=110), 2 列 × 3 主题:
  左列 = 日线,  右列 = 周线

3 个研判主题(每主题 = 主图 + 对应副图):
  Theme 1: K + MA5/10/20/60 + 布林带 + 成交量  /  MACD(DIF/DEA/柱)      [趋势+动能]
  Theme 2: K + Ichimoku 云 + 转换线/基准线     /  RSI(14) + 超买超卖区  [云图+强度]
  Theme 3: K + SAR + 自动趋势线                /  MACD 柱 + RSI 叠加    [反转+综合]

副图嵌入各主题 → 每张子图 = 1 个完整判断单元
日线服从周线 → Prompt 层优先级约束
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ─────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────

def _ma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=1).mean()


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _ma(close, n)
    std = close.rolling(n, min_periods=1).std()
    return mid + k * std, mid, mid - k * std


def _ichimoku(df: pd.DataFrame) -> dict[str, pd.Series]:
    high = df["high"]; low = df["low"]
    tenkan = (high.rolling(9, min_periods=1).max() + low.rolling(9, min_periods=1).min()) / 2
    kijun = (high.rolling(26, min_periods=1).max() + low.rolling(26, min_periods=1).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52, min_periods=1).max() + low.rolling(52, min_periods=1).min()) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a,
            "senkou_b": senkou_b, "chikou": chikou}


def _sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    if len(df) < 5:
        return pd.Series([np.nan] * len(df), index=df.index)
    high = df["high"].values
    low = df["low"].values
    sar = np.full(len(df), np.nan)
    trend = 1; ep = high[0]; sar[0] = low[0]; af = af_step
    for i in range(1, len(df)):
        prev = sar[i - 1]
        new = prev + af * (ep - prev)
        if trend == 1:
            new = min(new, low[i - 1], low[max(i - 2, 0)])
            if low[i] < new:
                trend = -1; new = ep; ep = low[i]; af = af_step
            elif high[i] > ep:
                ep = high[i]; af = min(af + af_step, af_max)
        else:
            new = max(new, high[i - 1], high[max(i - 2, 0)])
            if high[i] > new:
                trend = 1; new = ep; ep = high[i]; af = af_step
            elif low[i] < ep:
                ep = low[i]; af = min(af + af_step, af_max)
        sar[i] = new
    return pd.Series(sar, index=df.index)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ADX 趋势强度指标。"""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask = (high.diff() > -low.diff())
    plus_dm = plus_dm.where(mask, 0)
    minus_dm = minus_dm.where(~mask, 0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    pdi = 100 * plus_dm.rolling(n, min_periods=1).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus_dm.rolling(n, min_periods=1).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.rolling(n, min_periods=1).mean()
    return adx


def _zigzag(df: pd.DataFrame, pct_threshold: float = 0.03) -> list[dict]:
    """阈值法 ZigZag 摆动点识别。

    返回 list of {idx, price, type('high'|'low'), ts}, 按时间排序。
    pct_threshold: 反转阈值(默认 3%, 若波动率大可调大)。
    """
    if len(df) < 3:
        return []
    highs = df["high"].values
    lows = df["low"].values
    swings: list[dict] = []
    # 初始方向: 看前 10 根的方向
    initial_window = min(10, len(df))
    if df["close"].iloc[initial_window - 1] >= df["close"].iloc[0]:
        direction = 1  # 初始上升 → 找高点
        last_ext_idx = int(np.argmax(highs[:initial_window]))
        last_ext_price = highs[last_ext_idx]
    else:
        direction = -1
        last_ext_idx = int(np.argmin(lows[:initial_window]))
        last_ext_price = lows[last_ext_idx]

    for i in range(len(df)):
        if direction == 1:
            # 上升阶段, 更新高点
            if highs[i] > last_ext_price:
                last_ext_idx = i
                last_ext_price = highs[i]
            # 从高点回落 pct_threshold → 确认高点, 转下降
            elif (last_ext_price - lows[i]) / max(last_ext_price, 1e-6) >= pct_threshold:
                swings.append({
                    "idx": last_ext_idx, "price": float(last_ext_price),
                    "type": "high", "ts": df["ts"].iloc[last_ext_idx],
                })
                direction = -1
                last_ext_idx = i
                last_ext_price = lows[i]
        else:
            # 下降阶段, 更新低点
            if lows[i] < last_ext_price:
                last_ext_idx = i
                last_ext_price = lows[i]
            elif (highs[i] - last_ext_price) / max(last_ext_price, 1e-6) >= pct_threshold:
                swings.append({
                    "idx": last_ext_idx, "price": float(last_ext_price),
                    "type": "low", "ts": df["ts"].iloc[last_ext_idx],
                })
                direction = 1
                last_ext_idx = i
                last_ext_price = highs[i]

    # 末尾未确认的极值也加入(供预测"正在发展中"的一浪)
    swings.append({
        "idx": last_ext_idx, "price": float(last_ext_price),
        "type": "high" if direction == 1 else "low",
        "ts": df["ts"].iloc[last_ext_idx],
        "provisional": True,
    })
    return swings


def _fit_trendline_through_points(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """过多点拟合直线 y = a*x + b。"""
    if len(points) < 2:
        return None
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    if len(points) == 2:
        a = (y[1] - y[0]) / max(x[1] - x[0], 1)
        b = y[0] - a * x[0]
    else:
        a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


# ─────────────────────────────────────────────────────────────────
# 通用绘图工具
# ─────────────────────────────────────────────────────────────────

def _draw_candles(ax, df: pd.DataFrame, width: float = 0.6) -> None:
    """画 K 线蜡烛。红涨绿跌。"""
    x = np.arange(len(df))
    for i in range(len(df)):
        o = df["open"].iloc[i]; c = df["close"].iloc[i]
        h = df["high"].iloc[i]; l = df["low"].iloc[i]
        col = "#E74C3C" if c >= o else "#27AE60"
        ax.plot([x[i], x[i]], [l, h], color=col, linewidth=0.8, alpha=0.9)
        body_h = max(abs(c - o), 0.002 * c)
        rect = Rectangle((x[i] - width / 2, min(o, c)), width, body_h,
                         facecolor=col, edgecolor=col, alpha=0.85)
        ax.add_patch(rect)


def _setup_xticks(ax_top, ax_bot, df: pd.DataFrame, show_labels: bool = True) -> None:
    """在底部 ax 显示日期标签, 顶部 ax 清空。"""
    x = np.arange(len(df))
    step = max(1, len(df) // 7)
    labels = [df["ts"].iloc[i].strftime("%m-%d") for i in range(len(df))]
    ax_top.set_xticks([])
    if show_labels:
        ax_bot.set_xticks(x[::step])
        ax_bot.set_xticklabels(labels[::step], fontsize=7, rotation=0)
    else:
        ax_bot.set_xticks([])


# ─────────────────────────────────────────────────────────────────
# 主题绘图函数
# ─────────────────────────────────────────────────────────────────

def _plot_theme_trend(ax_main, ax_sub, df: pd.DataFrame, tf_label: str) -> None:
    """主题 1: K + MA + 布林 + 成交量 / MACD。"""
    x = np.arange(len(df))
    close = df["close"]

    # --- 主图 ---
    _draw_candles(ax_main, df, width=0.6)

    ma5, ma10, ma20, ma60 = _ma(close, 5), _ma(close, 10), _ma(close, 20), _ma(close, 60)
    ax_main.plot(x, ma5, color="#FFD700", linewidth=1.2, label="MA5", alpha=0.9)
    ax_main.plot(x, ma10, color="#FF8C00", linewidth=1.2, label="MA10", alpha=0.9)
    ax_main.plot(x, ma20, color="#FF1493", linewidth=1.4, label="MA20", alpha=0.9)
    ax_main.plot(x, ma60, color="#1E90FF", linewidth=1.4, label="MA60", alpha=0.9)

    bb_u, bb_m, bb_l = _bollinger(close, 20, 2)
    ax_main.plot(x, bb_u, color="#999", linewidth=0.7, linestyle="--", alpha=0.5)
    ax_main.plot(x, bb_l, color="#999", linewidth=0.7, linestyle="--", alpha=0.5)
    ax_main.fill_between(x, bb_u, bb_l, color="#CCCCCC", alpha=0.10)

    # 成交量柱 (主图底部 1/4 区域, twinx)
    ax_vol = ax_main.twinx()
    vol_colors = ["#E74C3C" if df["close"].iloc[i] >= df["open"].iloc[i] else "#27AE60"
                  for i in range(len(df))]
    ax_vol.bar(x, df["volume"], color=vol_colors, width=0.6, alpha=0.3)
    ax_vol.set_ylim(0, df["volume"].max() * 4)
    ax_vol.set_yticks([])

    latest = close.iloc[-1]
    ax_main.axhline(latest, color="#C41E3A", linewidth=0.7, linestyle=":", alpha=0.6)
    ax_main.annotate(f"{latest:.2f}", xy=(len(df) - 1, latest),
                     xytext=(5, 0), textcoords="offset points",
                     fontsize=8, color="#C41E3A", fontweight="bold", va="center")
    ax_main.set_title(f"{tf_label} · 主题1 趋势与动能 (K+MA+布林+量)",
                       fontsize=10, fontweight="bold", loc="left")
    ax_main.legend(loc="upper left", fontsize=7, ncol=4, frameon=True, framealpha=0.85)
    ax_main.grid(True, alpha=0.25)
    ax_main.set_ylabel("价格", fontsize=8)

    # --- 副图: MACD ---
    dif, dea, hist = _macd(close)
    ax_sub.plot(x, dif, color="#1E90FF", linewidth=1.0, label="DIF")
    ax_sub.plot(x, dea, color="#FFA500", linewidth=1.0, label="DEA")
    bar_colors = ["#E74C3C" if h >= 0 else "#27AE60" for h in hist]
    ax_sub.bar(x, hist, color=bar_colors, width=0.6, alpha=0.8)
    ax_sub.axhline(0, color="#888", linewidth=0.5)
    ax_sub.legend(loc="upper left", fontsize=7, ncol=2)
    ax_sub.set_ylabel("MACD", fontsize=8)
    ax_sub.grid(True, alpha=0.2)


def _plot_theme_ichimoku(ax_main, ax_sub, df: pd.DataFrame, tf_label: str) -> None:
    """主题 2: K + Ichimoku 云 / RSI。"""
    x = np.arange(len(df))
    close = df["close"]

    # --- 主图 ---
    _draw_candles(ax_main, df, width=0.5)

    ichi = _ichimoku(df)
    ax_main.plot(x, ichi["tenkan"], color="#FF4500", linewidth=1.1, alpha=0.8, label="转换线(9)")
    ax_main.plot(x, ichi["kijun"], color="#8B4513", linewidth=1.1, alpha=0.8, label="基准线(26)")

    # 云 (senkou a-b fill)
    sa, sb = ichi["senkou_a"], ichi["senkou_b"]
    ax_main.fill_between(x, sa, sb, where=(sa >= sb), color="#90EE90", alpha=0.3,
                          label="云(多)")
    ax_main.fill_between(x, sa, sb, where=(sa < sb), color="#FFB6C1", alpha=0.3,
                          label="云(空)")
    ax_main.plot(x, sa, color="#27AE60", linewidth=0.8, alpha=0.6)
    ax_main.plot(x, sb, color="#C41E3A", linewidth=0.8, alpha=0.6)

    latest = close.iloc[-1]
    ax_main.axhline(latest, color="#C41E3A", linewidth=0.6, linestyle=":", alpha=0.5)

    ax_main.set_title(f"{tf_label} · 主题2 Ichimoku 云图",
                       fontsize=10, fontweight="bold", loc="left")
    ax_main.legend(loc="upper left", fontsize=7, ncol=2, frameon=True, framealpha=0.85)
    ax_main.grid(True, alpha=0.25)
    ax_main.set_ylabel("价格", fontsize=8)

    # --- 副图: RSI ---
    rsi = _rsi(close, 14)
    ax_sub.plot(x, rsi, color="#8B008B", linewidth=1.3, label="RSI(14)")
    ax_sub.axhline(70, color="#C41E3A", linewidth=0.5, linestyle="--", alpha=0.6)
    ax_sub.axhline(30, color="#27AE60", linewidth=0.5, linestyle="--", alpha=0.6)
    ax_sub.axhline(50, color="#888", linewidth=0.4, linestyle=":", alpha=0.5)
    ax_sub.fill_between(x, 70, 100, color="#FFB6C1", alpha=0.15)
    ax_sub.fill_between(x, 0, 30, color="#90EE90", alpha=0.15)
    ax_sub.set_ylim(0, 100)
    ax_sub.legend(loc="upper left", fontsize=7)
    ax_sub.set_ylabel("RSI", fontsize=8)
    ax_sub.grid(True, alpha=0.2)


def _plot_theme_reversal(ax_main, ax_sub, df: pd.DataFrame, tf_label: str) -> None:
    """主题 3: K + SAR + 自动趋势线 / MACD 柱 + RSI 叠加。"""
    x = np.arange(len(df))
    close = df["close"]

    # --- 主图 ---
    _draw_candles(ax_main, df, width=0.5)

    sar = _sar(df)
    ax_main.scatter(x, sar, s=12, color="#444", marker=".", alpha=0.7, label="SAR")

    # ── trendln 专业趋势线识别 ──
    try:
        import trendln
        window = min(len(df), 60)
        result = trendln.calc_support_resistance(
            (df["high"], df["low"]),
            extmethod=trendln.METHOD_NUMDIFF,
            method=trendln.METHOD_NSQUREDLOGN,
            window=window,
            errpct=0.005,
        )
        # result[0] = support info, result[1] = resistance info
        # 每个是 (extrema_indices, [main_slope, main_intercept], [candidate_lines])

        # 主支撑线 (粗, 醒目绿色)
        if result[0] and len(result[0]) > 1 and result[0][1]:
            s_slope, s_intercept = float(result[0][1][0]), float(result[0][1][1])
            xs = np.array([0, len(df) - 1])
            ys = s_slope * xs + s_intercept
            ax_main.plot(xs, ys, color="#00AA3F", linewidth=2.5, linestyle="-",
                          alpha=0.95, label=f"支撑线(slope={s_slope:+.3f})")

        # 主阻力线 (粗, 醒目红色)
        if result[1] and len(result[1]) > 1 and result[1][1]:
            r_slope, r_intercept = float(result[1][1][0]), float(result[1][1][1])
            xs = np.array([0, len(df) - 1])
            ys = r_slope * xs + r_intercept
            ax_main.plot(xs, ys, color="#D00020", linewidth=2.5, linestyle="-",
                          alpha=0.95, label=f"阻力线(slope={r_slope:+.3f})")

        # 在极值点上画醒目标记
        minima_idx = result[0][0] if result[0] else []
        maxima_idx = result[1][0] if result[1] else []
        for idx in minima_idx[-5:] if minima_idx else []:
            if 0 <= idx < len(df):
                ax_main.scatter(idx, df["low"].iloc[idx], s=60, marker="^",
                                 color="#00AA3F", edgecolors="white", linewidths=1.2,
                                 zorder=6)
        for idx in maxima_idx[-5:] if maxima_idx else []:
            if 0 <= idx < len(df):
                ax_main.scatter(idx, df["high"].iloc[idx], s=60, marker="v",
                                 color="#D00020", edgecolors="white", linewidths=1.2,
                                 zorder=6)
    except Exception as e:
        logger.warning("[kline_charts] trendln 失败, 降级简单回归: %s", e)
        # fallback: 简单 polyfit
        tl_n = min(30, len(df))
        tl_x = x[-tl_n:]
        tl_y = close.iloc[-tl_n:].values
        if len(tl_x) >= 2:
            a, b = np.polyfit(tl_x, tl_y, 1)
            ax_main.plot(tl_x, a * tl_x + b, color="#0D3B66", linewidth=2.0,
                         linestyle="--", alpha=0.85, label=f"回归线")

    # 近期高低点水平线(次要参考)
    if len(df) >= 20:
        recent_hi = df["high"].iloc[-20:].max()
        recent_lo = df["low"].iloc[-20:].min()
        ax_main.axhline(recent_hi, color="#D00020", linewidth=0.5, linestyle=":", alpha=0.35)
        ax_main.axhline(recent_lo, color="#00AA3F", linewidth=0.5, linestyle=":", alpha=0.35)

    latest = close.iloc[-1]
    ax_main.axhline(latest, color="#C41E3A", linewidth=0.6, linestyle=":", alpha=0.5)

    ax_main.set_title(f"{tf_label} · 主题3 反转信号 (SAR+趋势线)",
                       fontsize=10, fontweight="bold", loc="left")
    ax_main.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.85)
    ax_main.grid(True, alpha=0.25)
    ax_main.set_ylabel("价格", fontsize=8)

    # --- 副图: MACD 柱 + RSI(双轴叠加) ---
    _, _, hist = _macd(close)
    bar_colors = ["#E74C3C" if h >= 0 else "#27AE60" for h in hist]
    ax_sub.bar(x, hist, color=bar_colors, width=0.6, alpha=0.7, label="MACD柱")
    ax_sub.axhline(0, color="#888", linewidth=0.4)
    ax_sub.set_ylabel("MACD 柱", fontsize=8, color="#444")
    ax_sub.tick_params(axis="y", labelcolor="#444")

    ax_rsi = ax_sub.twinx()
    rsi = _rsi(close, 14)
    ax_rsi.plot(x, rsi, color="#8B008B", linewidth=1.1, alpha=0.9, label="RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.axhline(70, color="#C41E3A", linewidth=0.4, linestyle="--", alpha=0.5)
    ax_rsi.axhline(30, color="#27AE60", linewidth=0.4, linestyle="--", alpha=0.5)
    ax_rsi.set_ylabel("RSI", fontsize=8, color="#8B008B")
    ax_rsi.tick_params(axis="y", labelcolor="#8B008B")

    lines_1, labels_1 = ax_sub.get_legend_handles_labels()
    lines_2, labels_2 = ax_rsi.get_legend_handles_labels()
    ax_sub.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=7, ncol=2)
    ax_sub.grid(True, alpha=0.2)


# ─────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────

def _load_csv(csv: Path, n_recent: int) -> pd.DataFrame | None:
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if len(df) > n_recent:
        df = df.tail(n_recent).reset_index(drop=True)
    return df


def generate_merged_expert_chart(
    run_dir: Path,
    symbol: str,
    name: str,
    daily_n: int = 100,
    weekly_n: int = 80,
) -> Path | None:
    """生成 2 列 × 3 主题六子图合成大图, 返回路径。"""
    run_dir = Path(run_dir)
    kline_dir = run_dir / "data" / "kline"
    df_day = _load_csv(kline_dir / "day.csv", daily_n)
    df_week = _load_csv(kline_dir / "week.csv", weekly_n)
    if df_day is None or df_week is None:
        logger.warning("[kline_charts] 缺数据 day=%s week=%s",
                       df_day is not None, df_week is not None)
        return None

    fig = plt.figure(figsize=(18, 21), dpi=110)
    # 6 行(主/副 × 3 主题), 2 列(日/周)
    gs = fig.add_gridspec(
        nrows=6, ncols=2,
        height_ratios=[3.0, 1.1, 3.0, 1.1, 3.0, 1.1],
        hspace=0.28, wspace=0.10,
        top=0.96, bottom=0.04, left=0.05, right=0.96,
    )

    # 日线左列 col=0
    axes_day = []
    for row_pair in range(3):
        r_main = row_pair * 2
        r_sub = r_main + 1
        ax_m = fig.add_subplot(gs[r_main, 0])
        ax_s = fig.add_subplot(gs[r_sub, 0], sharex=ax_m)
        axes_day.append((ax_m, ax_s))

    # 周线右列 col=1
    axes_wk = []
    for row_pair in range(3):
        r_main = row_pair * 2
        r_sub = r_main + 1
        ax_m = fig.add_subplot(gs[r_main, 1])
        ax_s = fig.add_subplot(gs[r_sub, 1], sharex=ax_m)
        axes_wk.append((ax_m, ax_s))

    # 绘日线 3 主题
    _plot_theme_trend(*axes_day[0], df_day, "日线")
    _plot_theme_ichimoku(*axes_day[1], df_day, "日线")
    _plot_theme_reversal(*axes_day[2], df_day, "日线")

    # 绘周线 3 主题
    _plot_theme_trend(*axes_wk[0], df_week, "周线")
    _plot_theme_ichimoku(*axes_wk[1], df_week, "周线")
    _plot_theme_reversal(*axes_wk[2], df_week, "周线")

    # 设置 xticks (只在每主题的副图下显示日期)
    for (axm, axs), df in [(axes_day[0], df_day), (axes_day[1], df_day), (axes_day[2], df_day)]:
        _setup_xticks(axm, axs, df, show_labels=True)
    for (axm, axs), df in [(axes_wk[0], df_week), (axes_wk[1], df_week), (axes_wk[2], df_week)]:
        _setup_xticks(axm, axs, df, show_labels=True)

    # 总标题
    fig.suptitle(
        f"{symbol}  {name}  ·  K 走势多模态分析图",
        fontsize=14, fontweight="bold", y=0.985,
    )

    out = run_dir / "charts" / "expert" / "merged.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


def image_to_base64(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii"), "image/png"


# ─────────────────────────────────────────────────────────────────
# 波浪专家图 - 简洁布局 (不叠加 MA/Boll/Ichimoku/SAR)
# ─────────────────────────────────────────────────────────────────

def _plot_wave_theme(ax_main, ax_sub, df: pd.DataFrame, tf_label: str,
                     zigzag_threshold: float = 0.04) -> None:
    """波浪专家专用: K + ZigZag 摆动点(编号) + 斐波 + 前高低 / RSI+ADX 副图。"""
    x = np.arange(len(df))
    close = df["close"]

    # --- 主图: 简洁 K 线(宽度稍大,更易看形态) ---
    _draw_candles(ax_main, df, width=0.75)

    # --- ZigZag 摆动点 ---
    swings = _zigzag(df, pct_threshold=zigzag_threshold)
    if swings:
        # 画 ZigZag 连线(灰色虚线串起所有摆动点)
        zz_x = [s["idx"] for s in swings]
        zz_y = [s["price"] for s in swings]
        ax_main.plot(zz_x, zz_y, color="#555", linewidth=1.2, linestyle="--",
                      alpha=0.6, zorder=3)
        # 编号摆动点(仅显示最近 6 个非 provisional)
        confirmed = [s for s in swings if not s.get("provisional")]
        recent = confirmed[-6:] if len(confirmed) >= 6 else confirmed
        for i, s in enumerate(recent, 1):
            col = "#D00020" if s["type"] == "high" else "#00AA3F"
            marker = "v" if s["type"] == "high" else "^"
            ax_main.scatter(s["idx"], s["price"], s=100, marker=marker,
                             color=col, edgecolors="white", linewidths=1.5, zorder=7)
            # 编号
            dy = s["price"] * 0.015 * (1 if s["type"] == "high" else -1)
            ax_main.annotate(f"{i}", xy=(s["idx"], s["price"]),
                              xytext=(0, 10 if s["type"] == "high" else -15),
                              textcoords="offset points",
                              fontsize=10, fontweight="bold",
                              color=col, ha="center",
                              bbox=dict(boxstyle="circle,pad=0.2",
                                       facecolor="white", edgecolor=col, alpha=0.9))
        # 标注 provisional 末点(正在发展中的极值)
        prov = [s for s in swings if s.get("provisional")]
        if prov:
            s = prov[-1]
            col = "#FF6600"
            ax_main.scatter(s["idx"], s["price"], s=120, marker="*",
                             color=col, edgecolors="white", linewidths=1.5, zorder=7)
            ax_main.annotate("?", xy=(s["idx"], s["price"]),
                              xytext=(0, 12 if s["type"] == "high" else -16),
                              textcoords="offset points",
                              fontsize=10, fontweight="bold", color=col, ha="center")

    # --- 斐波那契回撤位 (基于最近一次 ZigZag 高低点) ---
    confirmed = [s for s in swings if not s.get("provisional")]
    if len(confirmed) >= 2:
        # 找最近的 swing high 和 swing low
        recent_high = None
        recent_low = None
        for s in reversed(confirmed):
            if s["type"] == "high" and recent_high is None:
                recent_high = s
            elif s["type"] == "low" and recent_low is None:
                recent_low = s
            if recent_high and recent_low:
                break
        if recent_high and recent_low:
            hi = recent_high["price"]
            lo = recent_low["price"]
            # 斐波那契回撤位
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            span = hi - lo
            for level in fib_levels:
                # 从高点回撤 level
                y_retrace = hi - span * level
                ax_main.axhline(y_retrace, color="#0D3B66", linewidth=0.8,
                                 linestyle=":", alpha=0.5)
                ax_main.text(len(df) * 0.01, y_retrace,
                              f"Fib {level*100:.1f}% ({y_retrace:.2f})",
                              fontsize=7, color="#0D3B66", va="center", alpha=0.8)
            # 高低点水平线加粗
            ax_main.axhline(hi, color="#D00020", linewidth=1.0, linestyle="--", alpha=0.6)
            ax_main.axhline(lo, color="#00AA3F", linewidth=1.0, linestyle="--", alpha=0.6)

    # 最新价
    latest = close.iloc[-1]
    ax_main.axhline(latest, color="#C41E3A", linewidth=0.7, linestyle=":", alpha=0.55)
    ax_main.annotate(f"{latest:.2f}", xy=(len(df) - 1, latest),
                     xytext=(5, 0), textcoords="offset points",
                     fontsize=9, color="#C41E3A", fontweight="bold", va="center")

    ax_main.set_title(f"{tf_label} · 波浪识别主图 (K+ZigZag+斐波)",
                       fontsize=10, fontweight="bold", loc="left")
    ax_main.legend(loc="upper left", fontsize=7, framealpha=0.85)
    ax_main.grid(True, alpha=0.2)
    ax_main.set_ylabel("价格", fontsize=8)

    # --- 副图: RSI + ADX 双线 ---
    rsi = _rsi(close, 14)
    ax_sub.plot(x, rsi, color="#8B008B", linewidth=1.3, label="RSI(14)")
    ax_sub.axhline(70, color="#C41E3A", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_sub.axhline(30, color="#27AE60", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_sub.set_ylim(0, 100)
    ax_sub.set_ylabel("RSI", fontsize=8, color="#8B008B")
    ax_sub.tick_params(axis="y", labelcolor="#8B008B")

    # ADX 叠加在右轴
    try:
        adx = _adx(df, 14)
        ax_adx = ax_sub.twinx()
        ax_adx.plot(x, adx, color="#FF8C00", linewidth=1.3, label="ADX(14)")
        ax_adx.axhline(25, color="#FF8C00", linewidth=0.4, linestyle="--", alpha=0.4)
        ax_adx.set_ylim(0, 80)
        ax_adx.set_ylabel("ADX", fontsize=8, color="#FF8C00")
        ax_adx.tick_params(axis="y", labelcolor="#FF8C00")

        lines_1, labels_1 = ax_sub.get_legend_handles_labels()
        lines_2, labels_2 = ax_adx.get_legend_handles_labels()
        ax_sub.legend(lines_1 + lines_2, labels_1 + labels_2,
                      loc="upper left", fontsize=7, ncol=2)
    except Exception:
        ax_sub.legend(loc="upper left", fontsize=7)
    ax_sub.grid(True, alpha=0.2)


def generate_wave_expert_chart(
    run_dir: Path,
    symbol: str,
    name: str,
    weekly_n: int = 100,
    monthly_n: int = 60,
) -> Path | None:
    """波浪专家专用图: 周+月双周期, 简洁 K+ZigZag+斐波, 不叠加 MA/Boll。"""
    run_dir = Path(run_dir)
    kline_dir = run_dir / "data" / "kline"
    df_week = _load_csv(kline_dir / "week.csv", weekly_n)
    df_month = _load_csv(kline_dir / "month.csv", monthly_n)
    if df_week is None or df_month is None:
        logger.warning("[wave_charts] 缺数据 week=%s month=%s",
                       df_week is not None, df_month is not None)
        return None

    fig = plt.figure(figsize=(18, 14), dpi=110)
    # 2 行主+副, 2 列(周/月)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[3.2, 1.2],
        hspace=0.18, wspace=0.12,
        top=0.93, bottom=0.06, left=0.05, right=0.96,
    )

    ax_wk_main = fig.add_subplot(gs[0, 0])
    ax_wk_sub = fig.add_subplot(gs[1, 0], sharex=ax_wk_main)
    ax_mt_main = fig.add_subplot(gs[0, 1])
    ax_mt_sub = fig.add_subplot(gs[1, 1], sharex=ax_mt_main)

    # 周线阈值 4%, 月线阈值 6%(大周期波动大)
    _plot_wave_theme(ax_wk_main, ax_wk_sub, df_week, "周线", zigzag_threshold=0.04)
    _plot_wave_theme(ax_mt_main, ax_mt_sub, df_month, "月线", zigzag_threshold=0.06)

    _setup_xticks(ax_wk_main, ax_wk_sub, df_week, show_labels=True)
    _setup_xticks(ax_mt_main, ax_mt_sub, df_month, show_labels=True)

    fig.suptitle(
        f"{symbol}  {name}  ·  艾略特波浪识别图 (周线左, 月线右)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    out = run_dir / "charts" / "expert" / "wave.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────
# 向后兼容接口 (被 structure_expert.py 调用)
# ─────────────────────────────────────────────────────────────────

def generate_expert_charts(
    run_dir: Path,
    symbol: str,
    name: str,
    timeframes: tuple[str, ...] = ("day", "week"),
    n_recent: dict[str, int] | None = None,
) -> dict[str, Path]:
    """向后兼容: 返回 {'merged': path}(合成大图取代原先日/周分开)。"""
    merged = generate_merged_expert_chart(run_dir, symbol, name)
    return {"merged": merged} if merged else {}


def merge_charts_vertically(day_path: Path, week_path: Path, out_path: Path) -> Path:
    """向后兼容: 现在 generate_merged_expert_chart 已直接产出合成图。
    此函数仅为保持旧调用不报错, 直接 copy 一份。"""
    import shutil
    shutil.copy(day_path, out_path)
    return out_path
