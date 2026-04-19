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

    # 自动趋势线: 近 30 根 close 线性回归
    tl_n = min(30, len(df))
    tl_x = x[-tl_n:]
    tl_y = close.iloc[-tl_n:].values
    if len(tl_x) >= 2:
        a, b = np.polyfit(tl_x, tl_y, 1)
        trend_dir = "上升" if a > 0 else "下降"
        ax_main.plot(tl_x, a * tl_x + b, color="#0D3B66", linewidth=1.2,
                     linestyle="-.", alpha=0.8, label=f"趋势线({trend_dir} a={a:.3f})")

    # 近期高低点(水平支撑阻力)
    if len(df) >= 20:
        recent_hi = df["high"].iloc[-20:].max()
        recent_lo = df["low"].iloc[-20:].min()
        ax_main.axhline(recent_hi, color="#C41E3A", linewidth=0.6, linestyle="--", alpha=0.5)
        ax_main.axhline(recent_lo, color="#27AE60", linewidth=0.6, linestyle="--", alpha=0.5)
        ax_main.text(0, recent_hi, f" R:{recent_hi:.2f}", fontsize=7, color="#C41E3A", va="bottom")
        ax_main.text(0, recent_lo, f" S:{recent_lo:.2f}", fontsize=7, color="#27AE60", va="top")

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
