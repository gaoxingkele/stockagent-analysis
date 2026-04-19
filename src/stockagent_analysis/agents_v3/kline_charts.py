"""v3 专用 K 线图绘制 - 多模态 LLM 输入。

主图: K 线 + MA(5/10/20/60) + 布林带 + Ichimoku 云 + SAR + 成交量
副图 1: MACD (DIF/DEA + 柱状 + 零轴)
副图 2: RSI + 自动趋势线 + 超买超卖区

支持日/周双周期, 保存 PNG 供 LLMRouter.chat_with_image 使用。
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 非交互后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter, MonthLocator, WeekdayLocator

logger = logging.getLogger(__name__)


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
    high = df["high"]
    low = df["low"]
    tenkan = (high.rolling(9, min_periods=1).max() + low.rolling(9, min_periods=1).min()) / 2
    kijun = (high.rolling(26, min_periods=1).max() + low.rolling(26, min_periods=1).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((high.rolling(52, min_periods=1).max() + low.rolling(52, min_periods=1).min()) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a,
            "senkou_b": senkou_b, "chikou": chikou}


def _sar(df: pd.DataFrame, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    """Parabolic SAR。返回与 df 等长的 Series。"""
    if len(df) < 5:
        return pd.Series([np.nan] * len(df), index=df.index)
    high = df["high"].values
    low = df["low"].values
    sar = np.full(len(df), np.nan)
    trend = 1   # 1=上升, -1=下降
    ep = high[0]
    sar[0] = low[0]
    af = af_step
    for i in range(1, len(df)):
        prev_sar = sar[i - 1]
        new_sar = prev_sar + af * (ep - prev_sar)
        if trend == 1:
            new_sar = min(new_sar, low[i - 1], low[max(i - 2, 0)])
            if low[i] < new_sar:
                trend = -1
                new_sar = ep
                ep = low[i]
                af = af_step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            new_sar = max(new_sar, high[i - 1], high[max(i - 2, 0)])
            if high[i] > new_sar:
                trend = 1
                new_sar = ep
                ep = high[i]
                af = af_step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        sar[i] = new_sar
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


def _trendline(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """最小二乘拟合 y = a*x + b, 返回 (a, b)。"""
    if len(x) < 2:
        return 0.0, float(y[-1]) if len(y) else 0.0
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


# ─────────────────────────────────────────────────────────────────
# 主绘图函数
# ─────────────────────────────────────────────────────────────────

def _plot_one(csv_path: Path, symbol: str, name: str, timeframe: str,
              out_path: Path, n_recent: int = 100) -> Path:
    """绘制单一周期图(日或周), 保存到 out_path。"""
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    if len(df) > n_recent:
        df = df.tail(n_recent).reset_index(drop=True)

    x = np.arange(len(df))
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    # 计算指标
    ma5, ma10, ma20, ma60 = _ma(close, 5), _ma(close, 10), _ma(close, 20), _ma(close, 60)
    bb_upper, bb_mid, bb_lower = _bollinger(close, 20, 2)
    ichi = _ichimoku(df)
    sar = _sar(df)
    dif, dea, hist = _macd(close)
    rsi = _rsi(close)

    # 布局 3 subplots: 主图(60%), MACD(20%), RSI(20%)
    fig = plt.figure(figsize=(16, 10), dpi=110)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.0, 1.0], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_macd = fig.add_subplot(gs[1], sharex=ax_main)
    ax_rsi = fig.add_subplot(gs[2], sharex=ax_main)

    # ── 主图: K 线蜡烛 ──
    width = 0.6
    for i in range(len(df)):
        o, c, h, l = open_[i], close[i], high[i], low[i]
        col = "#E74C3C" if c >= o else "#27AE60"   # 红涨绿跌(A 股习惯)
        # 影线
        ax_main.plot([x[i], x[i]], [l, h], color=col, linewidth=0.8, alpha=0.9)
        # 实体
        body_h = max(abs(c - o), 0.002 * c)
        rect = Rectangle((x[i] - width / 2, min(o, c)), width, body_h,
                          facecolor=col, edgecolor=col, alpha=0.85)
        ax_main.add_patch(rect)

    # ── 主图: 均线 ──
    ax_main.plot(x, ma5, color="#FFD700", linewidth=1.3, label="MA5", alpha=0.9)
    ax_main.plot(x, ma10, color="#FF8C00", linewidth=1.3, label="MA10", alpha=0.9)
    ax_main.plot(x, ma20, color="#FF1493", linewidth=1.5, label="MA20", alpha=0.9)
    if ma60.notna().any() and (close > 0).any():
        ax_main.plot(x, ma60, color="#1E90FF", linewidth=1.5, label="MA60", alpha=0.9)

    # ── 主图: 布林带 ──
    ax_main.plot(x, bb_upper, color="#666", linewidth=0.8, linestyle="--", alpha=0.5, label="Boll上")
    ax_main.plot(x, bb_lower, color="#666", linewidth=0.8, linestyle="--", alpha=0.5, label="Boll下")
    ax_main.fill_between(x, bb_upper, bb_lower, color="#CCCCCC", alpha=0.12)

    # ── 主图: Ichimoku 云 ──
    ax_main.plot(x, ichi["tenkan"], color="#FF4500", linewidth=1, alpha=0.7, label="Tenkan")
    ax_main.plot(x, ichi["kijun"], color="#8B4513", linewidth=1, alpha=0.7, label="Kijun")
    # 云(Senkou span A/B 之间)
    senkou_a = ichi["senkou_a"]
    senkou_b = ichi["senkou_b"]
    ax_main.fill_between(x, senkou_a, senkou_b,
                         where=(senkou_a >= senkou_b),
                         color="#90EE90", alpha=0.2, label="云(多)")
    ax_main.fill_between(x, senkou_a, senkou_b,
                         where=(senkou_a < senkou_b),
                         color="#FFB6C1", alpha=0.2, label="云(空)")

    # ── 主图: SAR ──
    ax_main.scatter(x, sar, s=10, color="#444444", marker=".", alpha=0.6, label="SAR")

    # 最新价水平线 + 标注
    latest_close = close.iloc[-1]
    ax_main.axhline(latest_close, color="#C41E3A", linewidth=0.7, linestyle=":", alpha=0.6)
    ax_main.annotate(f"最新 {latest_close:.2f}",
                     xy=(len(df) - 1, latest_close),
                     xytext=(5, 0), textcoords="offset points",
                     fontsize=9, color="#C41E3A", fontweight="bold",
                     va="center")

    tf_cn = {"day": "日线", "week": "周线", "month": "月线", "1h": "1小时"}.get(timeframe, timeframe)
    ax_main.set_title(f"{symbol} {name} · {tf_cn} K 图(近 {len(df)} 根)", fontsize=13, fontweight="bold")
    ax_main.legend(loc="upper left", fontsize=8, ncol=5, frameon=True,
                   fancybox=True, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylabel("价格", fontsize=10)

    # 成交量(嵌入主图底部小区域, 用 twinx)
    ax_vol = ax_main.twinx()
    vol_colors = ["#E74C3C" if close.iloc[i] >= open_.iloc[i] else "#27AE60" for i in range(len(df))]
    ax_vol.bar(x, volume, color=vol_colors, width=width, alpha=0.3)
    ax_vol.set_ylim(0, volume.max() * 4)   # 压缩到底部 1/4 高度
    ax_vol.set_yticks([])
    ax_vol.set_ylabel("")

    # ── 副图 1: MACD ──
    ax_macd.plot(x, dif, color="#1E90FF", linewidth=1.2, label="DIF")
    ax_macd.plot(x, dea, color="#FFA500", linewidth=1.2, label="DEA")
    bar_colors = ["#E74C3C" if h >= 0 else "#27AE60" for h in hist]
    ax_macd.bar(x, hist, color=bar_colors, width=width * 0.9, alpha=0.8)
    ax_macd.axhline(0, color="#999", linewidth=0.6, linestyle="-")
    ax_macd.legend(loc="upper left", fontsize=8, ncol=2)
    ax_macd.set_ylabel("MACD", fontsize=10)
    ax_macd.grid(True, alpha=0.25)

    # ── 副图 2: RSI + 趋势线 ──
    ax_rsi.plot(x, rsi, color="#8B008B", linewidth=1.3, label="RSI(14)")
    ax_rsi.axhline(70, color="#C41E3A", linewidth=0.5, linestyle="--", alpha=0.6)
    ax_rsi.axhline(30, color="#27AE60", linewidth=0.5, linestyle="--", alpha=0.6)
    ax_rsi.axhline(50, color="#888", linewidth=0.4, linestyle=":", alpha=0.5)
    ax_rsi.fill_between(x, 70, 100, color="#FFB6C1", alpha=0.15)
    ax_rsi.fill_between(x, 0, 30, color="#90EE90", alpha=0.15)

    # 自动趋势线(对 close 线性回归, 近 30 根)
    tl_window = min(30, len(df))
    tl_x = x[-tl_window:]
    tl_y = close.iloc[-tl_window:].values
    a, b = _trendline(tl_x, tl_y)
    trend_vals = a * tl_x + b
    # 映射趋势线到 RSI 轴(仅作为方向参考)
    trend_dir = "上升" if a > 0 else "下降"
    ax_rsi.text(0.98, 0.95, f"Close趋势线斜率: {a:.3f} ({trend_dir})",
                transform=ax_rsi.transAxes, ha="right", va="top",
                fontsize=8, color="#333", bbox=dict(boxstyle="round", facecolor="#FFFFE0", alpha=0.6))
    # 在主图画 close 的趋势线
    ax_main.plot(tl_x, trend_vals,
                 color="#0D3B66", linewidth=1.2, linestyle="-.",
                 alpha=0.7, label=f"趋势线({trend_dir})")
    ax_main.legend(loc="upper left", fontsize=8, ncol=5, frameon=True, fancybox=True, framealpha=0.9)

    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI", fontsize=10)
    ax_rsi.legend(loc="upper left", fontsize=8)
    ax_rsi.grid(True, alpha=0.25)
    ax_rsi.set_xlabel("K 线序号", fontsize=9)

    # X 轴日期标签
    idx_labels = [df["ts"].iloc[i].strftime("%m-%d") for i in range(len(df))]
    step = max(1, len(df) // 8)
    ax_rsi.set_xticks(x[::step])
    ax_rsi.set_xticklabels(idx_labels[::step], fontsize=8, rotation=0)
    ax_main.set_xticks([])
    ax_macd.set_xticks([])

    # 中文字体(Windows)
    try:
        import matplotlib
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def generate_expert_charts(
    run_dir: Path,
    symbol: str,
    name: str,
    timeframes: tuple[str, ...] = ("day", "week"),
    n_recent: dict[str, int] | None = None,
) -> dict[str, Path]:
    """为 K 走势专家生成 K 线图。

    Args:
        run_dir: v3 run 目录(含 data/kline/<tf>.csv)
        symbol / name: 标的
        timeframes: 要生成的周期列表
        n_recent: 每个周期显示最近多少根(默认 day=100, week=80)

    Returns:
        {"day": Path, "week": Path}
    """
    run_dir = Path(run_dir)
    n_recent = n_recent or {"day": 100, "week": 80, "month": 60}
    out_dir = run_dir / "charts" / "expert"
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for tf in timeframes:
        csv = run_dir / "data" / "kline" / f"{tf}.csv"
        if not csv.exists():
            logger.warning("[charts] 缺 %s", csv)
            continue
        try:
            out = out_dir / f"{tf}_kline.png"
            _plot_one(csv, symbol, name, tf, out, n_recent=n_recent.get(tf, 100))
            result[tf] = out
        except Exception as e:
            logger.warning("[charts] %s 绘图失败: %s", tf, e)
    return result


def image_to_base64(path: Path) -> tuple[str, str]:
    """返回 (base64_str, mime_type)。"""
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii"), "image/png"


def merge_charts_vertically(day_path: Path, week_path: Path, out_path: Path) -> Path:
    """把日线+周线图上下拼合成一张, 供 vision LLM 一次性读取。"""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("需要 Pillow: pip install Pillow")

    img_day = Image.open(day_path).convert("RGB")
    img_week = Image.open(week_path).convert("RGB")

    # 统一宽度
    target_w = max(img_day.width, img_week.width)
    def _resize(img):
        if img.width == target_w:
            return img
        new_h = int(img.height * target_w / img.width)
        return img.resize((target_w, new_h), Image.LANCZOS)
    img_day = _resize(img_day)
    img_week = _resize(img_week)

    combined = Image.new("RGB", (target_w, img_day.height + img_week.height + 10), "white")
    combined.paste(img_day, (0, 0))
    combined.paste(img_week, (0, img_day.height + 10))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path, quality=92)
    return out_path
