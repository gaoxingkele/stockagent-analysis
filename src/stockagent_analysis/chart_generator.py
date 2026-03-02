# -*- coding: utf-8 -*-
"""K线图表生成模块。
生成多周期（1h/日/周/月）K线综合图，叠加均线、布林带、成交量、MACD/RSI/KDJ 副图及自动趋势线。
A股惯例：红涨绿跌。
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _setup_chinese_font() -> None:
    """配置 matplotlib 使用系统中文字体（Windows / Linux 均支持）。"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # 按优先级尝试常见中文字体
        font_candidates = [
            "Microsoft YaHei", "微软雅黑",
            "SimHei", "黑体",
            "Noto Sans CJK SC", "Noto Serif SC",
            "WenQuanYi Micro Hei",
        ]
        # 也扫描系统字体文件
        sys_fonts = {fm.FontProperties(fname=f).get_name(): f
                     for f in fm.findSystemFonts()
                     if any(kw in f.lower() for kw in ("msyh", "simhei", "simsun", "noto", "wqy", "yahei"))}
        font_candidates += list(sys_fonts.keys())

        available = {f.name for f in fm.fontManager.ttflist}
        for name in font_candidates:
            if name in available or name in sys_fonts:
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
                plt.rcParams["axes.unicode_minus"] = False
                return
    except Exception:
        pass


_setup_chinese_font()

# 每个周期目标K线根数（参考缠论与波浪理论）
TIMEFRAME_BARS: dict[str, int] = {
    "1h": 160,
    "day": 250,
    "week": 156,
    "month": 120,
}

# 周期中文标签
TIMEFRAME_LABEL: dict[str, str] = {
    "1h": "1小时",
    "day": "日线",
    "week": "周线",
    "month": "月线",
}


def generate_kline_chart(
    df,
    timeframe: str,
    symbol: str,
    name: str,
    save_path: str | Path | None = None,
) -> bytes | None:
    """
    生成单个周期的K线综合图。
    df: 标准化后的 DataFrame，列名: ts, open, high, low, close, volume
    返回 PNG bytes（同时按 save_path 落盘）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
        import pandas as pd
    except ImportError as e:
        logger.warning("chart_generator: import failed %s", e)
        return None

    if df is None or df.empty:
        logger.warning("chart_generator: empty df for %s %s", symbol, timeframe)
        return None

    try:
        df = df.copy()
        # 统一时间列
        if "ts" in df.columns:
            df["dt"] = pd.to_datetime(df["ts"], errors="coerce")
        elif "date" in df.columns:
            df["dt"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["dt"] = pd.RangeIndex(len(df))
        df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        df = df.dropna(subset=["open", "high", "low", "close"])
        if len(df) < 5:
            return None

        # 限制根数
        max_bars = TIMEFRAME_BARS.get(timeframe, 250)
        df = df.tail(max_bars).reset_index(drop=True)
        n = len(df)
        x = np.arange(n)

        indicators = _calc_indicators(df)

        # 布局：主图(4) + 成交量(1) + MACD(2) + RSI(1.5) + KDJ(1.5)
        fig = plt.figure(figsize=(16, 11), dpi=130)
        fig.patch.set_facecolor("#0d1117")
        gs = gridspec.GridSpec(
            5, 1,
            height_ratios=[4, 1, 2, 1.5, 1.5],
            hspace=0.04,
            top=0.94, bottom=0.05, left=0.06, right=0.97,
        )
        ax_main = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax_main)
        ax_macd = fig.add_subplot(gs[2], sharex=ax_main)
        ax_rsi = fig.add_subplot(gs[3], sharex=ax_main)
        ax_kdj = fig.add_subplot(gs[4], sharex=ax_main)

        _style_ax(ax_main)
        _style_ax(ax_vol)
        _style_ax(ax_macd)
        _style_ax(ax_rsi)
        _style_ax(ax_kdj)

        # ── 主图：蜡烛 ──────────────────────────────────────────────────
        _draw_candles(ax_main, df, x)

        # 均线
        ma_cfg = [
            (5,   "#FF6B35"),   # MA5  橙
            (10,  "#FFD700"),   # MA10 金
            (20,  "#00BFFF"),   # MA20 蓝
            (60,  "#FF69B4"),   # MA60 粉
            (120, "#98FB98"),   # MA120 浅绿
            (250, "#DDA0DD"),   # MA250 薰衣草
        ]
        for period, color in ma_cfg:
            col = f"ma{period}"
            if col in indicators and indicators[col] is not None:
                vals = indicators[col]
                valid = ~np.isnan(vals)
                if valid.sum() > 1:
                    ax_main.plot(x[valid], vals[valid], color=color, linewidth=0.9,
                                 label=f"MA{period}", alpha=0.85)

        # 布林带
        if "bb_upper" in indicators and indicators["bb_upper"] is not None:
            bb_u = indicators["bb_upper"]
            bb_m = indicators["bb_mid"]
            bb_l = indicators["bb_lower"]
            valid = ~np.isnan(bb_u)
            if valid.sum() > 1:
                ax_main.plot(x[valid], bb_u[valid], color="#7B68EE", linewidth=0.7, linestyle="--", alpha=0.7)
                ax_main.plot(x[valid], bb_m[valid], color="#7B68EE", linewidth=0.7, linestyle="-", alpha=0.5)
                ax_main.plot(x[valid], bb_l[valid], color="#7B68EE", linewidth=0.7, linestyle="--", alpha=0.7)
                ax_main.fill_between(x[valid], bb_l[valid], bb_u[valid], alpha=0.05, color="#7B68EE")

        # 趋势线（自动）
        _draw_trend_lines(ax_main, df, x)

        # 图例
        handles, labels = ax_main.get_legend_handles_labels()
        if handles:
            ax_main.legend(handles, labels, loc="upper left", fontsize=6,
                           facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
                           ncol=min(6, len(handles)), framealpha=0.7)

        label_cn = TIMEFRAME_LABEL.get(timeframe, timeframe)
        ax_main.set_title(f"{symbol} {name}  {label_cn}K线图", color="white", fontsize=11, pad=6)
        ax_main.set_ylabel("价格", color="#aaaaaa", fontsize=8)

        # ── 成交量 ─────────────────────────────────────────────────────
        _draw_volume(ax_vol, df, x)
        ax_vol.set_ylabel("量", color="#aaaaaa", fontsize=7)

        # ── MACD ──────────────────────────────────────────────────────
        _draw_macd(ax_macd, indicators, x)
        ax_macd.set_ylabel("MACD", color="#aaaaaa", fontsize=7)

        # ── RSI ───────────────────────────────────────────────────────
        _draw_rsi(ax_rsi, indicators, x)
        ax_rsi.set_ylabel("RSI", color="#aaaaaa", fontsize=7)

        # ── KDJ ───────────────────────────────────────────────────────
        _draw_kdj(ax_kdj, indicators, x)
        ax_kdj.set_ylabel("KDJ", color="#aaaaaa", fontsize=7)

        # X轴日期刻度（最后一个子图）
        _set_xticks(ax_kdj, df, x)
        for ax in (ax_main, ax_vol, ax_macd, ax_rsi):
            plt.setp(ax.get_xticklabels(), visible=False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=130)
        plt.close(fig)
        buf.seek(0)
        png_bytes = buf.read()

        if save_path:
            Path(save_path).write_bytes(png_bytes)
            logger.info("chart saved: %s (%d bytes)", save_path, len(png_bytes))

        return png_bytes

    except Exception as e:
        logger.warning("chart_generator: failed %s %s: %s", symbol, timeframe, e, exc_info=True)
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def generate_all_charts(
    kline_data: dict[str, Any],
    symbol: str,
    name: str,
    charts_dir: str | Path,
) -> dict[str, str]:
    """
    为所有已获取的K线数据生成图表，保存到 charts_dir。
    kline_data: {tf: {"ok": bool, "df": DataFrame, ...}}
    返回: {tf: 图片路径字符串} (仅成功的)
    """
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for tf in ("1h", "day", "week", "month"):
        item = kline_data.get(tf)
        if not item or not item.get("ok"):
            continue
        df = item.get("df")
        if df is None or df.empty:
            continue
        save_path = charts_dir / f"kline_{tf}.png"
        png = generate_kline_chart(df, tf, symbol, name, save_path=save_path)
        if png:
            result[tf] = str(save_path)
    return result


def load_image_base64(path: str | Path) -> str | None:
    """读取图片文件，返回 base64 字符串（无前缀）。"""
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except Exception as e:
        logger.warning("load_image_base64: failed %s: %s", path, e)
        return None


# ──────────────────────────────────────────────────────────────────
# 指标计算
# ──────────────────────────────────────────────────────────────────

def _calc_indicators(df) -> dict[str, Any]:
    """计算 MA / 布林带 / MACD / RSI / KDJ。优先用 pandas-ta，降级用纯 numpy。"""
    import numpy as np
    import pandas as pd

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)
    ind: dict[str, Any] = {}

    # ── MA ────────────────────────────────────────────────────────
    for p in (5, 10, 20, 60, 120, 250):
        if n >= p:
            ma = np.full(n, np.nan)
            for i in range(p - 1, n):
                ma[i] = close[i - p + 1:i + 1].mean()
            ind[f"ma{p}"] = ma
        else:
            ind[f"ma{p}"] = None

    # ── 布林带 (20,2) ─────────────────────────────────────────────
    if n >= 20:
        bb_u = np.full(n, np.nan)
        bb_m = np.full(n, np.nan)
        bb_l = np.full(n, np.nan)
        for i in range(19, n):
            sl = close[i - 19:i + 1]
            mid = sl.mean()
            std = sl.std(ddof=1)
            bb_m[i] = mid
            bb_u[i] = mid + 2 * std
            bb_l[i] = mid - 2 * std
        ind["bb_upper"] = bb_u
        ind["bb_mid"] = bb_m
        ind["bb_lower"] = bb_l
    else:
        ind["bb_upper"] = ind["bb_mid"] = ind["bb_lower"] = None

    # ── MACD (12,26,9) ────────────────────────────────────────────
    def _ema(arr, span):
        alpha = 2.0 / (span + 1)
        out = np.full(len(arr), np.nan)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out

    if n >= 26:
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        dif = ema12 - ema26
        dea = _ema(dif, 9)
        hist = (dif - dea) * 2
        ind["macd_dif"] = dif
        ind["macd_dea"] = dea
        ind["macd_hist"] = hist
    else:
        ind["macd_dif"] = ind["macd_dea"] = ind["macd_hist"] = None

    # ── RSI (6/12/24) ─────────────────────────────────────────────
    def _rsi(arr, period):
        if len(arr) <= period:
            return None
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        out = np.full(len(arr), np.nan)
        avg_gain = gain[:period].mean()
        avg_loss = loss[:period].mean()
        if avg_loss == 0:
            out[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[period] = 100.0 - 100.0 / (1.0 + rs)
        for i in range(period + 1, len(arr)):
            avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
            if avg_loss == 0:
                out[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[i] = 100.0 - 100.0 / (1.0 + rs)
        return out

    ind["rsi6"] = _rsi(close, 6)
    ind["rsi12"] = _rsi(close, 12)
    ind["rsi24"] = _rsi(close, 24)

    # ── KDJ (9,3,3) ──────────────────────────────────────────────
    if n >= 9:
        k_arr = np.full(n, 50.0)
        d_arr = np.full(n, 50.0)
        for i in range(8, n):
            sl_h = high[i - 8:i + 1]
            sl_l = low[i - 8:i + 1]
            hn = sl_h.max()
            ln = sl_l.min()
            if hn == ln:
                rsv = 50.0
            else:
                rsv = (close[i] - ln) / (hn - ln) * 100.0
            k_arr[i] = 2.0 / 3.0 * k_arr[i - 1] + 1.0 / 3.0 * rsv
            d_arr[i] = 2.0 / 3.0 * d_arr[i - 1] + 1.0 / 3.0 * k_arr[i]
        j_arr = 3.0 * k_arr - 2.0 * d_arr
        # 前8根置 NaN
        k_arr[:8] = np.nan
        d_arr[:8] = np.nan
        j_arr[:8] = np.nan
        ind["kdj_k"] = k_arr
        ind["kdj_d"] = d_arr
        ind["kdj_j"] = j_arr
    else:
        ind["kdj_k"] = ind["kdj_d"] = ind["kdj_j"] = None

    return ind


# ──────────────────────────────────────────────────────────────────
# 绘图辅助
# ──────────────────────────────────────────────────────────────────

def _style_ax(ax) -> None:
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#888888", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.grid(color="#1e2130", linewidth=0.4, linestyle="--", alpha=0.6)


def _draw_candles(ax, df, x) -> None:
    import numpy as np
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection, PatchCollection

    up_color = "#EF5350"    # 红涨（A股惯例）
    dn_color = "#26A69A"    # 绿跌

    rects_up, rects_dn = [], []
    wicks_up, wicks_dn = [], []
    bar_width = max(0.6, 0.85 - 0.002 * len(x))

    for i, row in df.iterrows():
        xi = x[i]
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        color = up_color if c >= o else dn_color
        body_y = min(o, c)
        body_h = abs(c - o) or (h - l) * 0.01
        rect = Rectangle((xi - bar_width / 2, body_y), bar_width, body_h)
        (rects_up if c >= o else rects_dn).append(rect)
        wick = [(xi, l), (xi, h)]
        (wicks_up if c >= o else wicks_dn).append(wick)

    for rects, color in ((rects_up, up_color), (rects_dn, dn_color)):
        if rects:
            pc = PatchCollection(rects, facecolor=color, edgecolor=color, linewidth=0.3, alpha=0.9)
            ax.add_collection(pc)
    for wicks, color in ((wicks_up, up_color), (wicks_dn, dn_color)):
        if wicks:
            lc = LineCollection(wicks, colors=color, linewidths=0.8, alpha=0.85)
            ax.add_collection(lc)

    ax.set_xlim(-1, len(x))
    price_min = df["low"].min()
    price_max = df["high"].max()
    pad = (price_max - price_min) * 0.05 or price_max * 0.02
    ax.set_ylim(price_min - pad, price_max + pad)


def _draw_volume(ax, df, x) -> None:
    import numpy as np

    up_color = "#EF5350"
    dn_color = "#26A69A"
    bar_width = max(0.6, 0.85 - 0.002 * len(x))

    colors = [up_color if float(row["close"]) >= float(row["open"]) else dn_color for _, row in df.iterrows()]
    ax.bar(x, df["volume"].values, width=bar_width, color=colors, alpha=0.75)
    ax.set_xlim(-1, len(x))


def _draw_macd(ax, ind: dict, x) -> None:
    import numpy as np

    dif = ind.get("macd_dif")
    dea = ind.get("macd_dea")
    hist = ind.get("macd_hist")
    if dif is None:
        ax.text(0.5, 0.5, "MACD: 数据不足", transform=ax.transAxes,
                ha="center", va="center", color="#888", fontsize=8)
        return
    bar_width = max(0.6, 0.85 - 0.002 * len(x))
    colors = ["#EF5350" if v >= 0 else "#26A69A" for v in hist]
    ax.bar(x, hist, width=bar_width, color=colors, alpha=0.7)
    valid = ~np.isnan(dif)
    if valid.sum() > 1:
        ax.plot(x[valid], dif[valid], color="#FF6B35", linewidth=0.9, label="DIF")
        ax.plot(x[valid], dea[valid], color="#FFD700", linewidth=0.9, label="DEA")
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.set_xlim(-1, len(x))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=6,
                  facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", framealpha=0.6)


def _draw_rsi(ax, ind: dict, x) -> None:
    import numpy as np

    rsi_cfg = [("rsi6", "#FF6B35"), ("rsi12", "#FFD700"), ("rsi24", "#00BFFF")]
    any_plotted = False
    for key, color in rsi_cfg:
        arr = ind.get(key)
        if arr is None:
            continue
        valid = ~np.isnan(arr)
        if valid.sum() > 1:
            ax.plot(x[valid], arr[valid], color=color, linewidth=0.9, label=key.upper())
            any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "RSI: 数据不足", transform=ax.transAxes,
                ha="center", va="center", color="#888", fontsize=8)
        return
    ax.axhline(70, color="#EF5350", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axhline(30, color="#26A69A", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 100)
    ax.set_xlim(-1, len(x))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=6,
                  facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", framealpha=0.6)


def _draw_kdj(ax, ind: dict, x) -> None:
    import numpy as np

    kdj_cfg = [("kdj_k", "#FF6B35"), ("kdj_d", "#FFD700"), ("kdj_j", "#00BFFF")]
    any_plotted = False
    for key, color in kdj_cfg:
        arr = ind.get(key)
        if arr is None:
            continue
        valid = ~np.isnan(arr)
        if valid.sum() > 1:
            ax.plot(x[valid], arr[valid], color=color, linewidth=0.9,
                    label=key.split("_")[1].upper())
            any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "KDJ: 数据不足", transform=ax.transAxes,
                ha="center", va="center", color="#888", fontsize=8)
        return
    ax.axhline(80, color="#EF5350", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.axhline(20, color="#26A69A", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_xlim(-1, len(x))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=6,
                  facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", framealpha=0.6)


def _draw_trend_lines(ax, df, x) -> None:
    """自动趋势线：近期最高点高点连线 + 最低点低点连线（简化版，取最近30根）。"""
    import numpy as np

    n = len(df)
    window = min(30, n)
    recent = df.tail(window).reset_index(drop=True)
    x_recent = x[n - window:]

    highs = recent["high"].values.astype(float)
    lows = recent["low"].values.astype(float)

    # 找局部极值点（简单峰谷）
    def _local_extrema(arr, is_max: bool, min_gap: int = 3):
        pts = []
        for i in range(min_gap, len(arr) - min_gap):
            sl = arr[i - min_gap:i + min_gap + 1]
            if is_max and arr[i] == sl.max():
                pts.append(i)
            elif not is_max and arr[i] == sl.min():
                pts.append(i)
        return pts

    top_pts = _local_extrema(highs, True)
    bot_pts = _local_extrema(lows, False)

    # 上升趋势（底底连线，取最近2个低点）
    if len(bot_pts) >= 2:
        p1, p2 = bot_pts[-2], bot_pts[-1]
        if p2 > p1:
            x1, y1 = x_recent[p1], lows[p1]
            x2, y2 = x_recent[p2], lows[p2]
            # 延伸到右侧
            xr = x_recent[-1] + window // 4
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            yr = y2 + slope * (xr - x2)
            color = "#26A69A" if y2 >= y1 else "#EF5350"
            ax.plot([x1, xr], [y1, yr], color=color, linewidth=0.9, linestyle="--", alpha=0.6)

    # 下降趋势（顶顶连线，取最近2个高点）
    if len(top_pts) >= 2:
        p1, p2 = top_pts[-2], top_pts[-1]
        if p2 > p1:
            x1, y1 = x_recent[p1], highs[p1]
            x2, y2 = x_recent[p2], highs[p2]
            xr = x_recent[-1] + window // 4
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            yr = y2 + slope * (xr - x2)
            color = "#EF5350" if y2 <= y1 else "#26A69A"
            ax.plot([x1, xr], [y1, yr], color=color, linewidth=0.9, linestyle="--", alpha=0.6)


def _set_xticks(ax, df, x) -> None:
    """在最后一个子图设置 X 轴日期刻度（最多显示 8 个）。"""
    import numpy as np

    n = len(df)
    num_ticks = min(8, n)
    if num_ticks < 2:
        return
    tick_idx = np.linspace(0, n - 1, num_ticks, dtype=int)
    tick_labels = []
    for i in tick_idx:
        dt = df.iloc[i]["dt"] if "dt" in df.columns else ""
        if hasattr(dt, "strftime"):
            tick_labels.append(dt.strftime("%y/%m/%d"))
        else:
            tick_labels.append(str(i))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, fontsize=7, color="#888888", rotation=20)
