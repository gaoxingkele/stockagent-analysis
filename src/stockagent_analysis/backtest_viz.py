"""Sprint 2.3: backtest 净值曲线 + 回撤可视化.

matplotlib 输出 PNG, 含:
  - 净值曲线 vs hs300 基准
  - 回撤曲线 (含底色)
  - 月度收益柱状图
  - 仓位/持仓数时序
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates


def _setup_font():
    for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        try:
            fm.fontManager.addfont(fp)
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


_setup_font()


def fetch_hs300_curve(start: str, end: str) -> pd.DataFrame:
    """拉 hs300 净值曲线 (从 Tushare)."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    import tushare as ts
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    pro = ts.pro_api()
    df = pro.index_daily(ts_code="000300.SH", start_date=start, end_date=end)
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df["hs300_nav"] = df["close"] / df["close"].iloc[0]
    return df[["trade_date", "hs300_nav"]]


def plot_backtest(nav_csv_paths: dict, out_png: Path,
                   title: str = "V12 Backtest 净值曲线"):
    """nav_csv_paths: {strategy_name: csv_path}."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.5, 1.5, 1])

    ax1 = fig.add_subplot(gs[0])  # 净值
    ax2 = fig.add_subplot(gs[1])  # 回撤
    ax3 = fig.add_subplot(gs[2])  # 月度收益
    ax4 = fig.add_subplot(gs[3])  # 仓位/持仓数

    colors = ["#7B61FF", "#3FB950", "#D29922", "#F85149"]

    # 拉 hs300 基准
    all_dates = []
    for p in nav_csv_paths.values():
        d = pd.read_csv(p)
        all_dates += d["date"].astype(str).tolist()
    start = min(all_dates); end = max(all_dates)
    hs = fetch_hs300_curve(start, end)
    hs["dt"] = pd.to_datetime(hs["trade_date"], format="%Y%m%d")
    ax1.plot(hs["dt"], hs["hs300_nav"], color="#8A8F9A",
              label="hs300 基准", linestyle="--", linewidth=1.5)

    # 各策略
    summaries = {}
    for i, (name, p) in enumerate(nav_csv_paths.items()):
        df = pd.read_csv(p)
        df["dt"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        df["dd"] = (df["nav"] / df["nav"].cummax() - 1) * 100
        c = colors[i % len(colors)]
        ax1.plot(df["dt"], df["nav"], color=c, label=name, linewidth=2.0)
        ax2.fill_between(df["dt"], df["dd"], 0, color=c, alpha=0.25, label=name)

        # 月度收益
        df["ym"] = df["date"].astype(str).str[:6]
        m = df.groupby("ym").agg(nav_start=("nav", "first"), nav_end=("nav", "last"))
        m["ret_pct"] = (m["nav_end"] / m["nav_start"] - 1) * 100
        x = np.arange(len(m))
        ax3.bar(x + i*0.3, m["ret_pct"], width=0.3, color=c, label=name)

        # 仓位曲线
        if "n_holdings" in df.columns:
            ax4.plot(df["dt"], df["n_holdings"], color=c, label=f"{name} 持仓数",
                      linewidth=1.5)

        summaries[name] = {
            "final_nav": float(df["nav"].iloc[-1]),
            "total_return_pct": (float(df["nav"].iloc[-1]) - 1) * 100,
            "max_dd_pct": float(df["dd"].min()),
        }

    ax1.set_title(title, fontsize=14, weight="bold")
    ax1.set_ylabel("净值")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax2.set_ylabel("回撤 %")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    if 'm' in locals():
        ax3.set_xticks(x)
        ax3.set_xticklabels(m.index, rotation=0)
    ax3.set_ylabel("月度收益 %")
    ax3.axhline(y=0, color="#333", linewidth=0.5)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(alpha=0.3, axis="y")

    ax4.set_ylabel("持仓数")
    ax4.set_xlabel("日期")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    return summaries
