#!/usr/bin/env python3
"""配对实验: 同买点 同持有期 选股 vs 基准.

每次系统给出买点 t0:
  策略: 买入 top 10 (我们的策略), 持有 5 天, 计算 r5_strategy
  基准: 同时买入 ETF / 等权小盘, 持有 5 天, 计算 r5_benchmark

每次 t0 都得到 (r5_strategy, r5_benchmark) 配对.
统计:
  - 跑赢比例: r5_strategy > r5_benchmark 的次数 / 总次数
  - 平均超额: mean(r5_strategy - r5_benchmark)
  - 显著性: t-test
  - 分布: 按 regime / mv 切片

输出: output/benchmark/paired.csv
"""
from __future__ import annotations
import json, os, struct, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
TRADES_PATH = ROOT / "output" / "portfolio_bt" / "trades.csv"
OUT_DIR = ROOT / "output" / "benchmark"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

INDICES = {
    "hs300": ("sh", "000300"),
    "cyb":   ("sz", "399006"),
    "zz500": ("sh", "000905"),
    "zz1000":("sh", "000852"),
    "bj50":  ("bj", "899050"),
}


def read_index(market, code):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    data = p.read_bytes()
    n = len(data) // 32
    rows = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append((d.strftime("%Y%m%d"), f[4]/100.0))
    return pd.DataFrame(rows, columns=["trade_date", "close"])


def index_r5(idx_df, t0):
    """同期 5 日 (t0 → t0+5) 收益. t0 当日收盘买, t0+5 当日收盘卖."""
    if idx_df is None: return None
    dates = idx_df["trade_date"].tolist()
    if t0 not in dates: return None
    i = dates.index(t0)
    if i + 5 >= len(dates): return None
    p0 = idx_df.iloc[i]["close"]
    p1 = idx_df.iloc[i+5]["close"]
    return (p1 / p0 - 1) * 100


def main():
    if not TRADES_PATH.exists():
        print("先跑 portfolio_backtest.py 生成 trades.csv")
        return
    trades = pd.read_csv(TRADES_PATH)
    trades["rb_date"] = trades["rb_date"].astype(str)
    print(f"载入 {len(trades)} 条 trades")

    # 每个换仓日 our portfolio r5 = top 10 平均
    portfolio = trades.groupby("rb_date").agg(
        port_r5=("r5", "mean"),
        regime=("regime", "first"),
        n_stocks=("ts_code", "count"),
    ).reset_index()
    print(f"换仓日数: {len(portfolio)}")

    # 加载指数
    idx_data = {}
    for name, (m, c) in INDICES.items():
        d = read_index(m, c)
        if d is not None:
            idx_data[name] = d.sort_values("trade_date").reset_index(drop=True)

    # 加载 20-50 亿等权 r5
    print("加载 20-50亿等权 r5...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code","trade_date","total_mv","r5"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= portfolio["rb_date"].min()) &
                (df["trade_date"] <= portfolio["rb_date"].max())]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full = full[(full["total_mv"]/1e4 >= 20) & (full["total_mv"]/1e4 < 50)]
    full = full.dropna(subset=["r5"])
    small_r5 = full.groupby("trade_date")["r5"].mean()

    # 每个换仓日配对
    rows = []
    for _, p in portfolio.iterrows():
        t0 = p["rb_date"]
        row = {"rb_date": t0, "port_r5": p["port_r5"], "regime": p["regime"]}
        for name, idx_df in idx_data.items():
            row[f"{name}_r5"] = index_r5(idx_df, t0)
        row["small_r5"] = small_r5.get(t0)
        rows.append(row)
    paired = pd.DataFrame(rows).dropna(subset=["port_r5"])
    paired.to_csv(OUT_DIR / "paired.csv", index=False, encoding="utf-8-sig")

    # 配对分析
    print("\n" + "=" * 78)
    print("配对实验: 每次买点 同持有 5 天 — 我们策略 vs 基准")
    print("=" * 78)
    print(f"\n{'基准':<22} {'平均超额':>10} {'跑赢次数':>10} {'胜率':>7} "
          f"{'p值':>7} {'累计超额(独立)':>14}")
    print("-" * 78)

    cn_map = {"hs300":"沪深300", "cyb":"创业板指", "zz500":"中证500",
              "zz1000":"中证1000", "bj50":"北证50", "small":"20-50亿等权"}

    for col, name in [("hs300_r5","hs300"), ("cyb_r5","cyb"), ("zz500_r5","zz500"),
                       ("zz1000_r5","zz1000"), ("bj50_r5","bj50"), ("small_r5","small")]:
        sub = paired.dropna(subset=[col])
        if len(sub) == 0: continue
        diff = sub["port_r5"] - sub[col]
        avg_excess = diff.mean()
        win_count = (diff > 0).sum()
        win_rate = win_count / len(sub) * 100
        # t-test 单尾 (我们 > 基准)
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2  # 单尾
        # 累计超额 (独立配对相加, 不复利)
        cum_excess = diff.sum()
        print(f"{cn_map[name]:<19} {avg_excess:>+9.3f}pp {win_count:>5}/{len(sub):<4} "
              f"{win_rate:>5.1f}% {p_one:>6.3f} {cum_excess:>+12.2f}pp")

    # 按 regime 切片
    print("\n=== 按 regime 切片 (vs 沪深300) ===")
    for regime, grp in paired.dropna(subset=["hs300_r5"]).groupby("regime", observed=True):
        diff = grp["port_r5"] - grp["hs300_r5"]
        win = (diff > 0).sum()
        print(f"  {regime:<22}: n={len(grp):>3}, 平均超额 {diff.mean():+6.3f}pp, "
              f"胜率 {win}/{len(grp)} = {win/len(grp)*100:.1f}%")

    # 按 regime 切片 (vs 中证500)
    print("\n=== 按 regime 切片 (vs 中证500) ===")
    for regime, grp in paired.dropna(subset=["zz500_r5"]).groupby("regime", observed=True):
        diff = grp["port_r5"] - grp["zz500_r5"]
        win = (diff > 0).sum()
        print(f"  {regime:<22}: n={len(grp):>3}, 平均超额 {diff.mean():+6.3f}pp, "
              f"胜率 {win}/{len(grp)} = {win/len(grp)*100:.1f}%")

    # 总结
    print(f"\n=== 总结 ===")
    print(f"配对实验在控制 '同买点 同持有期' 后, 才能反映系统的真 alpha.")
    print(f"如果配对胜率 ≈ 50%, 平均超额 ≈ 0 → 系统没有 alpha")
    print(f"如果配对胜率 > 55% AND 平均超额显著正 → 系统有 alpha")
    print(f"如果配对胜率 < 45% → 系统反向 alpha (越选越差)")


if __name__ == "__main__":
    main()
