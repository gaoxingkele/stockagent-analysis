#!/usr/bin/env python3
"""组合策略 vs 大盘指数基准对比.

加载:
  - 组合策略: output/portfolio_bt/equity_curve.csv
  - 沪深300/创业板/中证500: 从 daily_regime.parquet (含 close 衍生)
  - 等权小盘: 从因子 parquet 计算所有 20-50 亿股票每日等权收益

输出:
  output/benchmark/index_compare.csv
  output/benchmark/excess_summary.json
"""
from __future__ import annotations
import json, os, struct, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
PORTFOLIO_EQ = ROOT / "output" / "portfolio_bt" / "equity_curve.csv"
OUT_DIR = ROOT / "output" / "benchmark"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

INDICES = {
    "hs300": ("sh", "000300"),  # 沪深300
    "cyb":   ("sz", "399006"),  # 创业板指
    "zz500": ("sh", "000905"),  # 中证500
    "zz1000":("sh", "000852"),  # 中证1000 (小盘代表)
    "bj50":  ("bj", "899050"),  # 北证50
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


def main():
    # 加载组合策略
    if not PORTFOLIO_EQ.exists():
        print(f"❌ 没找到 {PORTFOLIO_EQ}")
        return
    eq = pd.read_csv(PORTFOLIO_EQ)
    eq["trade_date"] = eq["date"].astype(str)
    start_date = eq["trade_date"].min()
    end_date   = eq["trade_date"].max()
    print(f"组合策略期间: {start_date} → {end_date}")
    print(f"组合期末值: {eq['equity'].iloc[-1]:.4f}")

    # 加载各指数
    bench = pd.DataFrame({"trade_date": eq["trade_date"]})
    for name, (m, c) in INDICES.items():
        idx = read_index(m, c)
        if idx is None:
            print(f"  ❌ {name} 加载失败")
            continue
        idx = idx[(idx["trade_date"] >= start_date) & (idx["trade_date"] <= end_date)]
        # 归一化到 起点 = 1.0
        if len(idx) == 0: continue
        base_close = idx.iloc[0]["close"]
        idx[f"{name}_eq"] = idx["close"] / base_close
        bench = bench.merge(idx[["trade_date", f"{name}_eq"]], on="trade_date", how="left")
        print(f"  {name}: 期末 {idx[f'{name}_eq'].iloc[-1]:.4f}")

    # 加载等权小盘 (20-50亿) 收益 — 只在换仓日累加 r5 (避免 5 日重复)
    print("加载小盘等权 (20-50亿)...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code","trade_date","total_mv","r5"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full = full[(full["total_mv"]/1e4 >= 20) & (full["total_mv"]/1e4 < 50)]
    full = full.dropna(subset=["r5"])
    # 取换仓日 (周频, 与组合一致)
    rebalance_dates = sorted(eq["trade_date"].tolist())
    daily_avg = full.groupby("trade_date")["r5"].mean() / 100
    # 只在 rebalance 日累加 (与组合周频一致)
    cum = 1.0
    rows = [{"trade_date": rebalance_dates[0], "small_eq": 1.0}]
    for i in range(1, len(rebalance_dates)):
        rb = rebalance_dates[i]
        if rb in daily_avg.index:
            r = daily_avg.loc[rb]
            cum *= (1 + r)
        rows.append({"trade_date": rb, "small_eq": cum})
    small_eq = pd.DataFrame(rows)
    bench = bench.merge(small_eq, on="trade_date", how="left")
    print(f"  20-50亿等权: 期末 {bench['small_eq'].dropna().iloc[-1]:.4f}")

    # 合并组合
    bench = bench.merge(eq[["trade_date","equity"]], on="trade_date", how="left")
    bench = bench.rename(columns={"equity": "portfolio_eq"})
    bench = bench.dropna(subset=["portfolio_eq"])
    # 前向填充指数（处理换仓日不在交易日的情况）
    for col in [c for c in bench.columns if c.endswith("_eq")]:
        bench[col] = bench[col].ffill()

    bench.to_csv(OUT_DIR / "index_compare.csv", index=False, encoding="utf-8-sig")

    # 汇总
    print("\n" + "=" * 70)
    print(f"基准对比 ({start_date} → {end_date})")
    print("=" * 70)
    pn = bench["portfolio_eq"].iloc[-1] - 1
    print(f"\n{'策略':<25} {'累计收益':>10} {'年化收益':>10} {'相对组合':>10}")
    print("-" * 60)
    n_periods = len(bench) - 1
    days_held = n_periods * 5  # 周频
    for col in ["portfolio_eq", "hs300_eq", "cyb_eq", "zz500_eq", "zz1000_eq",
                 "bj50_eq", "small_eq"]:
        if col not in bench.columns: continue
        ret = bench[col].dropna().iloc[-1] - 1 if not bench[col].dropna().empty else 0
        ann = (1 + ret) ** (252 / days_held) - 1 if days_held > 0 else 0
        rel = (ret - pn) * 100
        cn = {"portfolio_eq":"组合策略 (Top10/周)",
              "hs300_eq":"沪深300",
              "cyb_eq":"创业板指",
              "zz500_eq":"中证500",
              "zz1000_eq":"中证1000",
              "bj50_eq":"北证50",
              "small_eq":"20-50亿等权"}.get(col, col)
        print(f"{cn:<22} {ret*100:>+8.2f}% {ann*100:>+8.2f}% {rel:>+8.2f}pp")

    # 计算超额 / 信息比率
    print("\n=== 超额收益 (vs 沪深300) ===")
    if "hs300_eq" in bench.columns:
        bench["port_ret"] = bench["portfolio_eq"].pct_change()
        bench["hs_ret"]   = bench["hs300_eq"].pct_change()
        bench["excess"]   = bench["port_ret"] - bench["hs_ret"]
        excess = bench["excess"].dropna().values
        if len(excess) > 0:
            ann_excess = excess.mean() * 252
            tracking_err = excess.std() * np.sqrt(252)
            ir = ann_excess / tracking_err if tracking_err > 0 else 0
            print(f"年化超额收益:  {ann_excess*100:+.2f}%")
            print(f"跟踪误差:      {tracking_err*100:.2f}%")
            print(f"信息比率 IR:   {ir:.2f}")
            print(f"周次跑赢比例:  {(excess > 0).mean()*100:.1f}%")

    # 保存
    summary = {}
    for col in [c for c in bench.columns if c.endswith("_eq")]:
        v = bench[col].dropna()
        if v.empty: continue
        ret = v.iloc[-1] - 1
        summary[col] = {
            "final_return_pct": float(ret * 100),
            "annual_return_pct": float(((1 + ret) ** (252 / max(days_held, 1)) - 1) * 100),
        }
    Path(OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
