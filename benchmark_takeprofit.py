#!/usr/bin/env python3
"""带止盈的择时配对实验 — 选股 vs ETF + Top 20 基金.

每次换仓 t0:
  - 买入 (持仓): 选股 top 10 / ETF / 基金
  - 持有 1~20 天:
    * 任一天 high >= entry × (1+止盈%) → 按止盈位卖出
    * 20 天到期未触发 → 按收盘价卖出
  - 计算实际"已实现收益"

止盈档位: +10%, +15%, +20% 三档
基金没有 high/low, 用 nav (当日净值) 触发

输出: output/benchmark/takeprofit_compare.csv + summary
"""
from __future__ import annotations
import json, os, struct, time, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
TRADES_PATH = ROOT / "output" / "portfolio_bt" / "trades.csv"
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OUT_DIR = ROOT / "output" / "benchmark"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

INDICES = {
    "hs300": ("sh", "000300", "沪深300"),
    "cyb":   ("sz", "399006", "创业板指"),
    "zz500": ("sh", "000905", "中证500"),
    "zz1000":("sh", "000852", "中证1000"),
    "bj50":  ("bj", "899050", "北证50"),
}

# Top 20 主动基金 (扩展到 20 只)
TOP_FUNDS = [
    ("004320", "前海开源沪港深乐享生活"),
    ("016370", "信澳业绩驱动混合A"),
    ("011369", "华商均衡成长混合A"),
    ("001753", "红土创新新兴产业混合A"),
    ("001412", "德邦鑫星价值灵活配置混合A"),
    ("001194", "景顺长城稳健回报混合A"),
    ("018956", "中航机遇领航混合发起A"),
    ("001437", "易方达瑞享混合I"),
    ("011891", "易方达先锋成长混合A"),
    ("010115", "易方达远见成长混合A"),
    ("001170", "宏利复兴混合A"),
    ("008326", "东财通信A"),
    ("006265", "红土创新新科技股票A"),
    ("005550", "汇安成长优选混合A"),
    ("004851", "华夏行业景气混合"),
    ("110011", "易方达优质精选混合"),
    ("016668", "永赢医药健康混合A"),
    ("020253", "信澳新能源精选混合A"),
    ("014218", "前海开源中航军工股票"),
    ("009535", "华夏中证5G通信主题ETF联接A"),
]

LOOKAHEAD = 20
TAKE_PROFITS = [0.10, 0.15, 0.20]


def read_index_ohlc(market, code):
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
        rows.append((d.strftime("%Y%m%d"), f[1]/100.0, f[2]/100.0,
                     f[3]/100.0, f[4]/100.0))
    return pd.DataFrame(rows, columns=["trade_date","open","high","low","close"])


def stock_takeprofit(stock_ohlc, t0, tp):
    """单股: t0 买入 (按 t0+1 open), 持有 20 天, 止盈或到期.
    返回实际收益 (%)"""
    if stock_ohlc is None or stock_ohlc.empty: return None
    dates = stock_ohlc["trade_date"].tolist()
    if t0 not in dates: return None
    i = dates.index(t0)
    if i + LOOKAHEAD >= len(dates): return None
    entry_open = stock_ohlc.iloc[i+1]["open"]
    if entry_open <= 0: return None
    target = entry_open * (1 + tp)
    for j in range(i+1, i+1+LOOKAHEAD):
        if stock_ohlc.iloc[j]["high"] >= target:
            # 止盈触发, 按止盈位卖
            return tp * 100
    # 到期
    close_end = stock_ohlc.iloc[i+LOOKAHEAD]["close"]
    return (close_end / entry_open - 1) * 100


def fund_takeprofit(nav_df, t0, tp):
    """基金: 用 nav 触发止盈."""
    if nav_df is None or nav_df.empty: return None
    dates = nav_df["trade_date"].tolist()
    if t0 not in dates: return None
    i = dates.index(t0)
    if i + LOOKAHEAD >= len(dates): return None
    entry_nav = nav_df.iloc[i+1]["nav"] if i+1 < len(dates) else None  # T+1 净值
    if entry_nav is None or entry_nav <= 0: return None
    target = entry_nav * (1 + tp)
    for j in range(i+1, i+1+LOOKAHEAD):
        if nav_df.iloc[j]["nav"] >= target:
            return tp * 100
    close_end = nav_df.iloc[i+LOOKAHEAD]["nav"]
    return (close_end / entry_nav - 1) * 100


def fetch_fund_nav(code: str) -> pd.DataFrame:
    import akshare as ak
    for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
        os.environ.pop(k, None)
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or df.empty: return pd.DataFrame()
        df["净值日期"] = pd.to_datetime(df["净值日期"]).dt.strftime("%Y%m%d")
        df = df.rename(columns={"净值日期":"trade_date", "单位净值":"nav"})
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        return df[["trade_date","nav"]].dropna().sort_values("trade_date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def read_stock_ohlc(ts):
    c, m = code_market(ts)
    return read_index_ohlc(m, c)


def main():
    t_start = time.time()
    if not TRADES_PATH.exists():
        print("先跑 portfolio_backtest.py")
        return
    trades = pd.read_csv(TRADES_PATH)
    trades["rb_date"] = trades["rb_date"].astype(str)

    # 我们的策略: 每只股 + 每个止盈档算 实际收益, top 10 平均
    print("加载所有股票 OHLC (cache by ts_code)...")
    stock_cache = {}
    for ts in trades["ts_code"].unique():
        stock_cache[ts] = read_stock_ohlc(ts)

    print(f"\n计算选股 {len(TAKE_PROFITS)} 档止盈...")
    for tp in TAKE_PROFITS:
        col = f"port_tp_{int(tp*100)}"
        trades[col] = trades.apply(
            lambda r: stock_takeprofit(stock_cache.get(r["ts_code"]), r["rb_date"], tp),
            axis=1)

    portfolio = trades.groupby("rb_date").agg(
        regime=("regime","first"),
        port_tp_10=("port_tp_10","mean"),
        port_tp_15=("port_tp_15","mean"),
        port_tp_20=("port_tp_20","mean"),
    ).reset_index()
    print(f"换仓日数: {len(portfolio)}")

    # 加载指数
    print("\n加载指数 OHLC...")
    idx_data = {}
    for name, (m, c, cn) in INDICES.items():
        d = read_index_ohlc(m, c)
        if d is not None:
            idx_data[name] = (cn, d.sort_values("trade_date").reset_index(drop=True))

    # 加载基金
    print("\n加载基金净值...")
    fund_data = {}
    for code, name in TOP_FUNDS:
        nav = fetch_fund_nav(code)
        if not nav.empty:
            fund_data[code] = (name, nav)
            print(f"  {code} {name[:20]}: {len(nav)} 行")
        time.sleep(0.3)

    # 配对计算每个 rb_date 各方止盈实际收益
    print("\n计算止盈对比...")
    rows = []
    for _, p in portfolio.iterrows():
        t0 = p["rb_date"]
        row = {"rb_date": t0, "regime": p["regime"]}
        for tp_pct in TAKE_PROFITS:
            row[f"port_tp_{int(tp_pct*100)}"] = p[f"port_tp_{int(tp_pct*100)}"]
        for name, (cn, idx_df) in idx_data.items():
            for tp_pct in TAKE_PROFITS:
                row[f"{name}_tp_{int(tp_pct*100)}"] = stock_takeprofit(idx_df, t0, tp_pct)
        for code, (cn, nav) in fund_data.items():
            for tp_pct in TAKE_PROFITS:
                row[f"f{code}_tp_{int(tp_pct*100)}"] = fund_takeprofit(nav, t0, tp_pct)
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT_DIR / "takeprofit_compare.csv", index=False, encoding="utf-8-sig")

    # 报告
    print(f"\n{'='*100}")
    print(f"带止盈实战对比 (D+20 持有期, 多档止盈, n={len(paired)} 个换仓日)")
    print(f"{'='*100}")

    for tp_pct in TAKE_PROFITS:
        tp = int(tp_pct * 100)
        port_col = f"port_tp_{tp}"
        port_avg = paired[port_col].mean()
        port_pos = (paired[port_col] > 0).mean() * 100
        print(f"\n--- 止盈位 +{tp}% ---")
        print(f"我们策略 (top 10 等权): 平均实际收益 {port_avg:+.2f}%/期, 正收益期数 {port_pos:.1f}%")

        print(f"\n{'基准':<28} {'实际收益':>10} {'胜率(超过我们)':>14}")
        # ETF
        for name, (cn, _) in idx_data.items():
            col = f"{name}_tp_{tp}"
            if col not in paired.columns: continue
            sub = paired.dropna(subset=[col, port_col])
            if len(sub) == 0: continue
            avg = sub[col].mean()
            our_win = ((sub[port_col] - sub[col]) > 0).sum()
            print(f"  {cn:<25} {avg:>+8.2f}% {our_win:>3}/{len(sub):<3} ({our_win/len(sub)*100:>4.1f}%)")
        # 基金 (汇总)
        fund_cols = [f"f{code}_tp_{tp}" for code, _ in fund_data.items()
                      if f"f{code}_tp_{tp}" in paired.columns]
        if fund_cols:
            paired_f = paired.copy()
            paired_f["funds_avg"] = paired_f[fund_cols].mean(axis=1, skipna=True)
            sub = paired_f.dropna(subset=["funds_avg", port_col])
            if len(sub) > 0:
                avg = sub["funds_avg"].mean()
                our_win = ((sub[port_col] - sub["funds_avg"]) > 0).sum()
                print(f"  Top 20 基金等权篮子        {avg:>+8.2f}% "
                      f"{our_win:>3}/{len(sub):<3} ({our_win/len(sub)*100:>4.1f}%)")

    # 累计收益对比 (复利, 假设每次换仓后剩余资金继续投)
    print(f"\n=== 累计复利收益 (36 期换仓, +15% 止盈位) ===")
    tp_col = "port_tp_15"
    sub = paired.dropna(subset=[tp_col])
    sub_cum = (1 + sub[tp_col].fillna(0) / 100).cumprod()
    print(f"  我们策略累计: {(sub_cum.iloc[-1] - 1) * 100:+.2f}%")
    for name, (cn, _) in idx_data.items():
        col = f"{name}_tp_15"
        if col not in paired.columns: continue
        sub2 = paired.dropna(subset=[col])
        cum = (1 + sub2[col] / 100).cumprod()
        print(f"  {cn:<22}: {(cum.iloc[-1] - 1) * 100:+.2f}%")
    # 基金平均
    if fund_cols:
        sub3 = paired.dropna()
        fund_avg = paired[[f"f{code}_tp_15" for code, _ in fund_data.items()
                           if f"f{code}_tp_15" in paired.columns]].mean(axis=1, skipna=True)
        cum = (1 + fund_avg.fillna(0) / 100).cumprod()
        print(f"  Top 20 基金等权        : {(cum.iloc[-1] - 1) * 100:+.2f}%")

    print(f"\n总耗时 {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
