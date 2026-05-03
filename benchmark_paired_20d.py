#!/usr/bin/env python3
"""配对实验 (20 天持有期) — 选股 vs 基准.

每次系统给出买点 t0:
  策略: 买入 top 10, 持有 20 天, 计算
    - max_gain_20: top 10 平均期间最高涨幅
    - r20:         top 10 平均 20 天累计涨幅 (close-to-close)
  基准: 同时买入 ETF/基金, 持有 20 天, 计算
    - 期间最高涨幅 (从 t0 close 到 t0+20 内最高)
    - 20 天累计涨幅 (close → close)

输出:
  output/benchmark/paired_20d.csv
  按 regime / mv 切片汇总
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
LABELS = ROOT / "output" / "labels" / "max_gain_labels.parquet"
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

# Top 10 主动基金 (近 2 年最强)
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
]


def read_index_ohlc(market, code):
    """读 TDX OHLC 完整 (含 high/low)."""
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


def index_20d_returns(idx_df, t0, lookahead=20):
    """从 t0 起持有 20 天: max_gain (期间最高) + r20 (终值)."""
    if idx_df is None: return None, None
    dates = idx_df["trade_date"].tolist()
    if t0 not in dates: return None, None
    i = dates.index(t0)
    if i + lookahead >= len(dates): return None, None
    p0 = idx_df.iloc[i]["close"]
    future = idx_df.iloc[i+1: i+1+lookahead]
    if len(future) < lookahead: return None, None
    max_h = future["high"].max()
    close_end = future.iloc[-1]["close"]
    max_gain = (max_h / p0 - 1) * 100
    r20 = (close_end / p0 - 1) * 100
    return max_gain, r20


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
    except Exception as e:
        return pd.DataFrame()


def fund_20d_returns(nav_df, t0, lookahead=20):
    """基金 20 天: max_gain (期间净值最高) + r20 (终值)."""
    if nav_df is None or nav_df.empty: return None, None
    dates = nav_df["trade_date"].tolist()
    if t0 not in dates: return None, None
    i = dates.index(t0)
    if i + lookahead >= len(dates): return None, None
    p0 = nav_df.iloc[i]["nav"]
    future = nav_df.iloc[i+1: i+1+lookahead]
    if len(future) < lookahead: return None, None
    max_n = future["nav"].max()
    close_end = future.iloc[-1]["nav"]
    return (max_n / p0 - 1) * 100, (close_end / p0 - 1) * 100


def main():
    t_start = time.time()
    if not TRADES_PATH.exists():
        print("先跑 portfolio_backtest.py")
        return

    trades = pd.read_csv(TRADES_PATH)
    trades["rb_date"] = trades["rb_date"].astype(str)
    # 我们的策略: top 10 平均 max_gain_20 + 平均 r20 close
    # trades 已经有 max_gain_20, 但需要 r20 (close-to-close)
    # 从 labels 拿 r20_close
    labels = pd.read_parquet(LABELS, columns=["ts_code","trade_date","max_gain_20","r20_close"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    trades = trades.merge(labels.rename(columns={"trade_date":"rb_date"}),
                            on=["ts_code","rb_date"], how="left",
                            suffixes=("","_label"))
    # 优先用 labels 的 max_gain_20
    if "max_gain_20_label" in trades.columns:
        trades["max_gain_20"] = trades["max_gain_20_label"].fillna(trades["max_gain_20"])

    portfolio = trades.groupby("rb_date").agg(
        port_max_gain=("max_gain_20","mean"),
        port_r20=("r20_close","mean"),
        regime=("regime","first"),
    ).reset_index().dropna(subset=["port_max_gain"])
    print(f"换仓日数: {len(portfolio)}")

    # 加载指数
    print("加载指数 OHLC...")
    idx_data = {}
    for name, (m, c, cn) in INDICES.items():
        d = read_index_ohlc(m, c)
        if d is not None:
            idx_data[name] = (cn, d.sort_values("trade_date").reset_index(drop=True))
            print(f"  {cn}: {len(d)} 行")

    # 加载基金
    print("\n加载基金净值...")
    fund_data = {}
    for code, name in TOP_FUNDS:
        nav = fetch_fund_nav(code)
        if not nav.empty:
            fund_data[code] = (name, nav)
            print(f"  {code} {name}: {len(nav)} 行")
        time.sleep(0.3)

    # 配对 20 天收益
    print("\n配对 20 天 (max_gain + r20)...")
    rows = []
    for _, p in portfolio.iterrows():
        t0 = p["rb_date"]
        row = {"rb_date": t0, "regime": p["regime"],
                "port_max_gain": p["port_max_gain"],
                "port_r20": p["port_r20"]}
        for name, (cn, idx_df) in idx_data.items():
            mg, r20 = index_20d_returns(idx_df, t0)
            row[f"{name}_max_gain"] = mg
            row[f"{name}_r20"] = r20
        for code, (cn, nav) in fund_data.items():
            mg, r20 = fund_20d_returns(nav, t0)
            row[f"f{code}_max_gain"] = mg
            row[f"f{code}_r20"] = r20
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT_DIR / "paired_20d.csv", index=False, encoding="utf-8-sig")

    # ── 报告 ────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"配对实验 (20 天持有, 比期间最高 + 终值): n={len(paired)} 个换仓日")
    print(f"{'='*92}")

    print(f"\n{'基准':<25} {'max_gain 平均超额':>18} {'胜率':>10} "
          f"{'r20 平均超额':>15} {'胜率':>10}")
    print("-" * 92)

    def report(col_prefix: str, name: str):
        sub = paired.dropna(subset=[f"{col_prefix}_max_gain", f"{col_prefix}_r20"])
        if len(sub) == 0: return None
        mg_diff = sub["port_max_gain"] - sub[f"{col_prefix}_max_gain"]
        r20_diff = sub["port_r20"] - sub[f"{col_prefix}_r20"]
        mg_win = (mg_diff > 0).sum()
        r20_win = (r20_diff > 0).sum()
        print(f"{name:<22} {mg_diff.mean():>+15.2f}pp "
              f"{mg_win:>3}/{len(sub):<3} ({mg_win/len(sub)*100:>4.1f}%) "
              f"{r20_diff.mean():>+12.2f}pp "
              f"{r20_win:>3}/{len(sub):<3} ({r20_win/len(sub)*100:>4.1f}%)")
        return {
            "name": name,
            "mg_avg_excess": float(mg_diff.mean()),
            "mg_win_rate": float(mg_win / len(sub) * 100),
            "r20_avg_excess": float(r20_diff.mean()),
            "r20_win_rate": float(r20_win / len(sub) * 100),
            "n": len(sub),
        }

    results = []
    print("\n--- 指数 ---")
    for name, (cn, _) in idx_data.items():
        r = report(name, cn)
        if r: results.append(r)
    print("\n--- Top 10 主动基金 ---")
    for code, (name, _) in fund_data.items():
        r = report(f"f{code}", f"{code} {name[:14]}")
        if r: results.append(r)

    # 我们 vs 基金等权篮子
    fund_mg_cols = [c for c in paired.columns if c.startswith("f") and c.endswith("_max_gain")]
    fund_r20_cols = [c for c in paired.columns if c.startswith("f") and c.endswith("_r20")]
    if fund_mg_cols:
        paired["funds_basket_mg"] = paired[fund_mg_cols].mean(axis=1, skipna=True)
        paired["funds_basket_r20"] = paired[fund_r20_cols].mean(axis=1, skipna=True)
        sub = paired.dropna(subset=["funds_basket_mg","funds_basket_r20","port_max_gain","port_r20"])
        mg_diff = sub["port_max_gain"] - sub["funds_basket_mg"]
        r20_diff = sub["port_r20"] - sub["funds_basket_r20"]
        print(f"\n--- vs Top 10 基金等权篮子 ---")
        print(f"{'10基金等权':<22} {mg_diff.mean():>+15.2f}pp "
              f"{(mg_diff>0).sum():>3}/{len(sub):<3} ({(mg_diff>0).mean()*100:>4.1f}%) "
              f"{r20_diff.mean():>+12.2f}pp "
              f"{(r20_diff>0).sum():>3}/{len(sub):<3} ({(r20_diff>0).mean()*100:>4.1f}%)")

    # 按 regime 切片 (vs 沪深300 + 基金篮子)
    print(f"\n=== 按 regime 切片 ===")
    print(f"{'regime':<25} {'n':>4} {'max_gain vs 沪深300':>18} "
          f"{'r20 vs 沪深300':>15} {'r20 vs 基金篮子':>15}")
    for regime, grp in paired.groupby("regime", observed=True):
        sub = grp.dropna(subset=["port_max_gain","hs300_max_gain","port_r20","hs300_r20"])
        if len(sub) < 3: continue
        mg_d = (sub["port_max_gain"] - sub["hs300_max_gain"]).mean()
        r20_d = (sub["port_r20"] - sub["hs300_r20"]).mean()
        sub2 = grp.dropna(subset=["port_r20","funds_basket_r20"])
        fund_d = (sub2["port_r20"] - sub2["funds_basket_r20"]).mean() if len(sub2) > 0 else None
        fund_str = f"{fund_d:+.2f}pp" if fund_d is not None else "N/A"
        print(f"{regime:<25} {len(sub):>4} {mg_d:>+15.2f}pp "
              f"{r20_d:>+12.2f}pp {fund_str:>15}")

    Path(OUT_DIR / "paired_20d_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
