#!/usr/bin/env python3
"""配对实验: 我们的策略 vs 过去 1.5 年 top 10 基金.

步骤:
  1. akshare 拉所有公募基金, 按"近2年"收益排序
  2. 过滤指数型/纯债型, 取 top 10 主动管理基金
  3. 每只基金拉日净值
  4. 每个换仓日 t0, 计算 t0 → t0+5 净值变化
  5. 与组合策略 r5 配对比较

输出: output/benchmark/vs_top_funds.csv
"""
from __future__ import annotations
import os, time, json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
TRADES = ROOT / "output" / "portfolio_bt" / "trades.csv"
OUT_DIR = ROOT / "output" / "benchmark"
OUT_DIR.mkdir(exist_ok=True)


def get_top_funds(top_n: int = 30) -> pd.DataFrame:
    """取近 2 年表现最好的主动股票型/混合型基金."""
    import akshare as ak
    for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"]:
        os.environ.pop(k, None)
    print("拉基金排行榜...")
    df = ak.fund_open_fund_rank_em(symbol="全部")
    print(f"  总基金: {len(df)}")
    # 转近 2 年为数字
    df["近2年"] = pd.to_numeric(df["近2年"], errors="coerce")
    df["近1年"] = pd.to_numeric(df["近1年"], errors="coerce")
    df = df.dropna(subset=["近2年"])
    # 过滤名字含"指数"/"债"/"货币"/"纯债"等
    EXCL = ("指数", "债", "货币", "纯债", "ETF联接", "C", "Y")
    name = df["基金简称"].astype(str)
    df = df[~name.str.contains("|".join(["指数","债","货币","纯债","ETF联接"]), na=False)]
    # C 类后缀通常是同名 A 类的 C 份额, 排除
    df = df[~name.str.endswith("C")]
    df = df.sort_values("近2年", ascending=False).head(top_n)
    print(f"  top {top_n} 主动基金 (按近2年收益):")
    print(df[["基金代码","基金简称","近2年","近1年","成立来"]].head(15).to_string(index=False))
    return df.head(10)  # 取 top 10


def fetch_fund_nav(code: str) -> pd.DataFrame:
    """拉单只基金日净值 (akshare)."""
    import akshare as ak
    try:
        df = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        if df is None or df.empty: return pd.DataFrame()
        df["净值日期"] = pd.to_datetime(df["净值日期"]).dt.strftime("%Y%m%d")
        df = df.rename(columns={"净值日期":"trade_date", "单位净值":"nav"})
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        return df[["trade_date","nav"]].dropna().sort_values("trade_date").reset_index(drop=True)
    except Exception as e:
        print(f"  {code} 失败: {e}")
        return pd.DataFrame()


def fund_r5(nav_df, t0):
    """t0 → t0+5 净值收益 (5 个交易日)."""
    if nav_df is None or nav_df.empty: return None
    dates = nav_df["trade_date"].tolist()
    if t0 not in dates: return None
    i = dates.index(t0)
    if i + 5 >= len(dates): return None
    p0 = nav_df.iloc[i]["nav"]
    p1 = nav_df.iloc[i+5]["nav"]
    return (p1 / p0 - 1) * 100


def main():
    if not TRADES.exists():
        print("先跑 portfolio_backtest.py")
        return
    trades = pd.read_csv(TRADES)
    portfolio = trades.groupby("rb_date").agg(
        port_r5=("r5","mean"), regime=("regime","first")
    ).reset_index()
    portfolio["rb_date"] = portfolio["rb_date"].astype(str)
    print(f"换仓日数: {len(portfolio)}")

    top = get_top_funds(top_n=30)

    print(f"\n拉 top 10 基金日净值...")
    funds = {}
    for _, r in top.iterrows():
        code = r["基金代码"]
        name = r["基金简称"]
        nav = fetch_fund_nav(code)
        if not nav.empty:
            funds[code] = (name, nav)
            print(f"  {code} {name}: {len(nav)} 行")
        time.sleep(0.5)

    if not funds:
        print("没拉到任何基金")
        return

    print(f"\n计算配对 r5...")
    rows = []
    for _, p in portfolio.iterrows():
        t0 = p["rb_date"]
        row = {"rb_date": t0, "port_r5": p["port_r5"], "regime": p["regime"]}
        for code, (name, nav) in funds.items():
            r5 = fund_r5(nav, t0)
            row[f"{code}"] = r5
        rows.append(row)
    paired = pd.DataFrame(rows)
    paired.to_csv(OUT_DIR / "vs_top_funds.csv", index=False, encoding="utf-8-sig")

    # 配对分析
    print("\n" + "=" * 90)
    print("配对实验: 我们策略 vs Top 10 基金 (同买点, 同持有 5 天)")
    print("=" * 90)
    print(f"\n{'基金代码':<10} {'基金名称':<28} {'近2年%':>8} "
          f"{'平均超额':>10} {'胜率':>10} {'累计超额':>10}")
    print("-" * 90)

    fund_results = []
    total_excess = 0
    for code, (name, _) in funds.items():
        if code not in paired.columns: continue
        sub = paired.dropna(subset=[code])
        if len(sub) == 0: continue
        diff = sub["port_r5"] - sub[code]
        win = (diff > 0).sum()
        avg = diff.mean()
        cum = diff.sum()
        nav_2y = top[top["基金代码"] == code]["近2年"].iloc[0]
        # 截短名字
        short_name = name[:14] + ".." if len(name) > 14 else name
        print(f"{code:<10} {short_name:<25} {nav_2y:>+7.1f}% "
              f"{avg:>+9.3f}pp {win:>3}/{len(sub):<3} ({win/len(sub)*100:>4.1f}%) {cum:>+9.2f}pp")
        fund_results.append({"code":code, "name":name, "近2年":nav_2y,
                              "avg_excess":avg, "win_count":win, "n":len(sub),
                              "cum_excess":cum})

    # 总结
    print(f"\n=== 总结 ===")
    avg_avg = np.mean([f["avg_excess"] for f in fund_results])
    avg_win = np.mean([f["win_count"]/f["n"] for f in fund_results]) * 100
    n_funds_we_beat = sum(1 for f in fund_results if f["avg_excess"] > 0)
    print(f"我们对 {len(fund_results)} 只 top 基金的平均: 超额 {avg_avg:+.3f}pp, 胜率 {avg_win:.1f}%")
    print(f"我们跑赢 {n_funds_we_beat}/{len(fund_results)} 只基金")

    # 按 regime
    if any(f["avg_excess"] > 0 for f in fund_results):
        print(f"\n=== 跑赢的基金 ===")
        for f in sorted(fund_results, key=lambda x: -x["avg_excess"])[:5]:
            if f["avg_excess"] > 0:
                print(f"  ✓ {f['code']} {f['name'][:20]}: 超额 {f['avg_excess']:+.3f}pp, 胜率 {f['win_count']/f['n']*100:.1f}%")

    # 平均基金 vs 我们 (basket of all 10 funds)
    print(f"\n=== vs 'top 10 基金等权篮子' ===")
    fund_cols = [c for c in paired.columns if c.startswith("0") or c.startswith("1") or c.startswith("9")]
    valid_cols = [c for c in fund_cols if c in paired.columns]
    if valid_cols:
        paired["funds_avg"] = paired[valid_cols].mean(axis=1, skipna=True)
        sub = paired.dropna(subset=["funds_avg", "port_r5"])
        diff = sub["port_r5"] - sub["funds_avg"]
        win = (diff > 0).sum()
        print(f"  n={len(sub)}, 平均超额 {diff.mean():+.3f}pp, 胜率 {win}/{len(sub)} = {win/len(sub)*100:.1f}%")
        # t-test
        t_stat, p_val = stats.ttest_1samp(diff, 0)
        print(f"  t-stat {t_stat:.2f}, p-value {p_val:.4f}")

    Path(OUT_DIR / "vs_top_funds_summary.json").write_text(
        json.dumps(fund_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(f"\n输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
