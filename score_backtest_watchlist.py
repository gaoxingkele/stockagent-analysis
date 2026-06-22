# -*- coding: utf-8 -*-
"""自定义清单逐日评分回测: 评分/ratio 是否与走势匹配。生产线只读。

对最近 N 个交易日 (factor_lab ext 中可评分的) 各跑 V12.31, 收集 49 股的 ratio/pump/r20 +
当日 pct_chg + 次日收益, 输出每股轨迹 + 匹配度 (横截面 ratio vs 次日收益 Spearman)。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.v12_scoring import V12Scorer

CODES = ["600160.SH","600549.SH","688257.SH","002171.SZ","603588.SH","600909.SH","300475.SZ",
"300566.SZ","688525.SH","600176.SH","000100.SZ","688498.SH","600206.SH","688766.SH","301099.SZ",
"300031.SZ","600522.SH","001696.SZ","600458.SH","601208.SH","300726.SZ","688233.SH","301536.SZ",
"688549.SH","300976.SZ","603225.SH","600498.SH","600459.SH","601211.SH","002842.SZ","600392.SH",
"300304.SZ","605589.SH","300666.SZ","300014.SZ","603271.SH","600378.SH","688809.SH","688378.SH",
"688002.SH","603045.SH","000725.SZ","002297.SZ","688388.SH","002378.SZ","920357.BJ","300975.SZ",
"920394.BJ","688146.SH"]
N_DAYS = 12   # 尝试最近 N 个交易日


def load_pct(date):
    p = ROOT / "output/tushare_cache/daily" / f"{date}.parquet"
    if not p.exists():
        return {}
    d = pd.read_parquet(p, columns=["ts_code", "pct_chg"])
    return dict(zip(d["ts_code"], d["pct_chg"]))


def main():
    dates = sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))[-N_DAYS:]
    s = V12Scorer.get(ROOT)
    pct_by_date = {d: load_pct(d) for d in dates}
    rows = []
    scored_dates = []
    for d in dates:
        try:
            df = s.score_market(d, cb=None)
        except Exception as e:
            print(f"  {d}: 跳过 ({str(e)[:50]})", flush=True)
            continue
        df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
        sub = df[df["ts_code"].isin(CODES)]
        for _, r in sub.iterrows():
            rows.append({"date": d, "ts_code": r["ts_code"],
                         "ratio": r["ratio"], "pump_up": r["pump_score"],
                         "pump_dn": r["pump_down_score"], "r20": r.get("buy_r20_score", np.nan),
                         "v7c": bool(r.get("v7c_recommend", False)),
                         "pct_chg": pct_by_date.get(d, {}).get(r["ts_code"], np.nan)})
        scored_dates.append(d)
        print(f"  {d}: 评分 {len(sub)}/{len(CODES)} 股", flush=True)
    if not rows:
        print("无可评分日期 (ext 缺历史). 需扩 factor_lab ext 窗。", flush=True)
        return
    bt = pd.DataFrame(rows)
    bt.to_csv(ROOT / "output/daily_pick/watchlist_score_bt.csv", index=False, encoding="utf-8-sig")

    # 次日收益
    sd = scored_dates
    nextret = {}
    for i, d in enumerate(sd[:-1]):
        nx = pct_by_date.get(sd[i + 1], {})
        for code in CODES:
            nextret[(d, code)] = nx.get(code, np.nan)
    bt["next_ret"] = bt.apply(lambda r: nextret.get((r["date"], r["ts_code"]), np.nan), axis=1)

    print(f"\n=== 评分回测 ({len(sd)}交易日: {sd[0]}~{sd[-1]}, {len(CODES)}股) ===\n", flush=True)
    # 1) 横截面匹配: 每日 Spearman(ratio_t, 当日/次日 pct_chg)
    print("逐日横截面 Spearman (49股内 ratio vs 收益):", flush=True)
    print(f"  {'日期':10s}{'ratio vs 当日':>14s}{'ratio vs 次日':>14s}  当日均涨", flush=True)
    same_ics, next_ics = [], []
    for d in sd:
        g = bt[bt["date"] == d].dropna(subset=["ratio"])
        same = stats.spearmanr(g["ratio"], g["pct_chg"])[0] if g["pct_chg"].notna().sum() > 5 else np.nan
        gn = g.dropna(subset=["next_ret"])
        nxt = stats.spearmanr(gn["ratio"], gn["next_ret"])[0] if len(gn) > 5 else np.nan
        if np.isfinite(same): same_ics.append(same)
        if np.isfinite(nxt): next_ics.append(nxt)
        avg = g["pct_chg"].mean()
        print(f"  {d:10s}{same:>+14.3f}{nxt:>+14.3f}  {avg:>+6.2f}%", flush=True)
    print(f"\n  均值: ratio↔当日={np.nanmean(same_ics):+.3f}  ratio↔次日={np.nanmean(next_ics):+.3f} "
          f"(>0=高ratio对应涨; 次日为预测力)", flush=True)

    # 2) 每股: ratio 轨迹 + 期间累计涨幅 + 个股 ratio-收益时序相关
    print(f"\n每股 ratio 轨迹 vs 走势:", flush=True)
    print(f"  {'代码':11s}{'名称':9s}{'ratio首':>7s}{'ratio末':>7s}{'ratio均':>7s}"
          f"{'期间涨%':>8s}{'ratio↔次日':>10s}", flush=True)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name"]].drop_duplicates("ts_code").set_index("ts_code")["name"].to_dict()
    summ = []
    for code in CODES:
        g = bt[bt["ts_code"] == code].sort_values("date")
        if len(g) < 2:
            continue
        cum = (1 + g["pct_chg"].fillna(0) / 100).prod() - 1
        ts_corr = (stats.spearmanr(g["ratio"], g["next_ret"])[0]
                   if g["next_ret"].notna().sum() > 2 else np.nan)
        summ.append({"code": code, "name": basic.get(code, "")[:8],
                     "r_first": g["ratio"].iloc[0], "r_last": g["ratio"].iloc[-1],
                     "r_mean": g["ratio"].mean(), "cum": cum * 100, "ts_corr": ts_corr})
    sdf = pd.DataFrame(summ).sort_values("r_mean", ascending=False)
    for _, r in sdf.iterrows():
        tc = f"{r['ts_corr']:+.2f}" if np.isfinite(r['ts_corr']) else "  n/a"
        print(f"  {r['code']:11s}{r['name']:9s}{r['r_first']:>7.1f}{r['r_last']:>7.1f}{r['r_mean']:>7.1f}"
              f"{r['cum']:>+8.1f}{tc:>10s}", flush=True)

    # 3) 总结: ratio 均值 vs 期间涨幅 的横截面相关 (高ratio股是否真涨更多)
    cs = stats.spearmanr(sdf["r_mean"], sdf["cum"])[0]
    print(f"\n  [横截面] ratio均值 vs 期间累计涨幅 Spearman = {cs:+.3f} "
          f"(>0=高ratio股期间涨更多=匹配; <0=背离)", flush=True)
    print(f"  CSV: output/daily_pick/watchlist_score_bt.csv", flush=True)


if __name__ == "__main__":
    main()
