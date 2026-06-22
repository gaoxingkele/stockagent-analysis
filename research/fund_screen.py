# -*- coding: utf-8 -*-
"""FUND-1 — 高收益基金筛选 + 平滑/回撤特征。

AKShare 排行(近1年) × tushare fund_basic(类型/公司) → 股票/混合型高收益候选 → 去重A/C →
取 top N distinct → tushare fund_nav(复权) 算 近1年 收益/maxDD/Sharpe/Calmar → 区分
'平滑低回撤'(用户描述) vs '爆发高波动'。输出 research/cache/fund_screen.parquet。
"""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
RANK_P = ROOT / "output/tushare_cache/fund_rank_em.parquet"
NAV_CACHE = ROOT / "research/cache/fund_nav_cache.parquet"
OUT = ROOT / "research/cache/fund_screen.parquet"
TOP_N = 70
MIN_1Y = 50.0
BAD = ['债','货币','黄金','原油','商品','美元','纯债','短债','中短债','REIT','FOF','量化对冲']


def _pro():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
    import tushare as ts; ts.set_token(os.environ["TUSHARE_TOKEN"]); return ts.pro_api()


def base_name(n):
    s = str(n)
    for suf in ["A", "B", "C", "D", "E", "O", "(A类)", "(C类)"]:
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s.strip()


def maxdd(nav):
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1).min())


def main():
    pro = _pro()
    rank = pd.read_parquet(RANK_P)
    rank["近1年"] = pd.to_numeric(rank["近1年"], errors="coerce")
    rank = rank.dropna(subset=["近1年"])
    rank = rank[~rank["基金简称"].str.contains("|".join(BAD), na=False)]
    rank = rank[rank["近1年"] >= MIN_1Y]
    # fund_basic 类型+公司
    fb = pro.fund_basic(market="O", status="L")[["ts_code", "name", "fund_type", "management"]]
    fb["code6"] = fb["ts_code"].str[:6]
    rank["code6"] = rank["基金代码"].astype(str).str.zfill(6)
    m = rank.merge(fb, on="code6", how="left")
    # 股票/混合型
    m = m[m["fund_type"].fillna("").str.contains("股票|混合", regex=True)]
    # 去重 A/C: 同 base_name 留近1年最高
    m["base"] = m["基金简称"].map(base_name)
    m = m.sort_values("近1年", ascending=False).drop_duplicates("base", keep="first")
    cand = m.head(TOP_N).copy()
    print(f"[cand] 股票/混合型 近1年>={MIN_1Y}% 去重后取 top {len(cand)} (公司分布前6):", flush=True)
    print("  " + " / ".join(f"{k}:{v}" for k, v in cand["management"].fillna("?").value_counts().head(6).items()), flush=True)

    # 拉净值算回撤 (checkpoint)
    done = {}
    if NAV_CACHE.exists():
        prev = pd.read_parquet(NAV_CACHE)
        done = {c: g for c, g in prev.groupby("ts_code")}
    rows, navparts = [], list(done.values())
    for _, r in cand.iterrows():
        tsc = r["ts_code"]
        if pd.isna(tsc):
            continue
        if tsc in done:
            nav = done[tsc]
        else:
            try:
                nav = pro.fund_nav(ts_code=tsc, start_date="20250601", end_date="20260618")
                time.sleep(0.25)
            except Exception as e:
                print(f"  {tsc} nav ERR {repr(e)[:60]}", flush=True); continue
            if nav is None or len(nav) < 60:
                continue
            navparts.append(nav)
        nav = nav.sort_values("nav_date")
        col = "adj_nav" if nav["adj_nav"].notna().sum() > 60 else "unit_nav"
        v = nav[col].dropna().to_numpy(float)
        if len(v) < 120:
            continue
        v1y = v[-252:] if len(v) >= 252 else v
        ret = v1y[-1] / v1y[0] - 1
        dd = maxdd(v1y)
        dr = np.diff(v1y) / v1y[:-1]
        sharpe = float(np.mean(dr) / (np.std(dr) + 1e-9) * np.sqrt(252))
        rows.append({"ts_code": tsc, "name": r["基金简称"], "company": r["management"],
                     "type": r["fund_type"], "ret_1y_nav": round(ret * 100, 1),
                     "ret_1y_em": r["近1年"], "maxdd": round(dd * 100, 1),
                     "sharpe": round(sharpe, 2), "calmar": round(ret / (abs(dd) + 1e-9), 2),
                     "n_nav": len(v)})
    if navparts:
        pd.concat(navparts, ignore_index=True).drop_duplicates(["ts_code", "nav_date"]).to_parquet(NAV_CACHE, index=False)
    df = pd.DataFrame(rows).sort_values("calmar", ascending=False).reset_index(drop=True)
    df.to_parquet(OUT, index=False)

    print(f"\n=== 高收益基金特征 ({len(df)}只, 按Calmar排序) ===", flush=True)
    print(f"  {'代码':11s}{'名称':22s}{'公司':10s}{'近1年%':>7s}{'maxDD%':>7s}{'Sharpe':>7s}{'Calmar':>7s}", flush=True)
    for _, r in df.iterrows():
        print(f"  {r['ts_code']:11s}{str(r['name'])[:20]:22s}{str(r['company'])[:8]:10s}"
              f"{r['ret_1y_nav']:>7.0f}{r['maxdd']:>7.0f}{r['sharpe']:>7.2f}{r['calmar']:>7.1f}", flush=True)
    smooth = df[(df["ret_1y_nav"] >= 50) & (df["maxdd"] >= -25)]
    print(f"\n  平滑低回撤(近1年>=50% 且 maxDD<=25%): {len(smooth)}只", flush=True)
    print(f"  全批均值: 近1年{df['ret_1y_nav'].mean():.0f}% maxDD{df['maxdd'].mean():.0f}% Sharpe{df['sharpe'].mean():.2f}", flush=True)
    print(f"  公司分布: " + " / ".join(f"{k}:{v}" for k, v in df['company'].value_counts().head(8).items()), flush=True)


if __name__ == "__main__":
    main()
