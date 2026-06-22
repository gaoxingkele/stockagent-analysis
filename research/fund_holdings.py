# -*- coding: utf-8 -*-
"""FUND-2 — 高收益基金持仓抱团 + 概念 + 季度变化 + 重叠度分析。

读 fund_screen 批量 → tushare fund_portfolio 近5季度 top10 持仓 → 抱团(个股被N基金持有+权重)
+ 概念抱团(concept_detail映射) + 季度变化(新进/退出/概念轮动) + 两两Jaccard重叠(抱团强度)。
输出 research/FUND_CROWDING_FINDINGS.md。
"""
import os, sys, time, json
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
SCREEN = ROOT / "research/cache/fund_screen.parquet"
PORT_CACHE = ROOT / "research/cache/fund_portfolio_cache.parquet"
CONCEPT = ROOT / "output/tushare_cache/concept_detail.parquet"
BASIC = ROOT / "output/tushare_cache/stock_basic.parquet"
OUT_MD = ROOT / "research/FUND_CROWDING_FINDINGS.md"
QUARTERS = ["20250331", "20250630", "20250930", "20251231", "20260331"]


def _pro():
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
    import tushare as ts; ts.set_token(os.environ["TUSHARE_TOKEN"]); return ts.pro_api()


def fetch_portfolios(pro, codes):
    done = {}
    if PORT_CACHE.exists():
        prev = pd.read_parquet(PORT_CACHE); done = {c: g for c, g in prev.groupby("fund")}
    parts = list(done.values())
    for c in codes:
        if c in done:
            continue
        try:
            d = pro.fund_portfolio(ts_code=c, start_date="20250101", end_date="20260430")
            time.sleep(0.3)
        except Exception as e:
            print(f"  {c} ERR {repr(e)[:50]}", flush=True); continue
        if d is None or len(d) == 0:
            continue
        d = d.copy(); d["fund"] = c
        parts.append(d)
    allp = pd.concat(parts, ignore_index=True).drop_duplicates(["fund", "end_date", "symbol"])
    allp.to_parquet(PORT_CACHE, index=False)
    return allp


def main():
    pro = _pro()
    scr = pd.read_parquet(SCREEN)
    codes = scr["ts_code"].dropna().tolist()
    name_map = dict(zip(scr["ts_code"], scr["name"]))
    print(f"[port] 拉 {len(codes)} 只基金近5季度持仓 ...", flush=True)
    port = fetch_portfolios(pro, codes)
    port["end_date"] = port["end_date"].astype(str)
    port["symbol"] = port["symbol"].astype(str)
    stk = pd.read_parquet(BASIC)[["ts_code", "name", "industry"]].drop_duplicates("ts_code")
    snmap = dict(zip(stk["ts_code"], stk["name"])); simap = dict(zip(stk["ts_code"], stk["industry"]))
    concept = pd.read_parquet(CONCEPT)[["ts_code", "concept_name"]]

    out = ["# 高收益基金持仓抱团分析\n",
           f"\n样本: {scr.shape[0]} 只近1年高收益基金(近1年均{scr['ret_1y_nav'].mean():.0f}% maxDD均{scr['maxdd'].mean():.0f}% Sharpe均{scr['sharpe'].mean():.2f}), top10持仓近5季度。\n"]

    # ── 最新季度抱团 (个股) ──
    latest = "20260331"
    L = port[port["end_date"] == latest]
    nf = L["fund"].nunique()
    g = L.groupby("symbol").agg(n_funds=("fund", "nunique"), sum_w=("stk_mkv_ratio", "sum"),
                                avg_w=("stk_mkv_ratio", "mean")).reset_index()
    g["name"] = g["symbol"].map(snmap); g["ind"] = g["symbol"].map(simap)
    g = g.sort_values(["n_funds", "sum_w"], ascending=False)
    out.append(f"\n## 一、最新季度({latest}) 个股抱团 (披露{nf}只基金 top10)\n\n")
    out.append("| 股票 | 名称 | 行业 | 持有基金数 | 占比(只%) | 合计权重 |\n|---|---|---|---|---|---|\n")
    for _, r in g.head(25).iterrows():
        out.append(f"| {r['symbol']} | {snmap.get(r['symbol'],'')} | {simap.get(r['symbol'],'')} | "
                   f"{int(r['n_funds'])} | {r['n_funds']/nf*100:.0f}% | {r['sum_w']:.1f} |\n")

    # ── 概念抱团 ──
    Lc = L.merge(concept, left_on="symbol", right_on="ts_code", how="left")
    cg = Lc.dropna(subset=["concept_name"]).groupby("concept_name").agg(
        n_funds=("fund", "nunique"), n_stocks=("symbol", "nunique"), sum_w=("stk_mkv_ratio", "sum")).reset_index()
    cg = cg.sort_values(["n_funds", "sum_w"], ascending=False)
    out.append(f"\n## 二、概念抱团 (个股映射 concept_detail)\n\n")
    out.append("| 概念 | 持有基金数 | 涉及股数 | 合计权重 |\n|---|---|---|---|\n")
    for _, r in cg.head(20).iterrows():
        out.append(f"| {r['concept_name']} | {int(r['n_funds'])} | {int(r['n_stocks'])} | {r['sum_w']:.0f} |\n")

    # ── 集中度 + 重叠度(Jaccard) ──
    holdings = {f: set(gg["symbol"]) for f, gg in L.groupby("fund")}
    jac = []
    for a, b in combinations(holdings, 2):
        u = holdings[a] | holdings[b]
        if u:
            jac.append(len(holdings[a] & holdings[b]) / len(u))
    mean_jac = float(np.mean(jac)) if jac else 0
    avg_hold = float(np.mean([len(s) for s in holdings.values()]))
    # 平均每基金 top10 集中度
    conc = L.groupby("fund")["stk_mkv_ratio"].sum().mean()
    out.append(f"\n## 三、抱团强度 + 集中度\n\n")
    out.append(f"- 两两持仓 Jaccard 重叠均值: **{mean_jac:.3f}** (越高=抱团越紧; 随机~0.02)\n")
    out.append(f"- 平均每基金 top10 披露 {avg_hold:.1f} 股, top10 合计权重均值 {conc:.0f}%\n")
    # 被≥半数基金持有的核心抱团股
    core = g[g["n_funds"] >= nf * 0.3]
    out.append(f"- 被 ≥30% 基金({int(nf*0.3)}只)持有的核心抱团股: {len(core)} 只 → "
               f"{', '.join(snmap.get(s,s) for s in core['symbol'].head(12))}\n")

    # ── 季度变化: 概念权重轮动 + 个股新进/退出 ──
    out.append(f"\n## 四、季度变化 (抱团演进)\n\n")
    out.append("### 概念抱团权重 (各季度 持有基金数 top概念)\n\n")
    for q in QUARTERS:
        Q = port[port["end_date"] == q]
        if Q.empty: continue
        Qc = Q.merge(concept, left_on="symbol", right_on="ts_code", how="left").dropna(subset=["concept_name"])
        top = Qc.groupby("concept_name")["fund"].nunique().sort_values(ascending=False).head(5)
        out.append(f"- **{q}** ({Q['fund'].nunique()}基金): " +
                   " / ".join(f"{c}({n})" for c, n in top.items()) + "\n")
    # 最新两季 个股抱团变化
    if len([q for q in QUARTERS if not port[port["end_date"]==q].empty]) >= 2:
        qs = [q for q in QUARTERS if not port[port["end_date"]==q].empty]
        q_prev, q_now = qs[-2], qs[-1]
        prev_n = port[port["end_date"]==q_prev].groupby("symbol")["fund"].nunique()
        now_n = port[port["end_date"]==q_now].groupby("symbol")["fund"].nunique()
        delta = (now_n - prev_n.reindex(now_n.index).fillna(0)).sort_values(ascending=False)
        out.append(f"\n### {q_prev}→{q_now} 抱团增持最多(基金数增加)个股:\n\n")
        for s in delta.head(12).index:
            out.append(f"- {snmap.get(s,s)} ({s}): +{int(delta[s])} 基金 (→{int(now_n[s])}只持有)\n")
        gone = (prev_n.reindex(now_n.index.union(prev_n.index)).fillna(0) -
                now_n.reindex(now_n.index.union(prev_n.index)).fillna(0)).sort_values(ascending=False)
        out.append(f"\n### {q_prev}→{q_now} 抱团退出最多(基金数减少)个股:\n\n")
        for s in gone.head(10).index:
            if gone[s] > 0:
                out.append(f"- {snmap.get(s,s)} ({s}): -{int(gone[s])} 基金\n")

    OUT_MD.write_text("".join(out), encoding="utf-8")
    # 控制台摘要
    print("\n=== 抱团摘要 ===", flush=True)
    print(f"  披露基金 {nf} 只 | 两两Jaccard均值 {mean_jac:.3f} | 核心抱团股(≥30%基金) {len(core)}只", flush=True)
    print(f"  个股抱团 top10:", flush=True)
    for _, r in g.head(10).iterrows():
        print(f"    {snmap.get(r['symbol'],''):8s}{r['symbol']:11s}{simap.get(r['symbol'],''):8s} "
              f"{int(r['n_funds'])}只({r['n_funds']/nf*100:.0f}%) 权重{r['sum_w']:.0f}", flush=True)
    print(f"  概念抱团 top8:", flush=True)
    for _, r in cg.head(8).iterrows():
        print(f"    {r['concept_name'][:16]:18s} {int(r['n_funds'])}基金 {int(r['n_stocks'])}股", flush=True)
    print(f"\n[out] -> {OUT_MD.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
