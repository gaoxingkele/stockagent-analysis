"""任务 3: 概念轮动跟踪 + 排名前三概念选股.

1. 跨日 (近 20 日) 跟踪每概念的"热度"
2. 排序找出近 N 日热度最高的 Top 3 概念
3. 这 3 概念里 V12 评分最高的股 (V7c 通过 + pump↑ 高 + pump↓ 低)

输出: output/concept_pick/<date>_top3_concept_picks.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.v12_scoring import V12Scorer

# 热度评分参数
LOOKBACK_DAYS = 10   # 近 N 日 (热度衡量窗口)
TOP_K_CONCEPTS = 5   # 取前 K 个热概念 (用户要 3, 给 5 留余量)
TOP_K_STOCKS_PER_CONCEPT = 8  # 每概念取前 K 股


def main():
    scorer = V12Scorer.get(ROOT)
    date = scorer.list_available_dates()[-1]
    print(f"\n=== {date} 概念轮动 + Top 3 概念选股 ===\n", flush=True)

    # 加载概念每日热度
    concept_daily = pd.read_parquet(ROOT / "output/concept_heat/concept_daily.parquet")
    concept_daily["trade_date"] = concept_daily["trade_date"].astype(str)

    # 1. 找最近 LOOKBACK 日的概念热度均值排序
    print(f"[1] 计算近 {LOOKBACK_DAYS} 日概念热度排名 ...", flush=True)
    dates_sorted = sorted(concept_daily["trade_date"].unique())
    end_date = date
    recent_dates = [d for d in dates_sorted if d <= end_date][-LOOKBACK_DAYS:]
    print(f"  实际窗口: {recent_dates[0]} ~ {recent_dates[-1]} ({len(recent_dates)} 日)", flush=True)

    recent = concept_daily[concept_daily["trade_date"].isin(recent_dates)]
    # 概念热度综合分: ret_5d * 1.0 + ret_20d * 0.3 + strong_5d * 30
    # 用 ret + strong 的加权和
    recent_agg = recent.groupby("concept_name").agg(
        avg_ret_5d=("cpt_ret_5d", "mean"),
        avg_ret_20d=("cpt_ret_20d", "mean"),
        avg_winners=("cpt_winners_5d", "mean"),
        avg_strong=("cpt_strong_5d", "mean"),
        avg_n_members=("cpt_n_members", "mean"),
    ).reset_index()
    # 综合分 (核心)
    recent_agg["heat_score"] = (
        recent_agg["avg_ret_5d"] * 1.0 +
        recent_agg["avg_ret_20d"] * 0.3 +
        recent_agg["avg_strong"] * 20  # strong 占 20% 分数权重
    )
    # 排除"指数/宽基"类 (中证 500/300, 上证 50 等), 因这些是宏观非概念
    EXCLUDE_KW = ["中证", "上证", "深证", "创业板指", "沪深300", "宽基"]
    recent_agg["is_macro"] = recent_agg["concept_name"].apply(
        lambda x: any(kw in str(x) for kw in EXCLUDE_KW))
    recent_agg = recent_agg[~recent_agg["is_macro"]]
    # 还要排除成员数 < 5 (太稀有可能噪声)
    recent_agg = recent_agg[recent_agg["avg_n_members"] >= 5]
    # 按 heat_score 降序
    recent_agg = recent_agg.sort_values("heat_score", ascending=False)

    top_concepts = recent_agg.head(TOP_K_CONCEPTS)
    print(f"\n=== Top {TOP_K_CONCEPTS} 热门概念 (近 {LOOKBACK_DAYS} 日) ===", flush=True)
    print(f"  {'rank':5s} {'concept':25s} {'avg_ret_5d':10s} {'avg_strong':10s} "
           f"{'n_members':10s} {'heat':8s}")
    for i, (_, r) in enumerate(top_concepts.iterrows(), 1):
        print(f"  #{i:3d}  {r['concept_name']:25s} {r['avg_ret_5d']:+9.2f}%  "
               f"{r['avg_strong']*100:8.1f}%  {r['avg_n_members']:9.0f}   "
               f"{r['heat_score']:7.2f}")

    # 2. V12 评分全市场
    print(f"\n[2] V12 评分全市场 ...", flush=True)
    df = scorer.score_market(date)

    # 3. 加载概念关联
    cm = pd.read_parquet(ROOT / "output/concept_local/concept_merged.parquet")
    cm = cm[["stock_code", "concept_name"]].drop_duplicates()

    # 4. Top K 概念里选股
    print(f"\n=== Top {TOP_K_CONCEPTS} 概念内选股 (V12 评分) ===\n", flush=True)
    all_picks = []
    OUT = ROOT / "output" / "concept_pick"
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# {date} 概念轮动 + Top 概念选股\n\n",
            f"近 {LOOKBACK_DAYS} 日热度排名前 {TOP_K_CONCEPTS} 概念:\n\n",
            "| # | 概念 | 5日均涨 | 强势率 | 成员数 | 热度分 |\n|---|---|---|---|---|---|\n"]
    for i, (_, r) in enumerate(top_concepts.iterrows(), 1):
        md.append(f"| {i} | {r['concept_name']} | {r['avg_ret_5d']:+.2f}% | "
                   f"{r['avg_strong']*100:.1f}% | {int(r['avg_n_members'])} | "
                   f"{r['heat_score']:.2f} |\n")
    md.append(f"\n## 各概念 Top {TOP_K_STOCKS_PER_CONCEPT} 选股 (V7c 池内 pump↑ 排序)\n\n")

    for i, (_, r) in enumerate(top_concepts.iterrows(), 1):
        cpt_name = r["concept_name"]
        print(f"\n--- 概念 #{i}: {cpt_name} (热度 {r['heat_score']:.2f}) ---", flush=True)
        # 该概念关联的股
        cpt_stocks = cm[cm["concept_name"] == cpt_name]["stock_code"].tolist()
        # 跟当日全市场 df 合并
        cpt_df = df[df["ts_code"].isin(cpt_stocks)].copy()
        if cpt_df.empty:
            print(f"  无股票数据", flush=True)
            continue

        # 优先选 V7c=True 的, 再按 pump↑ - pump↓ 排序
        cpt_df["composite"] = cpt_df["pump_score"] - 0.5 * cpt_df["pump_down_score"]
        v7c_pool = cpt_df[cpt_df.get("v7c_recommend", False) == True]
        if len(v7c_pool) >= 3:
            top_picks = v7c_pool.nlargest(TOP_K_STOCKS_PER_CONCEPT, "composite")
            print(f"  从 V7c 池内选 (n={len(v7c_pool)})", flush=True)
            pool_tag = "V7c"
        else:
            # V7c 不够, 全概念股按 composite 选
            top_picks = cpt_df.nlargest(TOP_K_STOCKS_PER_CONCEPT, "composite")
            print(f"  V7c 池只 {len(v7c_pool)}, 用全概念股按 composite 选", flush=True)
            pool_tag = "全概念"

        md.append(f"### #{i} {cpt_name} (热度 {r['heat_score']:.2f}, 池: {pool_tag})\n\n")
        md.append("| 代码 | 名称 | 行业 | r20 | pump↑ | pump↓ | V7c | composite |\n|---|---|---|---|---|---|---|---|\n")
        print(f"  {'code':12s} {'name':10s} {'industry':12s} {'r20':5s} {'pump↑':7s} "
               f"{'pump↓':7s} {'V7c':3s}")
        for _, p in top_picks.iterrows():
            nm = str(p.get("name", ""))[:10]
            ind = str(p.get("industry", ""))[:12]
            v7c_s = "Y" if p.get("v7c_recommend") else "N"
            print(f"  {p['ts_code']:12s} {nm:10s} {ind:12s} "
                   f"{p.get('buy_r20_score', 0):5.1f} {p['pump_score']:.3f}  "
                   f"{p['pump_down_score']:.3f}   {v7c_s}")
            md.append(f"| {p['ts_code']} | {p.get('name', '')} | {p.get('industry', '')} | "
                       f"{p.get('buy_r20_score', 0):.1f} | "
                       f"{p['pump_score']:.3f} | {p['pump_down_score']:.3f} | "
                       f"{'✓' if p.get('v7c_recommend') else '✗'} | "
                       f"{p['composite']:.3f} |\n")
            all_picks.append({"concept": cpt_name, "ts_code": p["ts_code"],
                                "name": p.get("name", ""), "pump_up": p["pump_score"],
                                "pump_down": p["pump_down_score"],
                                "composite": p["composite"],
                                "v7c": bool(p.get("v7c_recommend"))})
        md.append("\n")

    out_md = OUT / f"{date}_top_concept_picks.md"
    out_md.write_text("".join(md), encoding="utf-8")
    pd.DataFrame(all_picks).to_csv(OUT / f"{date}_top_concept_picks.csv", index=False)
    print(f"\n输出: {out_md}", flush=True)


if __name__ == "__main__":
    main()
