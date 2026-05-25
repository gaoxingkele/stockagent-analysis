"""7 只关注股的概念基线评分 (任务 1, 用新 concept_merged 表).

对每只关注股, 取其 top N 最稀有概念, 算各概念的全市场 pump↑/pump↓ 均值,
跟个股自身评分对比.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.v12_scoring import V12Scorer

TARGETS = ["688507.SH", "301313.SZ", "688260.SH", "688234.SH",
             "688525.SH", "688249.SH", "688322.SH"]
TOP_N_CONCEPTS = 5  # 每股取 top N 最稀有概念


def main():
    scorer = V12Scorer.get(ROOT)
    date = scorer.list_available_dates()[-1]
    print(f"\n=== {date} 7 只关注股 + 概念基线 ===\n", flush=True)

    df = scorer.score_market(date)

    # 加载新合并概念表
    cm = pd.read_parquet(ROOT / "output/concept_local/concept_merged.parquet")
    cm = cm[["stock_code", "concept_name"]].drop_duplicates()
    print(f"  concept_merged: {len(cm):,} 关联, {cm['concept_name'].nunique():,} 概念", flush=True)

    # 每概念全市场 pump 均值
    cm_with_pump = cm.merge(df[["ts_code", "pump_score", "pump_down_score"]],
                              left_on="stock_code", right_on="ts_code", how="inner")
    concept_stats = cm_with_pump.groupby("concept_name").agg(
        cpt_pump_up=("pump_score", "mean"),
        cpt_pump_dn=("pump_down_score", "mean"),
        cpt_n=("stock_code", "count"),
    ).reset_index()
    print(f"  概念统计: {len(concept_stats):,} 概念", flush=True)

    # 输出
    OUT = ROOT / "output" / "specific_stocks_score"
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# 7 只关注股 + 概念基线 ({date})\n\n",
            "## 每股 Top 5 主要概念 (按概念成员数升序, 越稀有越特色)\n\n"]

    for ts in TARGETS:
        # 个股自身评分
        ind_row = df[df["ts_code"] == ts]
        if ind_row.empty: continue
        ind = ind_row.iloc[0]
        nm = ind.get("name", "")
        industry = ind.get("industry", "")
        p_up = ind["pump_score"]
        p_dn = ind["pump_down_score"]

        # 该股的概念
        my_concepts = cm[cm["stock_code"] == ts]["concept_name"].tolist()
        if not my_concepts:
            md.append(f"### {ts} {nm}: 无关联概念\n\n")
            continue

        # 取最稀有 top N
        my_stats = concept_stats[concept_stats["concept_name"].isin(my_concepts)]
        my_stats = my_stats.sort_values("cpt_n").head(TOP_N_CONCEPTS)

        md.append(f"### {ts} {nm} ({industry})\n\n")
        md.append(f"个股: pump↑ = **{p_up:.3f}**, pump↓ = **{p_dn:.3f}**\n\n")
        md.append(f"| 概念 | 成员数 | 概念↑均 | 概念↓均 | 个股↑ vs 概念 | 个股↓ vs 概念 |\n")
        md.append(f"|---|---|---|---|---|---|\n")
        for _, r in my_stats.iterrows():
            d_up = p_up - r["cpt_pump_up"]
            d_dn = p_dn - r["cpt_pump_dn"]
            up_emoji = "🟢" if d_up > 0.02 else ("🔴" if d_up < -0.02 else "≈")
            dn_emoji = "🔴" if d_dn > 0.02 else ("🟢" if d_dn < -0.02 else "≈")
            md.append(f"| {r['concept_name']} | {r['cpt_n']} | "
                       f"{r['cpt_pump_up']:.3f} | {r['cpt_pump_dn']:.3f} | "
                       f"{up_emoji} {d_up:+.3f} | {dn_emoji} {d_dn:+.3f} |\n")
        md.append(f"\n")

        # 概念加权综合
        avg_cpt_up = my_stats["cpt_pump_up"].mean()
        avg_cpt_dn = my_stats["cpt_pump_dn"].mean()
        md.append(f"**Top {TOP_N_CONCEPTS} 概念均值**: pump↑ {avg_cpt_up:.3f}, "
                   f"pump↓ {avg_cpt_dn:.3f}\n")
        md.append(f"- 个股 vs 概念加权: ↑ {p_up - avg_cpt_up:+.3f}, ↓ {p_dn - avg_cpt_dn:+.3f}\n\n")

    # 控制台打印 7 股快速对比
    print("=== 个股 vs 主要概念均值快速对比 ===\n")
    print(f"  {'code':12s} {'name':10s} {'股pump↑':8s} {'概念↑均':8s} {'↑差':7s} | "
           f"{'股pump↓':8s} {'概念↓均':8s} {'↓差':7s}")
    rows = []
    for ts in TARGETS:
        ind_row = df[df["ts_code"] == ts]
        if ind_row.empty: continue
        ind = ind_row.iloc[0]
        nm = str(ind.get("name", ""))[:10]
        p_up, p_dn = ind["pump_score"], ind["pump_down_score"]
        my_concepts = cm[cm["stock_code"] == ts]["concept_name"].tolist()
        if not my_concepts: continue
        my_stats = concept_stats[concept_stats["concept_name"].isin(my_concepts)]
        my_stats = my_stats.sort_values("cpt_n").head(TOP_N_CONCEPTS)
        if my_stats.empty: continue
        c_up = my_stats["cpt_pump_up"].mean()
        c_dn = my_stats["cpt_pump_dn"].mean()
        print(f"  {ts:12s} {nm:10s} {p_up:7.3f}  {c_up:7.3f}  {p_up - c_up:+6.3f}  | "
               f"{p_dn:7.3f}  {c_dn:7.3f}  {p_dn - c_dn:+6.3f}")
        rows.append({"ts_code": ts, "name": nm, "pump_up": p_up, "concept_up_avg": c_up,
                       "diff_up": p_up - c_up, "pump_down": p_dn,
                       "concept_dn_avg": c_dn, "diff_down": p_dn - c_dn})

    pd.DataFrame(rows).to_csv(OUT / f"{date}_concept_baseline.csv", index=False)
    Path(OUT / f"{date}_concept_baseline.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / f'{date}_concept_baseline.md'}", flush=True)


if __name__ == "__main__":
    main()
