"""指定股票池 V12 评分 + 行业/概念相对比较 (相对基线版).

vs 旧 score_specific_stocks.py:
  - 加 "行业内 pump↑/pump↓ pct rank" (跟同行业股比, 而非跟全市场比)
  - 加 "所属行业 pump 均值" + "差值" 列
  - 概念板块 (top 3 主要概念) 的 pump 均值
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.v12_scoring import V12Scorer

TARGET_STOCKS = [
    "688507.SH", "301313.SZ", "688260.SH", "688234.SH",
    "688525.SH", "688249.SH", "688322.SH",
]


def main():
    scorer = V12Scorer.get(ROOT)
    date = scorer.list_available_dates()[-1]
    print(f"\n=== {date} 指定股票 V12 评分 (相对基线版) ===\n", flush=True)

    df = scorer.score_market(date)

    # 行业层 pump 均值
    ind_stats = df.groupby("industry").agg(
        ind_pump_up_mean=("pump_score", "mean"),
        ind_pump_dn_mean=("pump_down_score", "mean"),
        ind_n=("ts_code", "count"),
    ).reset_index()
    df = df.merge(ind_stats, on="industry", how="left")

    # 行业内 pct rank
    df["pump_up_rank_in_ind"] = df.groupby("industry")["pump_score"].rank(pct=True, method="first")
    df["pump_dn_rank_in_ind"] = df.groupby("industry")["pump_down_score"].rank(pct=True, method="first")

    # 全市场 pct rank (作为对照)
    df["pump_up_rank_mkt"] = df["pump_score"].rank(pct=True, method="first")
    df["pump_dn_rank_mkt"] = df["pump_down_score"].rank(pct=True, method="first")

    # 概念层 (直接用 concept_detail 全表, 不依赖 summary)
    concept_p = ROOT / "output" / "tushare_cache" / "concept_detail.parquet"
    target_concept_info = {}   # ts_code -> [(concept_name, cpt_up_mean, cpt_dn_mean, cpt_n)]
    if concept_p.exists():
        cd = pd.read_parquet(concept_p)[["ts_code", "concept_name"]].drop_duplicates()
        # 每个概念的 pump 均值
        cd_with_pump = cd.merge(df[["ts_code", "pump_score", "pump_down_score"]],
                                  on="ts_code", how="inner")
        concept_stats = cd_with_pump.groupby("concept_name").agg(
            cpt_pump_up=("pump_score", "mean"),
            cpt_pump_dn=("pump_down_score", "mean"),
            cpt_n=("ts_code", "count"),
        ).reset_index()

        # 每只 target 股, 找其关联的所有概念
        for ts in TARGET_STOCKS:
            concepts_for_ts = cd[cd["ts_code"] == ts]["concept_name"].tolist()
            stats_for_ts = concept_stats[concept_stats["concept_name"].isin(concepts_for_ts)]
            # 按 cpt_n 升序 (越稀有概念越特色), 取 top 5
            stats_for_ts = stats_for_ts.sort_values("cpt_n").head(5)
            target_concept_info[ts] = stats_for_ts.to_dict("records")

    sub = df[df["ts_code"].isin(TARGET_STOCKS)].copy()
    sub["__order"] = sub["ts_code"].apply(
        lambda x: TARGET_STOCKS.index(x) if x in TARGET_STOCKS else 999)
    sub = sub.sort_values("__order")

    # 打印表 1: 评分 + 行业相对
    print("=== 表 1: V12 评分 + 行业相对基线 ===\n")
    print(f"  {'code':12s} {'name':10s} {'industry':12s} {'pump↑':6s} {'pump↓':6s} {'ind↑均':6s} "
           f"{'ind↓均':6s} {'行内↑rank':9s} {'行内↓rank':9s} {'市场↓rank':9s}")
    for _, r in sub.iterrows():
        nm = str(r.get("name", ""))[:10] if r.get("name") else ""
        ind = str(r.get("industry", ""))[:12]
        print(f"  {r['ts_code']:12s} {nm:10s} {ind:12s} "
               f"{r['pump_score']:.3f} {r['pump_down_score']:.3f} "
               f"{r.get('ind_pump_up_mean', 0):.3f} {r.get('ind_pump_dn_mean', 0):.3f} "
               f"{r.get('pump_up_rank_in_ind', 0):.2f}     "
               f"{r.get('pump_dn_rank_in_ind', 0):.2f}      "
               f"{r.get('pump_dn_rank_mkt', 0):.2f}")

    # 输出 MD
    OUT = ROOT / "output" / "specific_stocks_score"
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# 指定股票 V12 评分 + 行业/概念相对基线 ({date})\n\n",
            "## 行业相对基线 (跟所属行业其他股比)\n\n",
            "| 代码 | 名称 | 行业 | pump↑ | pump↓ | 行业↑均 | 行业↓均 | 行内↑rank | 行内↓rank | 市场↓rank |\n",
            "|---|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in sub.iterrows():
        nm = str(r.get("name", "") or "")
        ind = str(r.get("industry", "") or "")
        # rank 越高表示越好 (pump↑ 高 = 涨概率大), pump↓ rank 高 = 跌概率大 (危险)
        in_up = r.get("pump_up_rank_in_ind", 0)
        in_dn = r.get("pump_dn_rank_in_ind", 0)
        mkt_dn = r.get("pump_dn_rank_mkt", 0)
        in_up_flag = "🟢" if in_up >= 0.7 else ("🟡" if in_up >= 0.4 else "🔴")
        in_dn_flag = "🔴" if in_dn >= 0.7 else ("🟡" if in_dn >= 0.4 else "🟢")
        md.append(f"| {r['ts_code']} | {nm} | {ind} | "
                   f"{r['pump_score']:.3f} | {r['pump_down_score']:.3f} | "
                   f"{r.get('ind_pump_up_mean', 0):.3f} | {r.get('ind_pump_dn_mean', 0):.3f} | "
                   f"{in_up:.2f} {in_up_flag} | {in_dn:.2f} {in_dn_flag} | "
                   f"{mkt_dn:.2f} |\n")

    # 概念相对基线 (每股 top 5 稀有概念)
    if target_concept_info:
        md.append(f"\n## 关联概念基线 (每股取 top 5 最稀有概念)\n\n")
        for _, r in sub.iterrows():
            ts = r["ts_code"]
            nm = str(r.get("name", "") or "")
            info = target_concept_info.get(ts, [])
            if not info:
                md.append(f"### {ts} {nm}: 未找到关联概念\n\n")
                continue
            md.append(f"### {ts} {nm} (个股 pump↑={r['pump_score']:.3f}, "
                       f"pump↓={r['pump_down_score']:.3f})\n\n")
            md.append(f"| 概念 | 成员数 | 概念↑均 | 概念↓均 | 个股 vs 概念↑ | 个股 vs 概念↓ |\n")
            md.append(f"|---|---|---|---|---|---|\n")
            for item in info:
                cpt = item["concept_name"]
                cpt_up = item["cpt_pump_up"]
                cpt_dn = item["cpt_pump_dn"]
                cpt_n = item["cpt_n"]
                diff_up = r["pump_score"] - cpt_up
                diff_dn = r["pump_down_score"] - cpt_dn
                up_emoji = "🟢" if diff_up > 0.02 else ("🔴" if diff_up < -0.02 else "≈")
                dn_emoji = "🔴" if diff_dn > 0.02 else ("🟢" if diff_dn < -0.02 else "≈")
                md.append(f"| {cpt} | {cpt_n} | {cpt_up:.3f} | {cpt_dn:.3f} | "
                           f"{up_emoji} {diff_up:+.3f} | {dn_emoji} {diff_dn:+.3f} |\n")
            md.append(f"\n")

    Path(OUT / f"{date}_relative.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / f'{date}_relative.md'}")


if __name__ == "__main__":
    main()
