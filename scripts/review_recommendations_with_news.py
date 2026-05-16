"""V12 推荐池 LLM 个股新闻复审 (Sprint 3 集成).

对当日 V12 主推 + 三引擎共识跑个股新闻 LLM, 补 V12 量化看不到的
基本面/治理风险 (减值/减持/诉讼/业绩下滑).

输入: output/v7c_full_inference/v7c_inference_{date}.csv
输出:
  - output/stock_news_llm/{date}/news_review.csv (每股一行)
  - output/stock_news_llm/{date}/news_review_summary.md (汇总报告)

使用:
  python scripts/review_recommendations_with_news.py 20260515
"""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env.cloubic"); load_dotenv(ROOT / ".env")

from stockagent_analysis.stock_news_llm import StockNewsAnalyzer


def main(date: str = "20260515"):
    csv = ROOT / "output" / "v7c_full_inference" / f"v7c_inference_{date}.csv"
    if not csv.exists():
        print(f"❌ 缺 {csv}")
        sys.exit(1)
    df = pd.read_csv(csv, dtype={"ts_code": str})

    # 选 V7c 主推 (可选: 加上池 4 底部突破 + 池 2 三引擎共识)
    if "pool" in df.columns:
        targets = df[df["pool"].isin([
            "pool1_v7c_main", "pool2_triple_consensus",
            "pool4_bottom_breakout", "pool6_strong_pullback",
        ])].copy()
    elif "v7c_recommend" in df.columns:
        targets = df[df["v7c_recommend"] == True].copy()
    else:
        print("❌ csv 缺 pool 或 v7c_recommend 列"); sys.exit(1)

    # 按 r20_pred 取 Top 80 (避免成本爆炸)
    targets = targets.sort_values("r20_pred", ascending=False).head(80).reset_index(drop=True)
    print(f"目标股票: {len(targets)} 只 (Top 80 by r20_pred)")

    analyzer = StockNewsAnalyzer(os.environ["CLOUBIC_API_KEY"])
    out_dir = ROOT / "output" / "stock_news_llm" / date
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    total_cost = 0.0
    for i, row in targets.iterrows():
        ts_code = row["ts_code"]
        symbol = ts_code.split(".")[0]  # 6 位代码
        if not symbol.isdigit():
            continue
        try:
            r = analyzer.analyze(symbol, lookback_days=30)
        except Exception as e:
            r = {"symbol": symbol, "error": str(e)[:80]}
        r["ts_code"] = ts_code
        r["name"] = row.get("name", "")
        r["pool"] = row.get("pool", "")
        r["buy_score"] = row.get("buy_score")
        r["r20_pred"] = row.get("r20_pred")
        r["policy_heat_score"] = row.get("policy_heat_score", 0)
        cost = r.get("llm_cost_usd", 0)
        total_cost += cost
        results.append(r)
        if (i+1) % 10 == 0 or i == len(targets) - 1:
            print(f"  [{i+1}/{len(targets)}] {symbol} {row.get('name','?')[:6]} "
                  f"sent={r.get('sentiment_score','?')} catalyst={r.get('catalyst_type','?')} "
                  f"${total_cost:.3f} ({time.time()-t0:.0f}s)")

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_dir / "news_review.csv", index=False, encoding="utf-8-sig")

    # Markdown 汇总
    md_lines = [f"# V12 推荐池 LLM 新闻复审 ({date})\n"]
    md_lines.append(f"覆盖 {len(out_df)} 股, LLM 总成本 \${total_cost:.3f}, 耗时 {time.time()-t0:.0f}s\n")

    # 高风险标记 (情感 < -0.3 或 催化为负)
    NEG_CATALYSTS = {"高管减持", "诉讼风险", "业绩下滑"}
    high_risk = out_df[
        (out_df["sentiment_score"].fillna(0) < -0.3) |
        (out_df["catalyst_type"].fillna("").isin(NEG_CATALYSTS))
    ].sort_values("sentiment_score").reset_index(drop=True)
    md_lines.append(f"\n## ⚠️ 高风险标记 ({len(high_risk)} 只 V12 推荐但 LLM 警告)\n")
    if len(high_risk):
        md_lines.append("| 代码 | 中文名 | V12 池 | V12 r20 | LLM 情感 | 催化 | 关键事件 |")
        md_lines.append("|---|---|---|---|---|---|---|")
        for _, r in high_risk.iterrows():
            ev = "; ".join(r.get("key_events", []) or [])[:80] if r.get("key_events") else ""
            md_lines.append(f"| {r['ts_code']} | {r.get('name','')[:6]} | {r.get('pool','')[:18]} | "
                            f"{r.get('r20_pred',0):+.2f}% | {r.get('sentiment_score',0):+.2f} | "
                            f"{r.get('catalyst_type','')} | {ev} |")

    # 强多头催化 (业绩预增/中标/政策受益 + sentiment > 0.3)
    POS_CATALYSTS = {"业绩预增", "中标订单", "回购增持", "新业务", "并购重组", "政策受益"}
    pos = out_df[
        (out_df["sentiment_score"].fillna(0) > 0.3) &
        (out_df["catalyst_type"].fillna("").isin(POS_CATALYSTS))
    ].sort_values("sentiment_score", ascending=False)
    md_lines.append(f"\n\n## ⭐ 强多头催化 ({len(pos)} 只 LLM 加分推荐)\n")
    if len(pos):
        md_lines.append("| 代码 | 中文名 | V12 池 | LLM 情感 | 催化 | 关键事件 |")
        md_lines.append("|---|---|---|---|---|---|")
        for _, r in pos.iterrows():
            ev = "; ".join(r.get("key_events", []) or [])[:80] if r.get("key_events") else ""
            md_lines.append(f"| {r['ts_code']} | {r.get('name','')[:6]} | {r.get('pool','')[:18]} | "
                            f"{r.get('sentiment_score',0):+.2f} | {r.get('catalyst_type','')} | {ev} |")

    md_out = out_dir / "news_review_summary.md"
    md_out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n输出: {out_dir}/news_review.csv + news_review_summary.md")
    print(f"总成本: ${total_cost:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "20260515")
