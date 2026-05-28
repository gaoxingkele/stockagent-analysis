"""0526 全量 V12 评分 + Top 20 推荐 (V7c 主推 + pump_up 排序)."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.v12_scoring import V12Scorer

DATE = "20260527"
TOP_N = 20

OUT = ROOT / "output" / "daily_pick"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    s = V12Scorer.get(ROOT)
    print(f"\n=== V12 {DATE} 全市场推理 ===\n", flush=True)
    df = s.score_market(DATE,
        cb=lambda phase, pct, msg, data: print(f"  [{pct:3d}%] {msg}", flush=True))

    # 补 name (从 stock_basic)
    basic_p = ROOT / "output/tushare_cache/stock_basic.parquet"
    if basic_p.exists():
        basic = pd.read_parquet(basic_p)[["ts_code", "name", "industry"]].drop_duplicates("ts_code")
        if "name" not in df.columns:
            df = df.merge(basic[["ts_code", "name"]], on="ts_code", how="left")

    # V7c 主推 + pump_up 排序
    main = df[df["v7c_recommend"] == True].copy()
    print(f"\n  V7c 主推 (5 铁律通过): {len(main)} 股", flush=True)

    # 2026-05-27: ratio 排序 (实战 +0.16pp/月 优于 composite, 灾难月 +2.30pp)
    main["ratio"] = main["pump_score"] / (main["pump_down_score"] + 0.01)
    main["composite"] = main["pump_score"] - 0.5 * main["pump_down_score"]  # 保留参考
    main_sorted = main.sort_values("ratio", ascending=False)

    # 行业分散: 单行业 ≤ 4 只 (Top 20 内)
    picks = []
    ind_count = {}
    for _, row in main_sorted.iterrows():
        ind = str(row.get("industry") or "unknown")
        if ind_count.get(ind, 0) >= 4:
            continue
        ind_count[ind] = ind_count.get(ind, 0) + 1
        picks.append(row)
        if len(picks) >= TOP_N:
            break

    top = pd.DataFrame(picks)

    print(f"\n=== {DATE} V12 Top {TOP_N} 推荐 (V7c 主推 + ratio 排序 + 行业 cap 4) ===\n",
           flush=True)
    print(f"  {'#':3s} {'代码':12s} {'名称':12s} {'行业':12s} {'r20':6s} "
           f"{'pump↑':7s} {'pump↓':7s} {'↑/↓':>7s} {'past_r5':>7s}", flush=True)
    for i, (_, p) in enumerate(top.iterrows(), 1):
        nm = str(p.get("name", "") or "")[:12]
        ind = str(p.get("industry", "") or "")[:12]
        print(f"  {i:>2d}. {p['ts_code']:12s} {nm:12s} {ind:12s} "
               f"{p.get('buy_r20_score', 0):>5.1f}  "
               f"{p['pump_score']:.3f}   {p['pump_down_score']:.3f}   "
               f"{p['ratio']:>5.1f}x  "
               f"{p.get('past_r5', 0)*100:+5.1f}%", flush=True)

    # 行业分布
    print(f"\n  行业分布:", flush=True)
    for ind, cnt in sorted(ind_count.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    {ind}: {cnt}", flush=True)

    # 保存
    cols = ["ts_code", "name", "industry", "buy_r20_score", "buy_r10_score",
              "pump_score", "pump_down_score", "past_r5", "composite",
              "v7c_recommend", "policy_heat_score", "policy_theme"]
    cols = [c for c in cols if c in top.columns]
    top[cols].to_csv(OUT / f"top20_{DATE}.csv", index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {OUT / f'top20_{DATE}.csv'}", flush=True)

    # markdown
    md = [f"# {DATE} V12 Top {TOP_N} 推荐\n\n",
           f"V7c 主推 (5 铁律) → composite (pump↑ - 0.5*pump↓) 排序 → 行业 cap 4\n\n",
           f"V7c 池总: {len(main)} 股\n\n",
           "| # | 代码 | 名称 | 行业 | r20 分 | pump↑ | pump↓ | past_r5 | 综合 |\n",
           "|---|------|------|------|--------|-------|-------|---------|------|\n"]
    for i, (_, p) in enumerate(top.iterrows(), 1):
        md.append(f"| {i} | {p['ts_code']} | {p.get('name','')} | "
                   f"{p.get('industry','')} | {p.get('buy_r20_score', 0):.1f} | "
                   f"{p['pump_score']:.3f} | {p['pump_down_score']:.3f} | "
                   f"{p.get('past_r5', 0)*100:+.1f}% | "
                   f"{p['composite']:.3f} |\n")
    md.append(f"\n## 行业分布\n\n")
    for ind, cnt in sorted(ind_count.items(), key=lambda x: -x[1]):
        if cnt > 0:
            md.append(f"- {ind}: {cnt}\n")
    (OUT / f"top20_{DATE}.md").write_text("".join(md), encoding="utf-8")
    print(f"  MD: {OUT / f'top20_{DATE}.md'}", flush=True)


if __name__ == "__main__":
    main()
