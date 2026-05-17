"""今日完整 1H 向下钻取 - 输出 strong_buy 清单.

流程:
  1. 加载当日 V12 推荐池 (Top 80 by r20_pred, 池 1-6)
  2. 对每只跑 1H 二次验证
  3. 按 dual_consensus 分级
  4. 输出 markdown 报告

用法:
  python scripts/drill_down_today.py [date]   默认 20260515
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.intraday_filter import drill_down_analyze


def main(date: str = "20260515"):
    t0 = time.time()
    # 1. 加载 V12 推荐池
    v12_csv = ROOT / "output" / "v7c_full_inference" / f"v7c_inference_{date}.csv"
    if not v12_csv.exists():
        # 备用: v12_portfolio
        v12_csv = ROOT / "output" / "v12_inference" / f"v12_portfolio_{date}.csv"
    df = pd.read_csv(v12_csv, dtype={"ts_code": str})
    print(f"加载 V12 推荐池: {len(df)} 股 from {v12_csv.name}")

    # 取池 1-6 (实战池, 排除暂禁的 7/8/9)
    if "pool" in df.columns:
        in_pool = df[df["pool"].isin([
            "pool1_v7c_main", "pool2_triple_consensus",
            "pool3_oversold_rebound", "pool4_bottom_breakout",
            "pool5_policy_wave", "pool6_strong_pullback",
        ])].copy()
    else:
        in_pool = df[df.get("v7c_recommend", False) == True].copy()

    # Top 80 by r20_pred
    in_pool = in_pool.sort_values("r20_pred", ascending=False).head(80).reset_index(drop=True)
    print(f"目标股票: {len(in_pool)} 只 (Top 80, 池 1-6)\n")

    # buy_r5_score 字典 (没有则用 buy_score 兜底)
    score_col = "buy_r5_score" if "buy_r5_score" in in_pool.columns else "buy_score"
    v12_scores = dict(zip(in_pool["ts_code"], in_pool[score_col]))

    # 2. 1H 钻取
    def cb(i, n, dt):
        print(f"  [{i}/{n}] {dt:.0f}s", flush=True)
    print("[1H 二次验证] 拉数据 + 计算信号 + 共识判断...")
    drilled = drill_down_analyze(
        in_pool["ts_code"].tolist(), end_date=date,
        v12_scores=v12_scores, lookback_days=10, progress_cb=cb,
    )

    # 3. merge 回 V12 信息
    in_pool_keep = in_pool[["ts_code", "name", "industry", "pool",
                             "buy_score", "sell_score", "r20_pred",
                             "policy_heat_score", "policy_theme"]].copy() \
                   if "name" in in_pool.columns \
                   else in_pool[["ts_code", "industry", "pool", "buy_score",
                                  "sell_score", "r20_pred", "policy_heat_score",
                                  "policy_theme"]].copy()
    merged = drilled.merge(in_pool_keep, on="ts_code", how="left")

    # 4. 按共识分类输出
    out_dir = ROOT / "output" / "drill_down"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f"drill_down_{date}.csv", index=False, encoding="utf-8-sig")

    # 统计
    consensus_counts = merged["dual_consensus"].value_counts().to_dict()
    print(f"\n=== {date} 钻取完成 (耗时 {time.time()-t0:.0f}s) ===")
    print(f"共识分布: {consensus_counts}")

    # Markdown 报告
    md = [f"# {date} V12 推荐池 1H 向下钻取报告\n"]
    md.append(f"覆盖 {len(merged)} 股 (Top 80 by r20_pred, 池 1-6), 耗时 {time.time()-t0:.0f}s\n")
    md.append(f"共识分布: {consensus_counts}\n")

    for cons, emoji in [("strong_buy", "⭐"), ("short_pulse", "🔥"),
                          ("neutral", "🟡"), ("wait_pullback", "⏸"),
                          ("strong_avoid", "❌")]:
        sub = merged[merged["dual_consensus"] == cons].sort_values("r20_pred", ascending=False)
        if len(sub) == 0: continue
        md.append(f"\n## {emoji} {cons} ({len(sub)} 只)\n")
        md.append("| 代码 | 中文名 | 行业 | 池 | V12 r20 | 1H 趋势 | 1H RSI | 1H 分 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            nm = str(r.get("name", ""))[:8]
            md.append(f"| {r['ts_code']} | {nm} | {str(r.get('industry','')[:8])} | "
                      f"{str(r.get('pool',''))[:20]} | {r.get('r20_pred',0):+.2f}% | "
                      f"{r.get('intraday_trend','?')} | {r.get('intraday_rsi14',0):.1f} | "
                      f"{r.get('intraday_score',0):+.2f} |")

    md_path = out_dir / f"drill_down_{date}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n输出: {md_path}")
    print(f"      {out_dir}/drill_down_{date}.csv")

    # 屏幕展示 strong_buy
    sb = merged[merged["dual_consensus"] == "strong_buy"].sort_values("r20_pred", ascending=False)
    if len(sb):
        print(f"\n=== ⭐ Strong Buy ({len(sb)} 只) - 今日双引擎共识 ===")
        print(f"{'代码':<11} {'中文名':<8} {'行业':<10} {'池':<22} {'V12 r20':>7} {'1H 分':>6} {'RSI':>5}")
        for _, r in sb.iterrows():
            nm = str(r.get('name',''))[:8]
            print(f"{r['ts_code']:<11} {nm:<8} {str(r.get('industry',''))[:10]:<10} "
                  f"{str(r.get('pool',''))[:22]:<22} {r.get('r20_pred',0):>+6.2f}% "
                  f"{r.get('intraday_score',0):>+6.2f} {r.get('intraday_rsi14',0):>5.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "20260515")
