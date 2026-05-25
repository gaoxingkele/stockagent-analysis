"""指定股票池的 V12 评分细节查询."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.v12_scoring import V12Scorer

# 用户指定股票
TARGET_STOCKS = [
    "688507.SH", "301313.SZ", "688260.SH", "688234.SH",
    "688525.SH", "688249.SH", "688322.SH",
]


def main():
    scorer = V12Scorer.get(ROOT)
    dates = scorer.list_available_dates()
    if not dates:
        print("!! factor_lab 无可用日期"); return
    date = dates[-1]
    print(f"\n=== {date} 指定股票 V12 评分细节 ===\n")

    df = scorer.score_market(date)

    sub = df[df["ts_code"].isin(TARGET_STOCKS)].copy()
    if sub.empty:
        print(f"!! 没有匹配股票 (检查代码格式)")
        print(f"  样本: {df['ts_code'].head(10).tolist()}")
        return

    cols_pump = ["ts_code", "name", "industry",
                   "buy_score", "buy_r5_score", "buy_r10_score", "buy_r20_score",
                   "sell_score",
                   "pump_score", "pump_down_score",
                   "v7c_recommend",
                   "quadrant",
                   "is_zombie",
                   "industry_mom_60d_rank"]

    avail = [c for c in cols_pump if c in sub.columns]
    sub = sub[avail].copy()
    # 按指定顺序排序
    sub["__order"] = sub["ts_code"].apply(
        lambda x: TARGET_STOCKS.index(x) if x in TARGET_STOCKS else 999)
    sub = sub.sort_values("__order").drop(columns="__order")

    # 打印表格
    print(f"\n{'代码':12s} {'名称':10s} {'行业':12s} | {'buy':5s} {'r5':5s} {'r10':5s} {'r20':5s} | "
           f"{'sell':5s} | {'pump↑':6s} {'pump↓':6s} | {'V7c':5s} {'象限':8s} {'僵尸':5s} {'行业mom':8s}")
    print(f"{'-'*12:12s} {'-'*10:10s} {'-'*12:12s} | {'-'*5:5s} {'-'*5:5s} {'-'*5:5s} {'-'*5:5s} | "
           f"{'-'*5:5s} | {'-'*6:6s} {'-'*6:6s} | {'-'*5:5s} {'-'*8:8s} {'-'*5:5s} {'-'*8:8s}")
    for _, r in sub.iterrows():
        name = str(r.get("name", ""))[:10] if r.get("name") else ""
        ind = str(r.get("industry", ""))[:12] if r.get("industry") else ""
        buy = f"{r.get('buy_score', 0):.1f}"
        r5 = f"{r.get('buy_r5_score', 0):.1f}"
        r10 = f"{r.get('buy_r10_score', 0):.1f}"
        r20 = f"{r.get('buy_r20_score', 0):.1f}"
        sell = f"{r.get('sell_score', 0):.1f}"
        pump_u = f"{r.get('pump_score', 0):.3f}"
        pump_d = f"{r.get('pump_down_score', 0):.3f}"
        v7c = "Y" if r.get("v7c_recommend") else "N"
        quad = str(r.get("quadrant", ""))[:8]
        zombie = "Y" if r.get("is_zombie") else "N"
        imom = f"{r.get('industry_mom_60d_rank', 0):.2f}" if r.get("industry_mom_60d_rank") else "-"
        print(f"{r['ts_code']:12s} {name:10s} {ind:12s} | {buy:>5s} {r5:>5s} {r10:>5s} {r20:>5s} | "
               f"{sell:>5s} | {pump_u:>6s} {pump_d:>6s} | {v7c:>5s} {quad:8s} {zombie:>5s} {imom:>8s}")

    # 输出 csv + md
    OUT = ROOT / "output" / "specific_stocks_score"
    OUT.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT / f"{date}.csv", index=False)

    # Markdown 表格
    md = [f"# 指定股票 V12 评分 ({date})\n\n",
            "| 代码 | 名称 | 行业 | buy | r5 | r10 | r20 | sell | pump↑ | pump↓ | V7c | 象限 | 僵尸 | 行业mom |\n",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in sub.iterrows():
        name = str(r.get("name", "") or "")
        ind = str(r.get("industry", "") or "")
        md.append(f"| {r['ts_code']} | {name} | {ind} | "
                   f"{r.get('buy_score', 0):.1f} | {r.get('buy_r5_score', 0):.1f} | "
                   f"{r.get('buy_r10_score', 0):.1f} | {r.get('buy_r20_score', 0):.1f} | "
                   f"{r.get('sell_score', 0):.1f} | "
                   f"**{r.get('pump_score', 0):.3f}** | {r.get('pump_down_score', 0):.3f} | "
                   f"{'✓' if r.get('v7c_recommend') else '✗'} | "
                   f"{r.get('quadrant', '')} | "
                   f"{'Y' if r.get('is_zombie') else 'N'} | "
                   f"{r.get('industry_mom_60d_rank', 0):.2f} |\n")

    md.append(f"\n## 字段说明\n\n")
    md.append(f"- **buy/r5/r10/r20**: 0-100 锚定评分 (V12 V7c 主要看 r20 top 5%)\n")
    md.append(f"- **sell**: 派发评分 (V7c 已屏蔽, 仅参考)\n")
    md.append(f"- **pump↑**: 启动子概率 (0-1, V11 池内排序信号, 高=好)\n")
    md.append(f"- **pump↓**: 跌启动子概率 (0-1, ≥0.60 硬过滤排除, 高=危险)\n")
    md.append(f"- **V7c**: 是否进入 V7c 主推荐池 (5 铁律全过)\n")
    md.append(f"- **象限**: 理想多/矛盾段/主流空/沉寂/中性区\n")
    md.append(f"- **僵尸**: MA60 走平 + 横盘 ≥90% (V7c 排除)\n")
    md.append(f"- **行业mom**: 行业 60 日 momentum 当日 pct rank (< 0.10 V7c 排除)\n")

    Path(OUT / f"{date}.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出:")
    print(f"  {OUT / f'{date}.csv'}")
    print(f"  {OUT / f'{date}.md'}")


if __name__ == "__main__":
    main()
