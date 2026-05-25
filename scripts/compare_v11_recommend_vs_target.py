"""V11 实际推荐 vs 用户关注股 同表对比."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.v12_scoring import V12Scorer

V11_RECOMMEND = [
    "603217.SH", "300615.SZ", "688786.SH", "688175.SH",
    "600241.SH", "300610.SZ", "603787.SH",   # A 轨 7 股
    "301001.SZ", "300512.SZ",                # B 轨 2 股
]

USER_TARGETS = [
    "688507.SH", "301313.SZ", "688260.SH", "688234.SH",
    "688525.SH", "688249.SH", "688322.SH",
]


def main():
    scorer = V12Scorer.get(ROOT)
    date = scorer.list_available_dates()[-1]
    df = scorer.score_market(date)

    all_codes = V11_RECOMMEND + USER_TARGETS
    sub = df[df["ts_code"].isin(all_codes)].copy()
    if sub.empty:
        print("无匹配"); return

    # 标签
    def tag(code):
        if code in V11_RECOMMEND[:7]: return "V11 A 轨"
        if code in V11_RECOMMEND[7:]: return "V11 B 轨"
        return "用户关注"
    sub["tag"] = sub["ts_code"].apply(tag)

    sub["__order"] = sub["ts_code"].apply(
        lambda x: all_codes.index(x) if x in all_codes else 999)
    sub = sub.sort_values("__order")

    print(f"\n=== {date} V11 推荐 vs 用户关注股对比 ===\n")

    # md 输出
    OUT = ROOT / "output" / "specific_stocks_score"
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"# V11 推荐 vs 用户关注股 ({date})\n\n",
            "| 来源 | 代码 | 名称 | 行业 | buy | r20 | sell | pump↑ | pump↓ | V7c | 象限 | 行业mom |\n",
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    last_tag = None
    for _, r in sub.iterrows():
        tag_s = r["tag"]
        if tag_s != last_tag:
            if last_tag is not None:
                md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            last_tag = tag_s
        name = str(r.get("name", "") or "")
        ind = str(r.get("industry", "") or "")
        pump_d_marker = " ⚠️" if r.get("pump_down_score", 0) >= 0.6 else ""
        md.append(f"| {tag_s} | {r['ts_code']} | {name} | {ind} | "
                   f"{r.get('buy_score', 0):.1f} | "
                   f"{r.get('buy_r20_score', 0):.1f} | "
                   f"{r.get('sell_score', 0):.1f} | "
                   f"**{r.get('pump_score', 0):.3f}** | "
                   f"{r.get('pump_down_score', 0):.3f}{pump_d_marker} | "
                   f"{'✓' if r.get('v7c_recommend') else '✗'} | "
                   f"{r.get('quadrant', '')} | "
                   f"{r.get('industry_mom_60d_rank', 0):.2f} |\n")
    md.append(f"\n## 跨组对比\n\n")

    # 对比指标
    v11 = sub[sub["tag"].isin(["V11 A 轨", "V11 B 轨"])]
    user = sub[sub["tag"] == "用户关注"]

    md.append(f"### V11 推荐 ({len(v11)} 股) 均值\n\n")
    md.append(f"- pump↑ 均值: **{v11['pump_score'].mean():.3f}**\n")
    md.append(f"- pump↓ 均值: {v11['pump_down_score'].mean():.3f}\n")
    md.append(f"- buy 均值: {v11['buy_score'].mean():.1f}\n")
    md.append(f"- r20 均值: {v11['buy_r20_score'].mean():.1f}\n")
    md.append(f"- sell 均值: {v11['sell_score'].mean():.1f}\n")

    md.append(f"\n### 用户关注 ({len(user)} 股) 均值\n\n")
    md.append(f"- pump↑ 均值: **{user['pump_score'].mean():.3f}**\n")
    md.append(f"- pump↓ 均值: {user['pump_down_score'].mean():.3f}\n")
    md.append(f"- buy 均值: {user['buy_score'].mean():.1f}\n")
    md.append(f"- r20 均值: {user['buy_r20_score'].mean():.1f}\n")
    md.append(f"- sell 均值: {user['sell_score'].mean():.1f}\n")

    Path(OUT / f"compare_{date}.md").write_text("".join(md), encoding="utf-8")

    # console 简表 (用 Y/N 避免 GBK)
    print(f"  {'tag':10s} {'code':12s} {'name':10s} {'buy':5s} {'r20':5s} {'pump↑':6s} {'pump↓':6s} {'V7c':3s}")
    for _, r in sub.iterrows():
        nm = str(r.get("name", ""))[:10]
        v7c = "Y" if r.get("v7c_recommend") else "N"
        print(f"  {r['tag']:10s} {r['ts_code']:12s} {nm:10s} "
               f"{r.get('buy_score', 0):5.1f} {r.get('buy_r20_score', 0):5.1f} "
               f"{r.get('pump_score', 0):.3f} {r.get('pump_down_score', 0):.3f} {v7c}")

    print(f"\n输出: {OUT / f'compare_{date}.md'}")


if __name__ == "__main__":
    main()
