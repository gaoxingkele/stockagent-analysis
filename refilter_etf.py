"""重新筛选 ETF: 标准放宽 + 加 2026 Q1 表现维度。

用现有 nav 缓存 (output/etf_analysis/nav/), 不重新拉数据。

新标准 (符合任一即可):
  - 过去 3 年年化 ≥ 30%
  - 2026 Q1 (1-3月) 单季涨幅 ≥ 15%
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ETF_DIR = ROOT / "output" / "etf_analysis"
NAV_DIR = ETF_DIR / "nav"
HOLD_DIR = ETF_DIR / "holdings"

with open(ETF_DIR / "etf_list.json", encoding="utf-8") as f:
    etf_list = json.load(f)
print(f"ETF 总数: {len(etf_list)}")

YEAR_TARGET = 0.30   # 30%
Q1_TARGET = 0.15     # 15%
Q1_START = "20260101"
Q1_END = "20260331"


candidates = []
for etf in etf_list:
    code = etf["ts_code"]
    cache = NAV_DIR / f"{code}.json"
    if not cache.exists(): continue
    try:
        with open(cache, encoding="utf-8") as f:
            rows = json.load(f)
    except: continue
    if len(rows) < 100: continue

    valid = [r for r in rows if (r.get("adj_nav") or r.get("unit_nav"))]
    if len(valid) < 100: continue

    # 过去 3 年年化
    first_v = valid[0].get("adj_nav") or valid[0].get("unit_nav")
    last_v = valid[-1].get("adj_nav") or valid[-1].get("unit_nav")
    first_d = valid[0]["nav_date"]
    last_d = valid[-1]["nav_date"]
    d1 = (int(last_d[:4]) - int(first_d[:4])) + (int(last_d[4:6]) - int(first_d[4:6]))/12
    if d1 < 0.5: continue
    annual = (last_v / first_v) ** (1.0 / d1) - 1
    total_ret = last_v / first_v - 1

    # Q1 涨幅
    q1_start_v = None; q1_end_v = None
    for r in valid:
        nv = r.get("adj_nav") or r.get("unit_nav")
        if r["nav_date"] >= Q1_START and q1_start_v is None:
            q1_start_v = nv
        if r["nav_date"] <= Q1_END:
            q1_end_v = nv
    q1_ret = None
    if q1_start_v and q1_end_v and q1_start_v > 0:
        q1_ret = q1_end_v / q1_start_v - 1

    # 是否候选?
    is_candidate = False
    cand_type = []
    if annual >= YEAR_TARGET:
        is_candidate = True
        cand_type.append("3y_year")
    if q1_ret is not None and q1_ret >= Q1_TARGET:
        is_candidate = True
        cand_type.append("q1")

    if is_candidate:
        candidates.append({
            **etf,
            "first_date": first_d, "last_date": last_d,
            "years": round(d1, 2),
            "total_return_pct": round(total_ret * 100, 2),
            "annual_return_pct": round(annual * 100, 2),
            "q1_2026_return_pct": round(q1_ret * 100, 2) if q1_ret else None,
            "candidate_type": "+".join(cand_type),
        })

candidates.sort(key=lambda c: -c["annual_return_pct"])

print(f"\n候选 ETF: {len(candidates)} 只")
print(f"  - 仅 3 年年化 ≥{YEAR_TARGET*100:.0f}%: {sum(1 for c in candidates if c['candidate_type']=='3y_year')}")
print(f"  - 仅 Q1 涨幅 ≥{Q1_TARGET*100:.0f}%: {sum(1 for c in candidates if c['candidate_type']=='q1')}")
print(f"  - 双标准都达: {sum(1 for c in candidates if '+' in c['candidate_type'])}")

# 写入新 winners
WINNERS = ETF_DIR / "winners_relaxed.json"
WINNERS.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"写入 {WINNERS}")

# 简化展示
print(f"\nTop 30 (按年化收益排序):")
print(f"{'代码':12s} {'名称':<30s} {'年化':>8s} {'全期':>8s} {'Q1':>8s} {'类型':<10s}")
for c in candidates[:30]:
    q1_str = f"{c['q1_2026_return_pct']:+.1f}%" if c['q1_2026_return_pct'] is not None else "-"
    print(f"{c['ts_code']:12s} {c['name'][:30]:<30s} {c['annual_return_pct']:>+7.1f}% {c['total_return_pct']:>+7.1f}% {q1_str:>8s} {c['candidate_type']:<10s}")

# 按 Q1 排序前 20
print(f"\nTop 20 (按 Q1 2026 涨幅排序, 仅有 Q1 数据的):")
q1_sorted = sorted([c for c in candidates if c['q1_2026_return_pct'] is not None],
                    key=lambda c: -c['q1_2026_return_pct'])[:20]
for c in q1_sorted:
    print(f"{c['ts_code']:12s} {c['name'][:30]:<30s} 年化 {c['annual_return_pct']:>+6.1f}%  Q1 {c['q1_2026_return_pct']:>+6.1f}%")
