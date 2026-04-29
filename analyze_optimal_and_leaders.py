"""(1) 行业内 (市值 × PE) 最优胜率组合 — D+10 / D+40
(2) 跨行业龙头股 (每行业市值 top 3) 的胜率分布

输出: output/backtest_factors_2026_04/report_optimal_leaders.md
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path("output/backtest_factors_2026_04")
OUT = ROOT / "report_optimal_leaders.md"

with open(ROOT / "universe.json", encoding="utf-8") as f:
    uni = json.load(f)
ts_to_ind = {s["ts_code"]: (s.get("industry") or "未分类") for s in uni["stocks"]}
ts_to_name = {s["ts_code"]: s.get("name", "") for s in uni["stocks"]}
ind_count = Counter(ts_to_ind.values())
MAJOR_INDS = sorted([k for k, n in ind_count.items() if n >= 40], key=lambda k: -ind_count[k])

print("加载 samples...")
samples = []
for gf in sorted((ROOT / "group_results").glob("group_*.jsonl")):
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
                s["industry"] = ts_to_ind.get(s["ts_code"], "未分类")
                samples.append(s)
            except Exception:
                pass
print(f"总样本: {len(samples):,}")

# ─── 桶 ───
def mv_bucket(mv):
    if mv is None or mv <= 0: return None
    if mv < 500000: return "20-50亿"
    if mv < 1000000: return "50-100亿"
    if mv < 3000000: return "100-300亿"
    if mv < 10000000: return "300-1000亿"
    return "1000亿+"

def pe_bucket(pe):
    if pe is None: return None
    if pe < 0: return "亏损"
    if pe < 15: return "0-15"
    if pe < 30: return "15-30"
    if pe < 50: return "30-50"
    if pe < 100: return "50-100"
    return "100+"

MV_ORDER = ["20-50亿", "50-100亿", "100-300亿", "300-1000亿", "1000亿+"]
PE_ORDER = ["亏损", "0-15", "15-30", "30-50", "50-100", "100+"]
HOLD = [10, 40]   # 用户关心的两个持有期


def winrate(samples, h):
    rs = [s.get(f"r{h}") for s in samples if s.get(f"r{h}") is not None]
    if not rs: return None
    return float(np.mean([1 if r > 0 else 0 for r in rs])) * 100

def avg_ret(samples, h):
    rs = [s.get(f"r{h}") for s in samples if s.get(f"r{h}") is not None]
    if not rs: return None
    return float(np.mean(rs))


# ═══════════════════════════════════════════════════════════════
# Part 1: 行业内 (市值 × PE) 最优胜率
# ═══════════════════════════════════════════════════════════════
lines = []
lines.append("# 行业内 (市值×PE) 最优胜率 + 跨行业龙头分析\n\n")
lines.append(f"> 样本: {len(samples):,}, 主要行业 (≥40 只): {len(MAJOR_INDS)}\n\n")
lines.append("---\n\n")

lines.append("## 一、各行业内 (市值 × PE) 二维最优组合\n\n")
lines.append("> D+10 短期 / D+40 中期, 找出每行业胜率最高的 (市值, PE) 组合\n")
lines.append("> 最低样本要求: 200 (确保统计意义)\n\n")

# 主表: 每行业 D+10 / D+40 最优组合
lines.append("### 1.1 速览表 (每行业的最优组合)\n\n")
lines.append("| 行业 | D+10 最优 (市值/PE) | D+10 胜率 | D+10 涨幅 | n | D+40 最优 (市值/PE) | D+40 胜率 | D+40 涨幅 | n |\n")
lines.append("|---|---|---|---|---|---|---|---|---|\n")

for ind in MAJOR_INDS:
    bucket = [s for s in samples if s["industry"] == ind]
    if len(bucket) < 200: continue
    line = f"| {ind} |"
    for h in HOLD:
        # 找最优 (mv, pe)
        best = None
        for mv in MV_ORDER:
            for pe in PE_ORDER:
                sub = [s for s in bucket
                       if mv_bucket(s.get("total_mv")) == mv
                       and pe_bucket(s.get("pe")) == pe]
                if len(sub) < 200: continue
                wr = winrate(sub, h)
                ret = avg_ret(sub, h)
                if wr is None: continue
                if best is None or wr > best["wr"]:
                    best = {"mv": mv, "pe": pe, "wr": wr, "ret": ret, "n": len(sub)}
        if best:
            line += f" {best['mv']} / {best['pe']} | **{best['wr']:.1f}%** | {best['ret']:+.2f}% | {best['n']:,} |"
        else:
            line += " - | - | - | - |"
    lines.append(line + "\n")
lines.append("\n")

# 详表: 每行业的 (市值 × PE) 二维矩阵 (D+40 胜率)
lines.append("### 1.2 各行业 D+40 胜率二维矩阵\n\n")
lines.append("> 单元格 = 该 (市值, PE) 桶的胜率, 加粗 = 该行业内最高 (n≥200)\n\n")

for ind in MAJOR_INDS[:20]:  # 只展示样本数最多的 20 个行业
    bucket = [s for s in samples if s["industry"] == ind]
    if len(bucket) < 500: continue

    lines.append(f"#### {ind} (样本 {len(bucket):,})\n\n")
    lines.append("| 市值 \\ PE |" + "".join(f" {pe} |" for pe in PE_ORDER) + "\n")
    lines.append("|---|" + "---|" * len(PE_ORDER) + "\n")

    # 先算所有 (mv, pe) 桶的胜率, 找最大值
    matrix = {}
    max_wr = -1
    max_key = None
    for mv in MV_ORDER:
        for pe in PE_ORDER:
            sub = [s for s in bucket
                   if mv_bucket(s.get("total_mv")) == mv
                   and pe_bucket(s.get("pe")) == pe]
            if len(sub) < 200:
                matrix[(mv, pe)] = (None, len(sub))
                continue
            wr = winrate(sub, 40)
            matrix[(mv, pe)] = (wr, len(sub))
            if wr is not None and wr > max_wr:
                max_wr = wr; max_key = (mv, pe)

    for mv in MV_ORDER:
        line = f"| **{mv}** |"
        for pe in PE_ORDER:
            wr, n = matrix.get((mv, pe), (None, 0))
            if wr is None:
                line += f" - ({n}) |" if n > 0 else " - |"
            elif (mv, pe) == max_key:
                line += f" **{wr:.1f}% (n={n:,})** |"
            else:
                line += f" {wr:.1f}% (n={n:,}) |"
        lines.append(line + "\n")
    lines.append("\n")

# ═══════════════════════════════════════════════════════════════
# Part 2: 跨行业龙头股分析
# ═══════════════════════════════════════════════════════════════
lines.append("---\n\n")
lines.append("## 二、跨行业龙头股分析\n\n")
lines.append("> 龙头定义: **每行业内按平均市值降序的 Top 3** (市场占有率代理指标)\n")
lines.append("> 技术领先无统一量化口径, 此处不区分\n\n")

# 计算每只股票的平均市值
print("计算每股平均市值, 提取龙头...")
stock_mvs = defaultdict(list)
for s in samples:
    mv = s.get("total_mv")
    if mv:
        stock_mvs[s["ts_code"]].append(mv)
stock_avg_mv = {ts: float(np.mean(mvs)) for ts, mvs in stock_mvs.items()}

# 每行业 Top 3
leaders = []  # list of (ts_code, name, industry, avg_mv, rank_in_ind)
ind_to_stocks = defaultdict(list)
for ts, mv in stock_avg_mv.items():
    ind = ts_to_ind.get(ts)
    if ind in MAJOR_INDS:
        ind_to_stocks[ind].append((ts, mv))
for ind, stocks in ind_to_stocks.items():
    stocks.sort(key=lambda x: -x[1])
    for rank, (ts, mv) in enumerate(stocks[:3]):
        leaders.append({
            "ts_code": ts, "name": ts_to_name.get(ts, ""),
            "industry": ind, "avg_mv": mv, "rank": rank + 1,
        })

leader_codes = {l["ts_code"] for l in leaders}
leader_samples = [s for s in samples if s["ts_code"] in leader_codes]
print(f"龙头股: {len(leader_codes)}, 龙头样本: {len(leader_samples):,}")

# 2.1 龙头清单
lines.append(f"### 2.1 龙头清单 (共 {len(leaders)} 家, {len(MAJOR_INDS)} 个行业)\n\n")
lines.append("| 行业 | 排名 | 代码 | 名称 | 平均市值 (亿元) |\n")
lines.append("|---|---|---|---|---|\n")
leaders_sorted = sorted(leaders, key=lambda x: (x["industry"], x["rank"]))
for l in leaders_sorted:
    lines.append(f"| {l['industry']} | #{l['rank']} | {l['ts_code']} | {l['name']} | {l['avg_mv']/10000:.1f} |\n")
lines.append("\n")

# 2.2 龙头整体 vs 全市场 vs 非龙头基线
lines.append("### 2.2 龙头 vs 全市场 vs 非龙头 — D+10 / D+40 胜率对比\n\n")

non_leader_samples = [s for s in samples if s["ts_code"] not in leader_codes
                       and s["industry"] in MAJOR_INDS]

lines.append("| 类别 | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
for label, pool in [("龙头 (前 3)", leader_samples),
                    ("非龙头 (主要行业内)", non_leader_samples),
                    ("全市场", samples)]:
    lines.append(f"| {label} | {len(pool):,} |"
                 f" {winrate(pool, 10):.1f}% | {avg_ret(pool, 10):+.2f}% |"
                 f" {winrate(pool, 40):.1f}% | {avg_ret(pool, 40):+.2f}% |\n")
lines.append("\n")

# 2.3 龙头按市值/PE 分层
lines.append("### 2.3 龙头股按市值分层胜率\n\n")
lines.append("| 市值 | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
mv_data = []
for mv in MV_ORDER:
    sub = [s for s in leader_samples if mv_bucket(s.get("total_mv")) == mv]
    if len(sub) < 50: continue
    mv_data.append({
        "mv": mv, "n": len(sub),
        "wr10": winrate(sub, 10), "ret10": avg_ret(sub, 10),
        "wr40": winrate(sub, 40), "ret40": avg_ret(sub, 40),
    })
max_wr10 = max(d["wr10"] or 0 for d in mv_data)
max_wr40 = max(d["wr40"] or 0 for d in mv_data)
for d in mv_data:
    wr10 = f"**{d['wr10']:.1f}%**" if d['wr10'] == max_wr10 else f"{d['wr10']:.1f}%"
    wr40 = f"**{d['wr40']:.1f}%**" if d['wr40'] == max_wr40 else f"{d['wr40']:.1f}%"
    lines.append(f"| {d['mv']} | {d['n']:,} | {wr10} | {d['ret10']:+.2f}% | {wr40} | {d['ret40']:+.2f}% |\n")
lines.append("\n")

lines.append("### 2.4 龙头股按 PE 分层胜率\n\n")
lines.append("| PE | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
pe_data = []
for pe in PE_ORDER:
    sub = [s for s in leader_samples if pe_bucket(s.get("pe")) == pe]
    if len(sub) < 50: continue
    pe_data.append({
        "pe": pe, "n": len(sub),
        "wr10": winrate(sub, 10), "ret10": avg_ret(sub, 10),
        "wr40": winrate(sub, 40), "ret40": avg_ret(sub, 40),
    })
if pe_data:
    max_wr10 = max(d["wr10"] or 0 for d in pe_data)
    max_wr40 = max(d["wr40"] or 0 for d in pe_data)
    for d in pe_data:
        wr10 = f"**{d['wr10']:.1f}%**" if d['wr10'] == max_wr10 else f"{d['wr10']:.1f}%"
        wr40 = f"**{d['wr40']:.1f}%**" if d['wr40'] == max_wr40 else f"{d['wr40']:.1f}%"
        lines.append(f"| {d['pe']} | {d['n']:,} | {wr10} | {d['ret10']:+.2f}% | {wr40} | {d['ret40']:+.2f}% |\n")
lines.append("\n")

# 2.5 龙头按行业排序的胜率
lines.append("### 2.5 龙头股按行业 — D+10 / D+40 胜率\n\n")
lines.append("| 行业 | 龙头数 | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|---|\n")
ind_data = []
for ind in MAJOR_INDS:
    leader_in_ind = [l["ts_code"] for l in leaders if l["industry"] == ind]
    sub = [s for s in leader_samples if s["industry"] == ind]
    if len(sub) < 100: continue
    ind_data.append({
        "ind": ind, "n_leader": len(leader_in_ind), "n": len(sub),
        "wr10": winrate(sub, 10), "ret10": avg_ret(sub, 10),
        "wr40": winrate(sub, 40), "ret40": avg_ret(sub, 40),
    })
ind_data.sort(key=lambda x: -(x["wr40"] or 0))
max_wr10 = max(d["wr10"] or 0 for d in ind_data)
max_wr40 = max(d["wr40"] or 0 for d in ind_data)
for d in ind_data:
    wr10 = f"**{d['wr10']:.1f}%**" if d['wr10'] == max_wr10 else f"{d['wr10']:.1f}%"
    wr40 = f"**{d['wr40']:.1f}%**" if d['wr40'] == max_wr40 else f"{d['wr40']:.1f}%"
    lines.append(f"| {d['ind']} | {d['n_leader']} | {d['n']:,} | {wr10} | {d['ret10']:+.2f}% | {wr40} | {d['ret40']:+.2f}% |\n")
lines.append("\n")

# 2.6 龙头 (市值×PE) 二维交叉
lines.append("### 2.6 龙头股 (市值 × PE) 二维 D+40 胜率\n\n")
lines.append("> 加粗 = 该交叉表内最高胜率\n\n")
lines.append("| 市值 \\ PE |" + "".join(f" {pe} |" for pe in PE_ORDER) + "\n")
lines.append("|---|" + "---|" * len(PE_ORDER) + "\n")
matrix = {}
max_wr = -1
max_key = None
for mv in MV_ORDER:
    for pe in PE_ORDER:
        sub = [s for s in leader_samples
               if mv_bucket(s.get("total_mv")) == mv
               and pe_bucket(s.get("pe")) == pe]
        if len(sub) < 50:
            matrix[(mv, pe)] = (None, len(sub))
            continue
        wr = winrate(sub, 40)
        matrix[(mv, pe)] = (wr, len(sub))
        if wr is not None and wr > max_wr:
            max_wr = wr; max_key = (mv, pe)

for mv in MV_ORDER:
    line = f"| **{mv}** |"
    for pe in PE_ORDER:
        wr, n = matrix.get((mv, pe), (None, 0))
        if wr is None:
            line += " - |"
        elif (mv, pe) == max_key:
            line += f" **{wr:.1f}% (n={n:,})** |"
        else:
            line += f" {wr:.1f}% (n={n:,}) |"
    lines.append(line + "\n")
lines.append("\n")

# Top 5 龙头股个股胜率
lines.append("### 2.7 单只龙头股 D+40 胜率 Top 20\n\n")
stock_stats = defaultdict(list)
for s in leader_samples:
    if s.get("r40") is not None:
        stock_stats[s["ts_code"]].append(s["r40"])
stock_wr40 = []
for ts, rs in stock_stats.items():
    if len(rs) < 50: continue
    wr = float(np.mean([1 if r > 0 else 0 for r in rs])) * 100
    avg = float(np.mean(rs))
    stock_wr40.append({
        "ts_code": ts, "name": ts_to_name.get(ts, ""),
        "industry": ts_to_ind.get(ts), "n": len(rs), "wr40": wr, "avg40": avg,
        "avg_mv": stock_avg_mv.get(ts, 0) / 10000,
    })
stock_wr40.sort(key=lambda x: -x["wr40"])

lines.append("| 排名 | 代码 | 名称 | 行业 | 平均市值(亿) | 样本 | **D+40 胜率** | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|---|---|\n")
for i, st in enumerate(stock_wr40[:20]):
    lines.append(f"| {i+1} | {st['ts_code']} | {st['name']} | {st['industry']} | {st['avg_mv']:.1f} | {st['n']:,} | **{st['wr40']:.1f}%** | {st['avg40']:+.2f}% |\n")
lines.append("\n")

lines.append("### 2.8 单只龙头股 D+40 胜率 Bottom 10 (最差)\n\n")
lines.append("| 排名 | 代码 | 名称 | 行业 | 平均市值(亿) | 样本 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|---|---|\n")
for i, st in enumerate(stock_wr40[-10:]):
    lines.append(f"| {len(stock_wr40)-9+i} | {st['ts_code']} | {st['name']} | {st['industry']} | {st['avg_mv']:.1f} | {st['n']:,} | {st['wr40']:.1f}% | {st['avg40']:+.2f}% |\n")
lines.append("\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"\n报告: {OUT}")
print(f"龙头股: {len(leader_codes)}")
print(f"龙头单股 D+40 胜率最高: {stock_wr40[0]['name']} ({stock_wr40[0]['ts_code']}) {stock_wr40[0]['wr40']:.1f}%")
print(f"龙头单股 D+40 胜率最低: {stock_wr40[-1]['name']} ({stock_wr40[-1]['ts_code']}) {stock_wr40[-1]['wr40']:.1f}%")
