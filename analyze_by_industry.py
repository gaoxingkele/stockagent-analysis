"""三层分析: 行业 → 因子 IC → 行业内 (市值 / PE) 二级分层。

输出: output/backtest_factors_2026_04/report_by_industry.md
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path("output/backtest_factors_2026_04")
OUT = ROOT / "report_by_industry.md"

# ─── 加载 universe + 行业映射 ───
with open(ROOT / "universe.json", encoding="utf-8") as f:
    uni = json.load(f)
ts_to_ind = {s["ts_code"]: (s.get("industry") or "未分类") for s in uni["stocks"]}

ind_count = Counter(ts_to_ind.values())
# 取股票数 ≥ 40 的行业
MAJOR_INDS = sorted([k for k, n in ind_count.items() if n >= 40],
                    key=lambda k: -ind_count[k])
print(f"主要行业 (≥40只): {len(MAJOR_INDS)} 个, 覆盖 {sum(ind_count[i] for i in MAJOR_INDS)}/{sum(ind_count.values())} 只")

# ─── 加载所有 samples 并加 industry 字段 ───
print("加载所有 samples...")
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

# 仅保留主要行业
major = [s for s in samples if s["industry"] in MAJOR_INDS]
print(f"主要行业内样本: {len(major):,}")

# ─── 通用工具 ───
HOLD = [5, 20, 30, 40]
FACTORS = ["mf_divergence", "mf_strength", "mf_consecutive",
           "market_score_adj", "adx", "winner_rate", "main_net", "holder_pct"]

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

def ic_of(samples, factor, hold):
    xy = [(s[factor], s.get(f"r{hold}")) for s in samples
          if s.get(factor) is not None and s.get(f"r{hold}") is not None]
    if len(xy) < 50: return None
    x, y = zip(*xy)
    if np.std(x) < 1e-6: return None
    return float(np.corrcoef(x, y)[0, 1])

def stats_of(samples, hold):
    rs = [s.get(f"r{hold}") for s in samples if s.get(f"r{hold}") is not None]
    dds = [s.get(f"dd{hold}") for s in samples if s.get(f"dd{hold}") is not None]
    if not rs: return None, None, None
    return float(np.mean(rs)), float(np.mean([1 if r > 0 else 0 for r in rs])) * 100, float(np.mean(dds)) if dds else 0

# ═══════════════════════════════════════════════════════════════
# Build report
# ═══════════════════════════════════════════════════════════════
lines = []
lines.append("# 行业分层因子表现报告\n\n")
lines.append(f"> 样本: {len(samples):,}, 主要行业: {len(MAJOR_INDS)} 个 (股票数≥40), 期间 2025-04 ~ 2026-04\n")
lines.append(f"> 主要行业样本占比: {len(major)/len(samples)*100:.1f}%\n\n")

# ─── Section 1: 行业基线 ───
lines.append("## 一、行业整体收益基线\n\n")
lines.append("| 行业 | 股票数 | 样本 | D+5 涨/胜 | D+20 涨/胜 | D+30 涨/胜 | D+40 涨/胜 | D+40 回撤 |\n")
lines.append("|---|---|---|---|---|---|---|---|\n")
ind_baseline = []
for ind in MAJOR_INDS:
    bucket = [s for s in major if s["industry"] == ind]
    if not bucket: continue
    row = {"ind": ind, "n_stock": ind_count[ind], "n": len(bucket)}
    for h in HOLD:
        avg, wr, dd = stats_of(bucket, h)
        row[f"avg{h}"] = avg; row[f"wr{h}"] = wr; row[f"dd{h}"] = dd
    ind_baseline.append(row)

# 按 D+40 胜率排序
ind_baseline.sort(key=lambda x: -(x.get("wr40") or 0))
for r in ind_baseline:
    line = f"| {r['ind']} | {r['n_stock']} | {r['n']:,} |"
    for h in HOLD:
        avg = r.get(f"avg{h}"); wr = r.get(f"wr{h}")
        line += f" {avg:+.2f}% / {wr:.1f}% |" if avg is not None else " - |"
    line += f" {r.get('dd40', 0):.2f}% |"
    lines.append(line + "\n")
lines.append("\n")

# ─── Section 2: 行业 × 因子 IC 表 (核心) ───
lines.append("## 二、行业 × 因子 IC 矩阵\n\n")
lines.append("> IC(20d) 为主, 标记 IC(20d) **加粗**：≥+0.04 ✓✓ / ≥+0.02 ✓ / ≤-0.04 ✗✗ / ≤-0.02 ✗\n\n")

strong_combos = []  # (industry, factor, ic20, ic40) for downstream sub-analysis
for factor in FACTORS:
    lines.append(f"### {factor}\n\n")
    lines.append("| 行业 | 样本 | IC(5d) | **IC(20d)** | IC(30d) | IC(40d) | 标记 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    rows = []
    for ind in MAJOR_INDS:
        bucket = [s for s in major if s["industry"] == ind]
        if len(bucket) < 200: continue
        ics = {}
        for h in HOLD:
            ics[h] = ic_of(bucket, factor, h)
        rows.append({"ind": ind, "n": len(bucket), **{f"ic{h}": ics[h] for h in HOLD}})
    # 按 IC(20d) 排序
    rows.sort(key=lambda x: -(x.get("ic20") or -99))
    for r in rows:
        ic5 = r.get("ic5"); ic20 = r.get("ic20"); ic30 = r.get("ic30"); ic40 = r.get("ic40")
        mark = ""
        if ic20 is not None:
            if ic20 >= 0.04: mark = "✓✓"
            elif ic20 >= 0.02: mark = "✓"
            elif ic20 <= -0.04: mark = "✗✗"
            elif ic20 <= -0.02: mark = "✗"
        line = f"| {r['ind']} | {r['n']:,} |"
        for ic in (ic5, ic20, ic30, ic40):
            line += f" {ic:+.4f} |" if ic is not None else " - |"
        line += f" {mark} |"
        lines.append(line + "\n")
        # Collect strong combos for sub-analysis
        if ic20 is not None and abs(ic20) >= 0.04:
            strong_combos.append((r["ind"], factor, ic20, ic40 or 0, r["n"]))
    lines.append("\n")

# ─── Section 3: 强信号组合的 mv/pe 二级分层 ───
lines.append("## 三、强信号 (|IC(20d)|≥0.04) 组合的市值/PE 二级分层\n\n")
lines.append(f"> 共 {len(strong_combos)} 个强信号组合, 按 |IC(20d)| 降序展开\n\n")

# 按 |IC| 降序
strong_combos.sort(key=lambda x: -abs(x[2]))

for idx, (ind, factor, ic20, ic40, n) in enumerate(strong_combos[:30]):  # top 30
    lines.append(f"### {idx+1}. {ind} × {factor} | IC(20d)={ic20:+.4f} | n={n:,}\n\n")
    bucket = [s for s in major if s["industry"] == ind]

    # 市值分层 IC
    lines.append("**市值分层 IC**:\n\n")
    lines.append("| 市值桶 | 样本 | σ | IC(5d) | IC(20d) | IC(30d) | IC(40d) |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for mv in MV_ORDER:
        sub = [s for s in bucket if mv_bucket(s.get("total_mv")) == mv]
        if len(sub) < 100: continue
        fvals = [s.get(factor) for s in sub if s.get(factor) is not None]
        sigma = float(np.std(fvals)) if fvals else 0
        line = f"| {mv} | {len(sub):,} | {sigma:.2f} |"
        for h in HOLD:
            ic = ic_of(sub, factor, h)
            line += f" {ic:+.4f} |" if ic is not None else " - |"
        lines.append(line + "\n")
    lines.append("\n")

    # PE 分层 IC
    lines.append("**PE 分层 IC**:\n\n")
    lines.append("| PE 桶 | 样本 | σ | IC(5d) | IC(20d) | IC(30d) | IC(40d) |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for pe in PE_ORDER:
        sub = [s for s in bucket if pe_bucket(s.get("pe")) == pe]
        if len(sub) < 100: continue
        fvals = [s.get(factor) for s in sub if s.get(factor) is not None]
        sigma = float(np.std(fvals)) if fvals else 0
        line = f"| {pe} | {len(sub):,} | {sigma:.2f} |"
        for h in HOLD:
            ic = ic_of(sub, factor, h)
            line += f" {ic:+.4f} |" if ic is not None else " - |"
        lines.append(line + "\n")
    lines.append("\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"\n报告: {OUT}")
print(f"强信号组合: {len(strong_combos)} 个 (|IC(20d)|≥0.04)")
print(f"展开二级分层: top {min(30, len(strong_combos))} 个")
