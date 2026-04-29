"""3 年回测分析: 市场风格分段 + 极端涨幅剔除 + 多维分层。

数据源:
  - 样本: output/backtest_3y_2023_2026/group_results/*.jsonl
  - 风格: output/backtest_3y_2023_2026/market_phases.json
  - 行业: output/backtest_3y_2023_2026/universe.json

输出: output/backtest_3y_2023_2026/report_3y_full.md
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "output" / "backtest_3y_2023_2026"
OUT = DIR / "report_3y_full.md"

print("[1/6] 加载 universe + 风格映射...")
with open(DIR / "universe.json", encoding="utf-8") as f:
    uni = json.load(f)
ts_to_ind = {s["ts_code"]: (s.get("industry") or "未分类") for s in uni["stocks"]}
ts_to_name = {s["ts_code"]: s.get("name", "") for s in uni["stocks"]}

with open(DIR / "market_phases.json", encoding="utf-8") as f:
    phases = json.load(f)
date_to_phase = {p["trade_date"]: p["phase"] for p in phases}

ind_count = Counter(ts_to_ind.values())
MAJOR_INDS = sorted([k for k, n in ind_count.items() if n >= 40], key=lambda k: -ind_count[k])

print("[2/6] 加载所有 samples (3 年, 367 万样本)...")
samples = []
for gf in sorted((DIR / "group_results").glob("group_*.jsonl")):
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
                s["industry"] = ts_to_ind.get(s["ts_code"], "未分类")
                s["phase"] = date_to_phase.get(s["trade_date"], "unknown")
                samples.append(s)
            except: pass
print(f"  样本: {len(samples):,}")

# 注入 daily_basic 字段 (mv, pe) 已在原 1y 数据中, 但 3y 数据需要重补
# 先看是否已含
sample_with_mv = sum(1 for s in samples if s.get("total_mv") is not None)
print(f"  含 total_mv 的样本: {sample_with_mv:,} ({sample_with_mv/len(samples)*100:.1f}%)")

# 桶
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
HOLD = [5, 10, 20, 30, 40]
PHASES = ["bull", "sideways", "bear"]

print("[3/6] 极端涨幅过滤...")
EXTREME = 50.0
n_orig = len(samples)
samples_clean = [s for s in samples
                 if s.get("r40") is None or abs(s["r40"]) <= EXTREME]
n_excluded = n_orig - len(samples_clean)
print(f"  剔除 |D+40 涨幅| > {EXTREME}% 极端样本: {n_excluded:,} 条 ({n_excluded/n_orig*100:.2f}%)")
print(f"  剔除后样本: {len(samples_clean):,}")
samples = samples_clean

print("[4/6] 风格分段统计...")
phase_count = Counter(s["phase"] for s in samples)
print(f"  bull: {phase_count.get('bull',0):,}")
print(f"  sideways: {phase_count.get('sideways',0):,}")
print(f"  bear: {phase_count.get('bear',0):,}")
print(f"  unknown: {phase_count.get('unknown',0):,}")


def winrate(samples, h):
    rs = [s.get(f"r{h}") for s in samples if s.get(f"r{h}") is not None]
    if not rs: return None
    return float(np.mean([1 if r > 0 else 0 for r in rs])) * 100

def avg_ret(samples, h):
    rs = [s.get(f"r{h}") for s in samples if s.get(f"r{h}") is not None]
    if not rs: return None
    return float(np.mean(rs))

def avg_dd(samples, h):
    dds = [s.get(f"dd{h}") for s in samples if s.get(f"dd{h}") is not None]
    if not dds: return None
    return float(np.mean(dds))


def fmt_winrate_5h_row(samples, label, n):
    line = f"| {label} | {n:,} |"
    wrs = [winrate(samples, h) for h in HOLD]
    rets = [avg_ret(samples, h) for h in HOLD]
    max_wr = max((w for w in wrs if w is not None), default=None)
    for h, wr, ret in zip(HOLD, wrs, rets):
        if wr is None:
            line += " - |"
        elif wr == max_wr:
            line += f" **{wr:.1f}%** ({ret:+.2f}%) |"
        else:
            line += f" {wr:.1f}% ({ret:+.2f}%) |"
    return line


print("[5/6] 生成报告...")
lines = []
lines.append("# 3 年回测 (2023-01 ~ 2026-03) 全维度分析报告\n\n")
lines.append(f"> 原始样本: {n_orig:,}, 剔除极端涨幅 (|D+40|>{EXTREME}%): {n_excluded:,} ({n_excluded/n_orig*100:.2f}%)\n")
lines.append(f"> 净样本: {len(samples):,}\n")
lines.append(f"> Universe: 5,149 只 (市值≥20亿, 排除 ST, 沿用 1 年回测股票池)\n")
lines.append(f"> 风格: 上证指数 60 日 MA + 60 日动量 (bull/sideways/bear)\n")
lines.append(f"> 期间风格分布: 牛 {phase_count.get('bull',0):,} / 震荡 {phase_count.get('sideways',0):,} / 熊 {phase_count.get('bear',0):,}\n\n")

# ─── Section 1: 市场风格基线 ───
lines.append("## 一、市场风格分段基线\n\n")
lines.append("> 5 个持有期胜率, 加粗 = 该行最高\n\n")
lines.append("| 风格 | 样本 |" + "".join(f" D+{h} (胜/涨) |" for h in HOLD) + "\n")
lines.append("|---|---|" + "---|" * len(HOLD) + "\n")
for ph in PHASES:
    bucket = [s for s in samples if s["phase"] == ph]
    if len(bucket) < 100: continue
    lines.append(fmt_winrate_5h_row(bucket, ph, len(bucket)) + "\n")
lines.append(fmt_winrate_5h_row(samples, "全期 (含 unknown)", len(samples)) + "\n")
lines.append("\n")

# ─── Section 2: 风格 × 行业 ───
lines.append("## 二、风格 × 行业 D+40 胜率矩阵\n\n")
lines.append("> 加粗 = 该行业内最强阶段\n\n")
lines.append("| 行业 | 样本 | 牛 | 震荡 | 熊 | 全期 |\n")
lines.append("|---|---|---|---|---|---|\n")

ind_data = []
for ind in MAJOR_INDS:
    bucket = [s for s in samples if s["industry"] == ind]
    if len(bucket) < 200: continue
    row = {"ind": ind, "n": len(bucket)}
    for ph in PHASES:
        sub = [s for s in bucket if s["phase"] == ph]
        if len(sub) < 50:
            row[ph] = None
        else:
            row[ph] = winrate(sub, 40)
    row["all"] = winrate(bucket, 40)
    ind_data.append(row)

ind_data.sort(key=lambda x: -(x["all"] or 0))
for r in ind_data:
    line = f"| {r['ind']} | {r['n']:,} |"
    vals = [r["bull"], r["sideways"], r["bear"]]
    valid = [v for v in vals if v is not None]
    max_v = max(valid) if valid else None
    for v in vals:
        if v is None:
            line += " - |"
        elif v == max_v:
            line += f" **{v:.1f}%** |"
        else:
            line += f" {v:.1f}% |"
    line += f" {r['all']:.1f}% |" if r["all"] is not None else " - |"
    lines.append(line + "\n")
lines.append("\n")

# ─── Section 3: 风格 × 市值 ───
lines.append("## 三、风格 × 市值 D+10/D+40 胜率\n\n")
lines.append("| 市值 | 风格 | 样本 | D+5 胜 | D+10 胜 | D+20 胜 | D+30 胜 | D+40 胜 |\n")
lines.append("|---|---|---|---|---|---|---|---|\n")
for mv in MV_ORDER:
    for ph in PHASES + ["all"]:
        if ph == "all":
            sub = [s for s in samples if mv_bucket(s.get("total_mv")) == mv]
            ph_label = "全期"
        else:
            sub = [s for s in samples if mv_bucket(s.get("total_mv")) == mv and s["phase"] == ph]
            ph_label = ph
        if len(sub) < 100: continue
        line = f"| {mv} | {ph_label} | {len(sub):,} |"
        wrs = [winrate(sub, h) for h in HOLD]
        max_wr = max((w for w in wrs if w is not None), default=None)
        for w in wrs:
            if w is None: line += " - |"
            elif w == max_wr: line += f" **{w:.1f}%** |"
            else: line += f" {w:.1f}% |"
        lines.append(line + "\n")
    lines.append("|  |  |  |  |  |  |  |  |\n")  # 分隔
lines.append("\n")

# ─── Section 4: 风格 × PE ───
lines.append("## 四、风格 × PE D+10/D+40 胜率\n\n")
lines.append("| PE | 风格 | 样本 | D+5 胜 | D+10 胜 | D+20 胜 | D+30 胜 | D+40 胜 |\n")
lines.append("|---|---|---|---|---|---|---|---|\n")
for pe in PE_ORDER:
    for ph in PHASES + ["all"]:
        if ph == "all":
            sub = [s for s in samples if pe_bucket(s.get("pe")) == pe]
            ph_label = "全期"
        else:
            sub = [s for s in samples if pe_bucket(s.get("pe")) == pe and s["phase"] == ph]
            ph_label = ph
        if len(sub) < 100: continue
        line = f"| {pe} | {ph_label} | {len(sub):,} |"
        wrs = [winrate(sub, h) for h in HOLD]
        max_wr = max((w for w in wrs if w is not None), default=None)
        for w in wrs:
            if w is None: line += " - |"
            elif w == max_wr: line += f" **{w:.1f}%** |"
            else: line += f" {w:.1f}% |"
        lines.append(line + "\n")
    lines.append("|  |  |  |  |  |  |  |  |\n")
lines.append("\n")

# ─── Section 5: 8 因子 IC 在不同风格的稳定性 ───
lines.append("## 五、8 因子 IC 在不同风格的稳定性\n\n")
FACTORS = ["mf_divergence", "mf_strength", "mf_consecutive",
           "market_score_adj", "adx", "winner_rate", "main_net", "holder_pct"]

def ic_of(samples, factor, hold):
    xy = [(s[factor], s.get(f"r{hold}")) for s in samples
          if s.get(factor) is not None and s.get(f"r{hold}") is not None]
    if len(xy) < 50: return None
    x, y = zip(*xy)
    if np.std(x) < 1e-6: return None
    return float(np.corrcoef(x, y)[0, 1])

for factor in FACTORS:
    lines.append(f"### {factor}\n\n")
    lines.append("| 风格 | 样本 | σ | IC(5d) | IC(10d) | IC(20d) | IC(30d) | IC(40d) |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for ph in PHASES + ["all"]:
        if ph == "all":
            sub = samples
            ph_label = "全期"
        else:
            sub = [s for s in samples if s["phase"] == ph]
            ph_label = ph
        if len(sub) < 200: continue
        fvals = [s.get(factor) for s in sub if s.get(factor) is not None]
        sigma = float(np.std(fvals)) if fvals else 0
        line = f"| {ph_label} | {len(sub):,} | {sigma:.2f} |"
        for h in HOLD:
            ic = ic_of(sub, factor, h)
            line += f" {ic:+.4f} |" if ic is not None else " - |"
        lines.append(line + "\n")
    lines.append("\n")

# ─── Section 6: 跨风格稳健的"行业 × 市值 × PE" 组合 ───
lines.append("## 六、跨风格稳健的最优组合 (3 风格 D+40 胜率均 ≥55%)\n\n")
lines.append("> 在牛/震荡/熊三段都至少 55% 胜率, 才算稳健\n\n")
lines.append("| 行业 | 市值 | PE | 牛 | 震荡 | 熊 | 全期 | 全期样本 |\n")
lines.append("|---|---|---|---|---|---|---|---|\n")

robust = []
for ind in MAJOR_INDS:
    for mv in MV_ORDER:
        for pe in PE_ORDER:
            base = [s for s in samples
                    if s["industry"] == ind
                    and mv_bucket(s.get("total_mv")) == mv
                    and pe_bucket(s.get("pe")) == pe]
            if len(base) < 200: continue
            wrs = {}
            ok = True
            for ph in PHASES:
                sub = [s for s in base if s["phase"] == ph]
                if len(sub) < 50:
                    ok = False; break
                w = winrate(sub, 40)
                if w is None or w < 55:
                    ok = False; break
                wrs[ph] = w
            if not ok: continue
            wrs["all"] = winrate(base, 40)
            robust.append({
                "ind": ind, "mv": mv, "pe": pe, "n": len(base), **wrs,
            })

robust.sort(key=lambda r: -(r["all"] or 0))
for r in robust[:30]:
    lines.append(f"| {r['ind']} | {r['mv']} | {r['pe']} | "
                 f"{r['bull']:.1f}% | {r['sideways']:.1f}% | {r['bear']:.1f}% | "
                 f"**{r['all']:.1f}%** | {r['n']:,} |\n")
lines.append(f"\n> 共找到 {len(robust)} 组跨风格稳健组合\n\n")

# 写入
OUT.write_text("".join(lines), encoding="utf-8")
print(f"[6/6] 报告: {OUT}")
print(f"\n关键统计:")
print(f"  净样本: {len(samples):,}")
print(f"  风格: 牛 {phase_count.get('bull',0):,} / 震荡 {phase_count.get('sideways',0):,} / 熊 {phase_count.get('bear',0):,}")
print(f"  跨风格稳健组合: {len(robust)} 组 (≥55% 胜率)")
