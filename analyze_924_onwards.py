"""924 行情后 (2024-09-24 起) 的多维分层分析。

复用 3 年回测样本, 按 trade_date >= 20240924 切片。
分层维度: 行业 / 市值 / PE / 风格段。
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path("output/backtest_3y_2023_2026")
OUT = ROOT / "report_924_onwards.md"
START_924 = "20240924"

with open(ROOT / "universe.json", encoding="utf-8") as f:
    uni = json.load(f)
ts_to_ind = {s["ts_code"]: (s.get("industry") or "未分类") for s in uni["stocks"]}
ts_to_name = {s["ts_code"]: s.get("name", "") for s in uni["stocks"]}

with open(ROOT / "market_phases.json", encoding="utf-8") as f:
    phases = json.load(f)
date_to_phase = {p["trade_date"]: p["phase"] for p in phases}

ind_count = Counter(ts_to_ind.values())
MAJOR_INDS = sorted([k for k, n in ind_count.items() if n >= 40], key=lambda k: -ind_count[k])

print(f"加载 924 后样本 (>= {START_924})...")
samples = []
for gf in sorted((ROOT / "group_results").glob("group_*.jsonl")):
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
                if s.get("trade_date", "") < START_924: continue
                s["industry"] = ts_to_ind.get(s["ts_code"], "未分类")
                s["phase"] = date_to_phase.get(s["trade_date"], "unknown")
                samples.append(s)
            except: pass
print(f"样本数: {len(samples):,}")

# 极端剔除
EXTREME = 50.0
n_orig = len(samples)
samples = [s for s in samples if s.get("r40") is None or abs(s["r40"]) <= EXTREME]
print(f"剔除 |D+40|>{EXTREME}% 极端样本: {n_orig - len(samples):,}, 净 {len(samples):,}")

phase_count = Counter(s["phase"] for s in samples)
print(f"风格分布: {dict(phase_count)}")

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

def winrate(samples, h):
    rs = [s.get(f"r{h}") for s in samples if s.get(f"r{h}") is not None]
    return float(np.mean([1 if r > 0 else 0 for r in rs])) * 100 if rs else None
def avg_ret(samples, h):
    rs = [s.get(f"r{h}") for s in samples if s.get(f"r{h}") is not None]
    return float(np.mean(rs)) if rs else None

def fmt_winrate_5h(samples, label, n):
    line = f"| {label} | {n:,} |"
    wrs = [winrate(samples, h) for h in HOLD]
    rets = [avg_ret(samples, h) for h in HOLD]
    max_wr = max((w for w in wrs if w is not None), default=None)
    for w, r in zip(wrs, rets):
        if w is None: line += " - |"
        elif w == max_wr: line += f" **{w:.1f}%** ({r:+.2f}%) |"
        else: line += f" {w:.1f}% ({r:+.2f}%) |"
    return line

# Build report
lines = []
lines.append(f"# 924 行情后 (2024-09-24 起) 多维分层分析\n\n")
lines.append(f"> 样本: {len(samples):,} (剔除极端涨幅后)\n")
lines.append(f"> 期间: {START_924} ~ 2026-03 末\n")
lines.append(f"> 风格: 牛 {phase_count.get('bull',0):,} / 震荡 {phase_count.get('sideways',0):,} / 熊 {phase_count.get('bear',0):,}\n\n")

# 1. 风格基线
lines.append("## 一、风格基线 (924 起)\n\n")
lines.append("| 风格 | 样本 |" + "".join(f" D+{h} (胜/涨) |" for h in HOLD) + "\n")
lines.append("|---|---|" + "---|" * len(HOLD) + "\n")
for ph in ["bull", "sideways", "bear"]:
    sub = [s for s in samples if s["phase"] == ph]
    if len(sub) < 100: continue
    lines.append(fmt_winrate_5h(sub, ph, len(sub)) + "\n")
lines.append(fmt_winrate_5h(samples, "全期", len(samples)) + "\n\n")

# 2. 行业 D+10 / D+40 胜率 (按 D+40 降序)
lines.append("## 二、行业 D+10 / D+40 胜率 (按 D+40 降序, 加粗 = 最高)\n\n")
lines.append("| 行业 | 样本 | D+5 | D+10 | D+20 | D+30 | D+40 |\n")
lines.append("|---|---|---|---|---|---|---|\n")
ind_data = []
for ind in MAJOR_INDS:
    bucket = [s for s in samples if s["industry"] == ind]
    if len(bucket) < 200: continue
    row = {"ind": ind, "n": len(bucket)}
    for h in HOLD: row[f"wr{h}"] = winrate(bucket, h)
    ind_data.append(row)
ind_data.sort(key=lambda x: -(x.get("wr40") or 0))
max_wr40 = max((d.get("wr40") or 0) for d in ind_data)
for r in ind_data:
    line = f"| {r['ind']} | {r['n']:,} |"
    for h in HOLD:
        v = r.get(f"wr{h}")
        if v is None: line += " - |"
        elif h == 40 and v == max_wr40: line += f" **{v:.1f}%** |"
        else: line += f" {v:.1f}% |"
    lines.append(line + "\n")
lines.append("\n")

# 3. 市值 × 风格
lines.append("## 三、市值 × 风格 D+40 胜率\n\n")
lines.append("| 市值 | 牛 | 震荡 | 熊 | 全期 |\n")
lines.append("|---|---|---|---|---|\n")
for mv in MV_ORDER:
    base = [s for s in samples if mv_bucket(s.get("total_mv")) == mv]
    if len(base) < 200: continue
    line = f"| {mv} |"
    vals = []
    for ph in ["bull", "sideways", "bear"]:
        sub = [s for s in base if s["phase"] == ph]
        v = winrate(sub, 40) if len(sub) >= 50 else None
        vals.append(v)
    max_v = max((v for v in vals if v is not None), default=None)
    for v in vals:
        if v is None: line += " - |"
        elif v == max_v: line += f" **{v:.1f}%** |"
        else: line += f" {v:.1f}% |"
    line += f" {winrate(base, 40):.1f}% |"
    lines.append(line + "\n")
lines.append("\n")

# 4. PE × 风格
lines.append("## 四、PE × 风格 D+40 胜率\n\n")
lines.append("| PE | 牛 | 震荡 | 熊 | 全期 |\n")
lines.append("|---|---|---|---|---|\n")
for pe in PE_ORDER:
    base = [s for s in samples if pe_bucket(s.get("pe")) == pe]
    if len(base) < 200: continue
    line = f"| {pe} |"
    vals = []
    for ph in ["bull", "sideways", "bear"]:
        sub = [s for s in base if s["phase"] == ph]
        v = winrate(sub, 40) if len(sub) >= 50 else None
        vals.append(v)
    max_v = max((v for v in vals if v is not None), default=None)
    for v in vals:
        if v is None: line += " - |"
        elif v == max_v: line += f" **{v:.1f}%** |"
        else: line += f" {v:.1f}% |"
    line += f" {winrate(base, 40):.1f}% |"
    lines.append(line + "\n")
lines.append("\n")

# 5. (市值 × PE) 二维 D+40 胜率
lines.append("## 五、(市值 × PE) 二维 D+40 胜率 — 924 起\n\n")
lines.append("| 市值 \\ PE |" + "".join(f" {pe} |" for pe in PE_ORDER) + "\n")
lines.append("|---|" + "---|" * len(PE_ORDER) + "\n")
mat = {}; max_v = -1; max_k = None
for mv in MV_ORDER:
    for pe in PE_ORDER:
        sub = [s for s in samples if mv_bucket(s.get("total_mv")) == mv and pe_bucket(s.get("pe")) == pe]
        if len(sub) < 200:
            mat[(mv, pe)] = (None, len(sub)); continue
        v = winrate(sub, 40)
        mat[(mv, pe)] = (v, len(sub))
        if v is not None and v > max_v: max_v = v; max_k = (mv, pe)
for mv in MV_ORDER:
    line = f"| **{mv}** |"
    for pe in PE_ORDER:
        v, n = mat.get((mv, pe), (None, 0))
        if v is None: line += " - |"
        elif (mv, pe) == max_k: line += f" **{v:.1f}% (n={n:,})** |"
        else: line += f" {v:.1f}% (n={n:,}) |"
    lines.append(line + "\n")
lines.append("\n")

# 6. 行业 × 市值 × PE 强组合 (D+40 ≥65%)
lines.append("## 六、强组合 (D+40 胜率 ≥65%, 样本 ≥200)\n\n")
combos = []
for ind in MAJOR_INDS:
    for mv in MV_ORDER:
        for pe in PE_ORDER:
            sub = [s for s in samples if s["industry"] == ind
                   and mv_bucket(s.get("total_mv")) == mv
                   and pe_bucket(s.get("pe")) == pe]
            if len(sub) < 200: continue
            v = winrate(sub, 40)
            if v is not None and v >= 65:
                combos.append({"ind": ind, "mv": mv, "pe": pe, "n": len(sub), "wr40": v,
                                "wr10": winrate(sub, 10), "ret40": avg_ret(sub, 40)})
combos.sort(key=lambda x: -x["wr40"])
lines.append(f"> 共找到 {len(combos)} 组强组合\n\n")
lines.append("| 行业 | 市值 | PE | 样本 | D+10 胜率 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|---|\n")
def _fmt(v, suffix="%"):
    return f"{v:.1f}{suffix}" if v is not None else "-"
def _fmt_ret(v):
    return f"{v:+.2f}%" if v is not None else "-"
for c in combos[:40]:
    lines.append(f"| {c['ind']} | {c['mv']} | {c['pe']} | {c['n']:,} | {_fmt(c['wr10'])} | **{_fmt(c['wr40'])}** | {_fmt_ret(c['ret40'])} |\n")
lines.append("\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"\n报告: {OUT}")
print(f"找到 D+40 ≥65% 强组合: {len(combos)}")
