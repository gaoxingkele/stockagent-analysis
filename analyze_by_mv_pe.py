"""按市值/PE 分层分析: 8 因子 × 4 持有期 × 多个 mv/pe 桶。
输出 output/backtest_factors_2026_04/report_by_mv_pe.md
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path("output/backtest_factors_2026_04")
OUT = ROOT / "report_by_mv_pe.md"

print("加载所有 samples...")
samples = []
for gf in sorted((ROOT / "group_results").glob("group_*.jsonl")):
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                samples.append(json.loads(line))
            except Exception:
                pass
print(f"总样本: {len(samples):,}")

# ─── 市值 / PE 分桶 ───
def mv_bucket(mv):
    """total_mv 单位是万元。20亿=200000万"""
    if mv is None or mv <= 0: return None
    if mv < 500000: return "20-50亿"
    if mv < 1000000: return "50-100亿"
    if mv < 3000000: return "100-300亿"
    if mv < 10000000: return "300-1000亿"
    return "1000亿+"

def pe_bucket(pe):
    if pe is None: return None
    if pe < 0: return "亏损(PE<0)"
    if pe < 15: return "0-15"
    if pe < 30: return "15-30"
    if pe < 50: return "30-50"
    if pe < 100: return "50-100"
    return "100+"

MV_ORDER = ["20-50亿", "50-100亿", "100-300亿", "300-1000亿", "1000亿+"]
PE_ORDER = ["亏损(PE<0)", "0-15", "15-30", "30-50", "50-100", "100+"]
HOLD = [5, 20, 30, 40]
FACTORS = ["mf_divergence", "mf_strength", "mf_consecutive",
           "market_score_adj", "adx", "winner_rate", "main_net", "holder_pct"]

# 给每个 sample 标桶
for s in samples:
    s["_mv_b"] = mv_bucket(s.get("total_mv"))
    s["_pe_b"] = pe_bucket(s.get("pe"))

# ─── 1) 大盘环境基线: 不分因子, 仅看 mv/pe 桶的整体表现 ───
def calc_baseline(samples, key_fn, order):
    rows = []
    for k in order:
        bucket = [s for s in samples if key_fn(s) == k]
        if not bucket: continue
        row = {"key": k, "n": len(bucket), "pct": len(bucket) / len(samples) * 100}
        for h in HOLD:
            rs = [s.get(f"r{h}") for s in bucket if s.get(f"r{h}") is not None]
            dds = [s.get(f"dd{h}") for s in bucket if s.get(f"dd{h}") is not None]
            if rs:
                row[f"avg{h}"] = float(np.mean(rs))
                row[f"wr{h}"] = float(np.mean([1 if r > 0 else 0 for r in rs])) * 100
                row[f"dd{h}"] = float(np.mean(dds)) if dds else 0
        rows.append(row)
    return rows


# ─── 2) 因子 IC 按市值/PE 分层 ───
def calc_factor_ic_by_layer(samples, factor, key_fn, order):
    """每个 mv/pe 桶里, 该因子的 IC + σ。"""
    rows = []
    for k in order:
        bucket = [s for s in samples if key_fn(s) == k]
        n = len(bucket)
        if n < 50: continue
        fvals = [s.get(factor) for s in bucket if s.get(factor) is not None]
        sigma = float(np.std(fvals)) if fvals else 0
        row = {"key": k, "n": n, "sigma": sigma}
        for h in HOLD:
            x, y = [], []
            for s in bucket:
                fv = s.get(factor)
                rv = s.get(f"r{h}")
                if fv is None or rv is None: continue
                x.append(fv); y.append(rv)
            if len(x) > 50 and np.std(x) > 1e-6:
                ic = float(np.corrcoef(x, y)[0, 1])
                row[f"ic{h}"] = ic
            else:
                row[f"ic{h}"] = None
        rows.append(row)
    return rows


# ─── 输出 ───
def fmt_table_baseline(rows, label):
    out = []
    out.append(f"| {label} | 占比 | 样本 |")
    for h in HOLD:
        out.append(f" D+{h}涨 | D+{h}胜率 | D+{h}回撤 |")
    out = ["".join(out)]
    out.append("|---|---|---|" + "---|---|---|" * len(HOLD))
    for r in rows:
        line = f"| {r['key']} | {r['pct']:.1f}% | {r['n']:,} |"
        for h in HOLD:
            avg = r.get(f"avg{h}")
            wr = r.get(f"wr{h}")
            dd = r.get(f"dd{h}")
            line += f" {avg:+.2f}% | {wr:.1f}% | {dd:.2f}% |" if avg is not None else " - | - | - |"
        out.append(line)
    return "\n".join(out)


def fmt_table_ic(rows, label):
    head = f"| {label} | 样本 | σ | " + " | ".join(f"IC({h}d)" for h in HOLD) + " |"
    sep = "|---|---|---|" + "---|" * len(HOLD)
    out = [head, sep]
    for r in rows:
        line = f"| {r['key']} | {r['n']:,} | {r['sigma']:.2f} |"
        for h in HOLD:
            ic = r.get(f"ic{h}")
            line += f" {ic:+.4f} |" if ic is not None else " - |"
        out.append(line)
    return "\n".join(out)


lines = []
lines.append("# 按市值/PE 分层的因子表现\n\n")
lines.append(f"> 样本: {len(samples):,}, 期间 2025-04 ~ 2026-04 (12 个月)\n")
lines.append(f"> mv/pe 取自每条样本买入日的 daily_basic, 即时数据\n\n")

# ─── 1. 市值 baseline ───
lines.append("## 一、按市值分层的整体收益基线 (不区分因子)\n\n")
lines.append(fmt_table_baseline(calc_baseline(samples, lambda s: s["_mv_b"], MV_ORDER), "市值桶"))
lines.append("\n\n")

# ─── 2. PE baseline ───
lines.append("## 二、按 PE 分层的整体收益基线 (不区分因子)\n\n")
lines.append(fmt_table_baseline(calc_baseline(samples, lambda s: s["_pe_b"], PE_ORDER), "PE 桶"))
lines.append("\n\n")

# ─── 3. 8 因子 × 5 市值桶 IC ───
lines.append("## 三、8 因子在不同市值层的 IC 表现\n\n")
for factor in FACTORS:
    rows = calc_factor_ic_by_layer(samples, factor, lambda s: s["_mv_b"], MV_ORDER)
    if not rows: continue
    lines.append(f"### {factor}\n\n")
    lines.append(fmt_table_ic(rows, "市值桶"))
    lines.append("\n\n")

# ─── 4. 8 因子 × 6 PE 桶 IC ───
lines.append("## 四、8 因子在不同 PE 层的 IC 表现\n\n")
for factor in FACTORS:
    rows = calc_factor_ic_by_layer(samples, factor, lambda s: s["_pe_b"], PE_ORDER)
    if not rows: continue
    lines.append(f"### {factor}\n\n")
    lines.append(fmt_table_ic(rows, "PE 桶"))
    lines.append("\n\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"\n报告: {OUT}\n")
