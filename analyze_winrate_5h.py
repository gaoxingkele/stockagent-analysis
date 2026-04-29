"""按行业 + 强信号组合, 列出 D+5/10/20/30/40 5 个持有期胜率, 加粗最高。
输出: output/backtest_factors_2026_04/report_winrate_5h.md
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path("output/backtest_factors_2026_04")
OUT = ROOT / "report_winrate_5h.md"

with open(ROOT / "universe.json", encoding="utf-8") as f:
    uni = json.load(f)
ts_to_ind = {s["ts_code"]: (s.get("industry") or "未分类") for s in uni["stocks"]}
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
major = [s for s in samples if s["industry"] in MAJOR_INDS]

HOLD = [5, 10, 20, 30, 40]
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


def fmt_winrates_with_bold(values):
    """list of (label, wr) → 找最高 wr, 加粗。返回 markdown cell 字符串列表(顺序保持)"""
    if not any(v is not None for _, v in values):
        return [" - "] * len(values)
    valid = [(l, v) for l, v in values if v is not None]
    if not valid: return [" - "] * len(values)
    max_v = max(v for _, v in valid)
    return [f" **{v:.1f}%** " if v == max_v and abs(v - max_v) < 1e-9 else
            (f" {v:.1f}% " if v is not None else " - ")
            for l, v in values]


def ic_of(samples, factor, hold):
    xy = [(s[factor], s.get(f"r{hold}")) for s in samples
          if s.get(factor) is not None and s.get(f"r{hold}") is not None]
    if len(xy) < 50: return None
    x, y = zip(*xy)
    if np.std(x) < 1e-6: return None
    return float(np.corrcoef(x, y)[0, 1])


def group_table_winrate(rows, key_label):
    """rows: list of {key, n, wr_h: ...}, 输出 markdown 行(每行一个分组,胜率列加粗最高)"""
    out = [f"| {key_label} | 样本 |" + "".join(f" D+{h} 胜率 |" for h in HOLD)]
    out.append("|---|---|" + "---|" * len(HOLD))
    # 找各列最高
    col_max = {h: max((r.get(f"wr{h}") for r in rows if r.get(f"wr{h}") is not None), default=None) for h in HOLD}
    for r in rows:
        line = f"| {r['key']} | {r['n']:,} |"
        for h in HOLD:
            v = r.get(f"wr{h}")
            if v is None:
                line += " - |"
            elif col_max[h] is not None and abs(v - col_max[h]) < 1e-6:
                line += f" **{v:.1f}%** |"
            else:
                line += f" {v:.1f}% |"
        out.append(line)
    return "\n".join(out)


# ─── Build report ───
lines = []
lines.append("# 行业 + 因子分层 D+5/10/20/30/40 胜率全表 (加粗 = 该列最高)\n\n")
lines.append(f"> 样本: {len(samples):,} ({len(major):,} 在主要行业内)\n\n")

# 1. 行业基线 (D+5/10/20/30/40 胜率)
lines.append("## 一、行业基线胜率 (按 D+40 胜率降序)\n\n")
ind_rows = []
for ind in MAJOR_INDS:
    bucket = [s for s in major if s["industry"] == ind]
    if len(bucket) < 200: continue
    row = {"key": ind, "n": len(bucket)}
    for h in HOLD:
        row[f"wr{h}"] = winrate(bucket, h)
    ind_rows.append(row)
ind_rows.sort(key=lambda x: -(x.get("wr40") or 0))
lines.append(group_table_winrate(ind_rows, "行业"))
lines.append("\n\n")

# 2. 整体市值层胜率
lines.append("## 二、市值层胜率 (全市场)\n\n")
mv_rows = []
for mv in MV_ORDER:
    bucket = [s for s in samples if mv_bucket(s.get("total_mv")) == mv]
    if len(bucket) < 200: continue
    row = {"key": mv, "n": len(bucket)}
    for h in HOLD: row[f"wr{h}"] = winrate(bucket, h)
    mv_rows.append(row)
lines.append(group_table_winrate(mv_rows, "市值"))
lines.append("\n\n")

# 3. 整体 PE 层胜率
lines.append("## 三、PE 层胜率 (全市场)\n\n")
pe_rows = []
for pe in PE_ORDER:
    bucket = [s for s in samples if pe_bucket(s.get("pe")) == pe]
    if len(bucket) < 200: continue
    row = {"key": pe, "n": len(bucket)}
    for h in HOLD: row[f"wr{h}"] = winrate(bucket, h)
    pe_rows.append(row)
lines.append(group_table_winrate(pe_rows, "PE"))
lines.append("\n\n")

# 4. 8 因子分桶胜率 (全市场)
lines.append("## 四、各因子分桶胜率 (全市场, 加粗 = 该列最高)\n\n")
for factor in FACTORS:
    vals = sorted({s.get(factor) for s in samples if s.get(factor) is not None})
    if not vals: continue
    if len(vals) > 10:  # 连续因子, 跳过(用区间桶 — 简化为分位数)
        continue
    lines.append(f"### {factor}\n\n")
    rows = []
    for v in vals:
        bucket = [s for s in samples if s.get(factor) == v]
        if len(bucket) < 100: continue
        row = {"key": f"{v:+.1f}", "n": len(bucket)}
        for h in HOLD: row[f"wr{h}"] = winrate(bucket, h)
        rows.append(row)
    lines.append(group_table_winrate(rows, "因子分"))
    lines.append("\n\n")

# 5. Top 强信号"行业 × 因子"组合的胜率展开
lines.append("## 五、Top 强信号 (行业 × 因子) 5 持有期胜率展开\n\n")
# 重算强组合
strong = []
for factor in FACTORS:
    for ind in MAJOR_INDS:
        bucket = [s for s in major if s["industry"] == ind]
        if len(bucket) < 200: continue
        ic20 = ic_of(bucket, factor, 20)
        if ic20 is None: continue
        if abs(ic20) >= 0.04:
            strong.append((ind, factor, ic20, len(bucket)))
strong.sort(key=lambda x: -abs(x[2]))
print(f"强组合 ≥0.04: {len(strong)}")

for idx, (ind, factor, ic20, n) in enumerate(strong[:20]):
    bucket = [s for s in major if s["industry"] == ind]
    factor_vals = sorted({s.get(factor) for s in bucket if s.get(factor) is not None})
    direction = "正向" if ic20 > 0 else "反向"
    lines.append(f"### {idx+1}. {ind} × {factor} | IC(20d)={ic20:+.4f} ({direction}) | n={n:,}\n\n")

    # 5.1 该 (行业, 因子) 的因子分桶胜率
    rows = []
    for v in factor_vals:
        sub = [s for s in bucket if s.get(factor) == v]
        if len(sub) < 50: continue
        row = {"key": f"{v:+.1f}", "n": len(sub)}
        for h in HOLD: row[f"wr{h}"] = winrate(sub, h)
        rows.append(row)
    lines.append("**因子分桶胜率**:\n\n")
    lines.append(group_table_winrate(rows, "因子分"))
    lines.append("\n\n")

    # 5.2 该 (行业, 因子) 内市值再分层胜率
    lines.append("**市值再分层胜率** (该行业内):\n\n")
    sub_rows = []
    for mv in MV_ORDER:
        sub = [s for s in bucket if mv_bucket(s.get("total_mv")) == mv]
        if len(sub) < 100: continue
        row = {"key": mv, "n": len(sub)}
        for h in HOLD: row[f"wr{h}"] = winrate(sub, h)
        sub_rows.append(row)
    if sub_rows:
        lines.append(group_table_winrate(sub_rows, "市值"))
    else:
        lines.append("(样本不足)")
    lines.append("\n\n")

    # 5.3 该 (行业, 因子) 内 PE 再分层胜率
    lines.append("**PE 再分层胜率** (该行业内):\n\n")
    sub_rows = []
    for pe in PE_ORDER:
        sub = [s for s in bucket if pe_bucket(s.get("pe")) == pe]
        if len(sub) < 100: continue
        row = {"key": pe, "n": len(sub)}
        for h in HOLD: row[f"wr{h}"] = winrate(sub, h)
        sub_rows.append(row)
    if sub_rows:
        lines.append(group_table_winrate(sub_rows, "PE"))
    else:
        lines.append("(样本不足)")
    lines.append("\n\n---\n\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"\n报告: {OUT}")
