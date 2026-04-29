"""Top 50 ETF 的持仓股在 924 行情段 (2024-09-24 起) 的多维分析。

关键差别:
  - ETF 池: 上次只有 24 只 winner, 这次 Top 50 (按全期年化排序)
  - 31 只有股票持仓 (vs 上次 8 只), 持仓股池更全
  - 时间窗口: 924 行情后 (vs 上次 1 年)
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
ETF_DIR = ROOT / "output" / "etf_analysis"
SAMPLE_DIR = ROOT / "output" / "backtest_3y_2023_2026"
OUT = ETF_DIR / "report_top50_924.md"
START_924 = "20240924"

# Top 50 ETF
with open(ETF_DIR / "top50.json", encoding="utf-8") as f:
    top50 = json.load(f)
print(f"Top 50 ETF: {len(top50)}")

# 收集所有持仓
def to_ts_code(sym):
    if not sym: return None
    s = str(sym).strip()
    if "." in s: return s
    if len(s) == 6:
        if s.startswith("6"): return f"{s}.SH"
        if s.startswith("0") or s.startswith("3"): return f"{s}.SZ"
    return None

etf_holdings = {}
for etf in top50:
    code = etf["ts_code"]
    f = ETF_DIR / "holdings" / f"{code}.json"
    if not f.exists(): continue
    rows = json.loads(f.read_text(encoding="utf-8"))
    if rows:
        etf_holdings[code] = rows

print(f"有持仓的 ETF: {len(etf_holdings)}")

# 持仓股集合 (并集)
held_stocks = set()
stock_etf_count = defaultdict(int)
for etf_code, rows in etf_holdings.items():
    seen_stocks = set()
    for r in rows:
        ts = to_ts_code(r.get("symbol"))
        if ts:
            held_stocks.add(ts)
            seen_stocks.add(ts)
    for ts in seen_stocks:
        stock_etf_count[ts] += 1
print(f"被持仓股票数: {len(held_stocks)}")

# 加载 universe + 样本 (只加载 924 起)
with open(SAMPLE_DIR / "universe.json", encoding="utf-8") as fp:
    uni = json.load(fp)
ts_to_ind = {s["ts_code"]: (s.get("industry") or "未分类") for s in uni["stocks"]}
ts_to_name = {s["ts_code"]: s.get("name", "") for s in uni["stocks"]}

print(f"加载 924 起样本...")
samples = []
for gf in sorted((SAMPLE_DIR / "group_results").glob("group_*.jsonl")):
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
                if s.get("trade_date", "") < START_924: continue
                # 极端剔除
                if s.get("r40") is not None and abs(s["r40"]) > 50: continue
                s["industry"] = ts_to_ind.get(s["ts_code"], "未分类")
                s["is_held"] = s["ts_code"] in held_stocks
                s["etf_count"] = stock_etf_count.get(s["ts_code"], 0)
                samples.append(s)
            except: pass
print(f"样本: {len(samples):,}, 持仓股样本: {sum(1 for s in samples if s['is_held']):,}")

held = [s for s in samples if s["is_held"]]
non_held = [s for s in samples if not s["is_held"]]

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

# Build report
lines = []
lines.append(f"# Top 50 ETF 持仓股 × 924 行情段 多维分析\n\n")
lines.append(f"> ETF 池: Top 50 (按全期年化排序), 31 只有股票持仓\n")
lines.append(f"> 持仓股池: {len(held_stocks)} 只 (并集)\n")
lines.append(f"> 样本: 924 行情后 (2024-09-24 起), 已剔除 |D+40|>50% 极端\n")
lines.append(f"> 总样本: {len(samples):,}, 持仓股样本: {len(held):,} ({len(held)/len(samples)*100:.1f}%)\n\n")

# 1. 基线
lines.append("## 一、持仓股 vs 非持仓股 vs 全市场\n\n")
lines.append("| 类别 | 样本 | D+5 胜率 | D+10 胜率 | D+20 胜率 | D+30 胜率 | D+40 胜率 |\n")
lines.append("|---|---|---|---|---|---|---|\n")
for label, pool in [("ETF 持仓股 (Top 50)", held), ("非持仓股", non_held), ("全市场", samples)]:
    line = f"| {label} | {len(pool):,} |"
    wrs = [winrate(pool, h) for h in HOLD]
    rets = [avg_ret(pool, h) for h in HOLD]
    for w, r in zip(wrs, rets):
        line += f" {w:.1f}% ({r:+.2f}%) |" if w is not None else " - |"
    lines.append(line + "\n")
lines.append("\n")

# 2. 持仓密度分层 (被多少 ETF 持有)
lines.append("## 二、按持仓密度 D+10 / D+40 胜率\n\n")
lines.append("> ETF 数 = 多少只 winner ETF 持有该股\n\n")
lines.append("| ETF 数 | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
density_buckets = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-9", 5, 9), ("10+", 10, 999)]
def _fmt(v, suffix="%"):
    return f"{v:.1f}{suffix}" if v is not None else "-"
def _fmt_ret(v):
    return f"{v:+.2f}%" if v is not None else "-"
for name, lo, hi in density_buckets:
    sub = [s for s in held if lo <= s["etf_count"] <= hi]
    if len(sub) < 100: continue
    w10 = winrate(sub, 10); r10 = avg_ret(sub, 10)
    w40 = winrate(sub, 40); r40 = avg_ret(sub, 40)
    line = f"| {name} | {len(sub):,} | {_fmt(w10)} | {_fmt_ret(r10)} | "
    line += f"**{_fmt(w40)}** | {_fmt_ret(r40)} |"
    lines.append(line + "\n")
lines.append("\n")

# 3. 持仓股按行业
lines.append("## 三、持仓股按行业 D+10 / D+40 胜率 (Top 30)\n\n")
lines.append("| 行业 | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
ind_data = []
for ind in set(s["industry"] for s in held):
    sub = [s for s in held if s["industry"] == ind]
    if len(sub) < 50: continue
    ind_data.append({
        "ind": ind, "n": len(sub),
        "wr10": winrate(sub, 10), "ret10": avg_ret(sub, 10),
        "wr40": winrate(sub, 40), "ret40": avg_ret(sub, 40),
    })
ind_data.sort(key=lambda x: -(x["wr40"] or 0))
for d in ind_data[:30]:
    lines.append(f"| {d['ind']} | {d['n']:,} | {_fmt(d['wr10'])} | {_fmt_ret(d['ret10'])} | **{_fmt(d['wr40'])}** | {_fmt_ret(d['ret40'])} |\n")
lines.append("\n")

# 4. 持仓股按市值
lines.append("## 四、持仓股按市值 D+10 / D+40 胜率\n\n")
lines.append("| 市值 | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
for mv in MV_ORDER:
    sub = [s for s in held if mv_bucket(s.get("total_mv")) == mv]
    if len(sub) < 50: continue
    lines.append(f"| {mv} | {len(sub):,} | {_fmt(winrate(sub, 10))} | {_fmt_ret(avg_ret(sub, 10))} | **{_fmt(winrate(sub, 40))}** | {_fmt_ret(avg_ret(sub, 40))} |\n")
lines.append("\n")

# 5. 持仓股按 PE
lines.append("## 五、持仓股按 PE D+10 / D+40 胜率\n\n")
lines.append("| PE | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
for pe in PE_ORDER:
    sub = [s for s in held if pe_bucket(s.get("pe")) == pe]
    if len(sub) < 50: continue
    lines.append(f"| {pe} | {len(sub):,} | {_fmt(winrate(sub, 10))} | {_fmt_ret(avg_ret(sub, 10))} | **{_fmt(winrate(sub, 40))}** | {_fmt_ret(avg_ret(sub, 40))} |\n")
lines.append("\n")

# 6. (市值 × PE) 二维
lines.append("## 六、持仓股 (市值 × PE) 二维 D+40 胜率\n\n")
lines.append("| 市值 \\ PE |" + "".join(f" {pe} |" for pe in PE_ORDER) + "\n")
lines.append("|---|" + "---|" * len(PE_ORDER) + "\n")
mat = {}; max_v = -1; max_k = None
for mv in MV_ORDER:
    for pe in PE_ORDER:
        sub = [s for s in held if mv_bucket(s.get("total_mv")) == mv and pe_bucket(s.get("pe")) == pe]
        if len(sub) < 50:
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

# 7. Top 30 持仓股
lines.append("## 七、Top 30 持仓股 (D+40 胜率, 至少 50 样本)\n\n")
stock_stats = defaultdict(list)
for s in held:
    if s.get("r40") is not None:
        stock_stats[s["ts_code"]].append(s["r40"])
top_stocks = []
for ts, rs in stock_stats.items():
    if len(rs) < 50: continue
    wr = float(np.mean([1 if r > 0 else 0 for r in rs])) * 100
    avg = float(np.mean(rs))
    top_stocks.append({
        "ts": ts, "name": ts_to_name.get(ts, ""),
        "industry": ts_to_ind.get(ts), "n": len(rs),
        "wr40": wr, "avg40": avg,
        "etf_count": stock_etf_count.get(ts, 0),
    })
top_stocks.sort(key=lambda x: -x["wr40"])
lines.append("| # | 代码 | 名称 | 行业 | ETF 数 | 样本 | **D+40 胜率** | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|---|---|\n")
for i, st in enumerate(top_stocks[:30]):
    lines.append(f"| {i+1} | {st['ts']} | {st['name']} | {st['industry']} | {st['etf_count']} | {st['n']:,} | **{st['wr40']:.1f}%** | {st['avg40']:+.2f}% |\n")
lines.append("\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"\n报告: {OUT}")
print(f"持仓股最高胜率: {top_stocks[0]['name']} ({top_stocks[0]['ts']}) {top_stocks[0]['wr40']:.1f}%")
