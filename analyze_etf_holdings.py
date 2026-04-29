"""ETF 持仓股的市值/PE/行业维度分析。

数据源:
  - winner ETF 列表 + 持仓: output/etf_analysis/winners_relaxed.json + holdings/
  - 股票样本 (含 mv/pe/r5/r10/r20/r30/r40): output/backtest_factors_2026_04/
  - 行业映射: universe.json

输出: output/etf_analysis/report_holdings_analysis.md
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent
ETF_DIR = ROOT / "output" / "etf_analysis"
SAMPLE_DIR = ROOT / "output" / "backtest_factors_2026_04"
OUT = ETF_DIR / "report_holdings_analysis.md"

# ─── 加载 ETF 持仓 ───
with open(ETF_DIR / "winners_relaxed.json", encoding="utf-8") as f:
    winners = json.load(f)
print(f"winner ETF: {len(winners)}")

etf_holdings = {}  # ts_code -> list of holding records
for etf in winners:
    code = etf["ts_code"]
    f = ETF_DIR / "holdings" / f"{code}.json"
    if not f.exists(): continue
    rows = json.loads(f.read_text(encoding="utf-8"))
    if rows:
        etf_holdings[code] = rows

print(f"有持仓数据的 ETF: {len(etf_holdings)}")

# ─── 收集所有持仓股 (各 ETF 各季度并集) ───
all_holdings = defaultdict(list)  # symbol -> list of (etf_code, end_date, ratio)
for etf_code, rows in etf_holdings.items():
    for r in rows:
        sym = r.get("symbol")
        if sym:
            all_holdings[sym].append({
                "etf": etf_code, "end_date": r["end_date"],
                "ratio": r.get("stk_mkv_ratio"),
            })

# 持仓股池: 至少出现一次的股票 (转 ts_code)
def to_ts_code(sym):
    if not sym: return None
    s = str(sym).strip()
    if "." in s: return s
    if len(s) == 6:
        if s.startswith("6"): return f"{s}.SH"
        if s.startswith("0") or s.startswith("3"): return f"{s}.SZ"
        if s.startswith("8") or s.startswith("4") or s.startswith("9"): return f"{s}.BJ"
    return None

held_stocks = set()
for sym in all_holdings:
    ts = to_ts_code(sym)
    if ts: held_stocks.add(ts)
print(f"被持仓的股票总数 (各 ETF 各季度并集): {len(held_stocks)}")

# ─── 加载 universe + 样本 ───
with open(SAMPLE_DIR / "universe.json", encoding="utf-8") as f:
    uni = json.load(f)
ts_to_ind = {s["ts_code"]: (s.get("industry") or "未分类") for s in uni["stocks"]}
ts_to_name = {s["ts_code"]: s.get("name", "") for s in uni["stocks"]}

print("加载样本...")
samples = []
for gf in sorted((SAMPLE_DIR / "group_results").glob("group_*.jsonl")):
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
                s["industry"] = ts_to_ind.get(s["ts_code"], "未分类")
                s["is_etf_held"] = s["ts_code"] in held_stocks
                samples.append(s)
            except: pass
print(f"样本: {len(samples):,}")
held_samples = [s for s in samples if s["is_etf_held"]]
print(f"持仓股样本: {len(held_samples):,}, 占 {len(held_samples)/len(samples)*100:.1f}%")

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
HOLD = [5, 10, 20, 30, 40]


def winrate(samples, h):
    rs = [s.get(f"r{h}") for s in samples if s.get(f"r{h}") is not None]
    if not rs: return None
    return float(np.mean([1 if r > 0 else 0 for r in rs])) * 100

def avg_ret(samples, h):
    rs = [s.get(f"r{h}") for s in samples if s.get(f"r{h}") is not None]
    if not rs: return None
    return float(np.mean(rs))


def fmt_winrate_row(samples, label, n):
    line = f"| {label} | {n:,} |"
    for h in HOLD:
        wr = winrate(samples, h)
        line += f" {wr:.1f}% |" if wr is not None else " - |"
    return line


# ─── Build report ───
lines = []
lines.append(f"# Winner ETF 持仓股的市值/PE/行业维度分析\n\n")
lines.append(f"> ETF 候选 (年化 ≥30% 或 Q1 ≥15%): {len(winners)} 只\n")
lines.append(f"> 有股票持仓数据的: {len(etf_holdings)} 只 (黄金 ETF 持实物黄金, 无个股)\n")
lines.append(f"> 被持仓股票池 (各 ETF 各季度并集): {len(held_stocks)} 只\n")
lines.append(f"> 1 年样本 (2025-04 ~ 2026-04): {len(samples):,}, 持仓股样本: {len(held_samples):,}\n\n")
lines.append("---\n\n")

# Section 1: ETF 列表
lines.append("## 一、Winner ETF 清单 (24 只)\n\n")
lines.append("| 代码 | 名称 | 类型 | 年化 | 全期 | Q1 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
for w in sorted(winners, key=lambda x: -x["annual_return_pct"]):
    q1 = f"{w['q1_2026_return_pct']:+.1f}%" if w['q1_2026_return_pct'] is not None else "-"
    has_hold = "✓" if w["ts_code"] in etf_holdings else " "
    lines.append(f"| {w['ts_code']} | {has_hold} {w['name']} | {w['candidate_type']} | {w['annual_return_pct']:+.1f}% | {w['total_return_pct']:+.1f}% | {q1} |\n")
lines.append("\n> ✓ = 有股票持仓数据 (黄金 ETF 因持实物无 stock holdings)\n\n")

# Section 2: 8 只有持仓数据的 ETF 持仓汇总
lines.append("## 二、8 只股票型 ETF 的持仓股汇总\n\n")
for etf_code, rows in etf_holdings.items():
    etf_meta = next((w for w in winners if w["ts_code"] == etf_code), {})
    syms = sorted({r["symbol"] for r in rows if r.get("symbol")})
    n_hold = len(syms)
    quarters = sorted({r["end_date"] for r in rows})
    lines.append(f"### {etf_code} {etf_meta.get('name', '')}\n")
    lines.append(f"- 年化收益: **{etf_meta.get('annual_return_pct', 0):+.1f}%**, 全期 {etf_meta.get('total_return_pct', 0):+.1f}%, Q1 {etf_meta.get('q1_2026_return_pct') or '-'}%\n")
    lines.append(f"- 报告期数: {len(quarters)} 个季度\n")
    lines.append(f"- 持仓股数 (并集): {n_hold} 只\n")
    # Top 持仓 (按最近一期 ratio 排序)
    latest = max(quarters)
    latest_rows = sorted([r for r in rows if r["end_date"] == latest],
                          key=lambda r: -(r.get("stk_mkv_ratio") or 0))[:10]
    if latest_rows:
        lines.append(f"- 最近期 ({latest}) 前 10 大持仓:\n")
        for r in latest_rows:
            ts = to_ts_code(r["symbol"])
            name = ts_to_name.get(ts, "")
            ind = ts_to_ind.get(ts, "")
            ratio = r.get("stk_mkv_ratio") or 0
            lines.append(f"  - {r['symbol']} {name} ({ind}) - {ratio:.2f}%\n")
    lines.append("\n")

# Section 3: 持仓股 vs 非持仓股 vs 全市场基线
lines.append("## 三、持仓股 vs 非持仓股 vs 全市场 — D+5/10/20/30/40 胜率\n\n")
lines.append("| 类别 | 样本 |" + "".join(f" D+{h} 胜率 |" for h in HOLD) + "\n")
lines.append("|---|---|" + "---|" * len(HOLD) + "\n")
non_held = [s for s in samples if not s["is_etf_held"]]
for label, pool in [("ETF 持仓股", held_samples),
                    ("非持仓股", non_held),
                    ("全市场", samples)]:
    lines.append(fmt_winrate_row(pool, label, len(pool)) + "\n")
lines.append("\n")

# Section 4: 持仓股按行业
lines.append("## 四、持仓股按行业 — D+10 / D+40 胜率\n\n")
lines.append("| 行业 | 持仓股数 | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|---|\n")
ind_data = []
for ind in set(ts_to_ind.values()):
    in_ind_held = [s for s in held_samples if s["industry"] == ind]
    if len(in_ind_held) < 100: continue
    n_stocks = len({s["ts_code"] for s in in_ind_held})
    ind_data.append({
        "ind": ind, "n_stocks": n_stocks, "n": len(in_ind_held),
        "wr10": winrate(in_ind_held, 10), "ret10": avg_ret(in_ind_held, 10),
        "wr40": winrate(in_ind_held, 40), "ret40": avg_ret(in_ind_held, 40),
    })
ind_data.sort(key=lambda x: -(x["wr40"] or 0))
for d in ind_data:
    lines.append(f"| {d['ind']} | {d['n_stocks']} | {d['n']:,} |"
                 f" {d['wr10']:.1f}% | {d['ret10']:+.2f}% |"
                 f" **{d['wr40']:.1f}%** | {d['ret40']:+.2f}% |\n")
lines.append("\n")

# Section 5: 持仓股按市值
lines.append("## 五、持仓股按市值 — D+10 / D+40 胜率\n\n")
lines.append("| 市值 | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
mv_data = []
for mv in MV_ORDER:
    sub = [s for s in held_samples if mv_bucket(s.get("total_mv")) == mv]
    if len(sub) < 50: continue
    mv_data.append({
        "key": mv, "n": len(sub),
        "wr10": winrate(sub, 10), "ret10": avg_ret(sub, 10),
        "wr40": winrate(sub, 40), "ret40": avg_ret(sub, 40),
    })
max_wr40 = max((d["wr40"] or 0) for d in mv_data)
for d in mv_data:
    bold40 = "**" if d["wr40"] == max_wr40 else ""
    lines.append(f"| {d['key']} | {d['n']:,} | {d['wr10']:.1f}% | {d['ret10']:+.2f}% | {bold40}{d['wr40']:.1f}%{bold40} | {d['ret40']:+.2f}% |\n")
lines.append("\n")

# Section 6: 持仓股按 PE
lines.append("## 六、持仓股按 PE — D+10 / D+40 胜率\n\n")
lines.append("| PE | 样本 | D+10 胜率 | D+10 涨幅 | D+40 胜率 | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|\n")
pe_data = []
for pe in PE_ORDER:
    sub = [s for s in held_samples if pe_bucket(s.get("pe")) == pe]
    if len(sub) < 50: continue
    pe_data.append({
        "key": pe, "n": len(sub),
        "wr10": winrate(sub, 10), "ret10": avg_ret(sub, 10),
        "wr40": winrate(sub, 40), "ret40": avg_ret(sub, 40),
    })
if pe_data:
    max_wr40 = max((d["wr40"] or 0) for d in pe_data)
    for d in pe_data:
        bold40 = "**" if d["wr40"] == max_wr40 else ""
        lines.append(f"| {d['key']} | {d['n']:,} | {d['wr10']:.1f}% | {d['ret10']:+.2f}% | {bold40}{d['wr40']:.1f}%{bold40} | {d['ret40']:+.2f}% |\n")
    lines.append("\n")

# Section 7: 持仓股 (市值 × PE) 二维 D+40 胜率
lines.append("## 七、持仓股 (市值 × PE) 二维 D+40 胜率\n\n")
lines.append("> 加粗 = 该交叉表内最高胜率\n\n")
lines.append("| 市值 \\ PE |" + "".join(f" {pe} |" for pe in PE_ORDER) + "\n")
lines.append("|---|" + "---|" * len(PE_ORDER) + "\n")
matrix = {}; max_wr = -1; max_key = None
for mv in MV_ORDER:
    for pe in PE_ORDER:
        sub = [s for s in held_samples if mv_bucket(s.get("total_mv")) == mv and pe_bucket(s.get("pe")) == pe]
        if len(sub) < 50:
            matrix[(mv, pe)] = (None, len(sub)); continue
        wr = winrate(sub, 40)
        matrix[(mv, pe)] = (wr, len(sub))
        if wr is not None and wr > max_wr: max_wr = wr; max_key = (mv, pe)
for mv in MV_ORDER:
    line = f"| **{mv}** |"
    for pe in PE_ORDER:
        wr, n = matrix.get((mv, pe), (None, 0))
        if wr is None: line += " - |"
        elif (mv, pe) == max_key: line += f" **{wr:.1f}% (n={n:,})** |"
        else: line += f" {wr:.1f}% (n={n:,}) |"
    lines.append(line + "\n")
lines.append("\n")

# Section 8: 持仓股个股 D+40 胜率 Top 20
lines.append("## 八、ETF 持仓股 D+40 胜率 Top 20\n\n")
stock_stats = defaultdict(list)
for s in held_samples:
    if s.get("r40") is not None:
        stock_stats[s["ts_code"]].append(s["r40"])
stock_wr40 = []
for ts, rs in stock_stats.items():
    if len(rs) < 50: continue
    wr = float(np.mean([1 if r > 0 else 0 for r in rs])) * 100
    avg = float(np.mean(rs))
    held_in = [code for code, hrows in etf_holdings.items() if any(to_ts_code(r["symbol"]) == ts for r in hrows)]
    stock_wr40.append({
        "ts_code": ts, "name": ts_to_name.get(ts, ""),
        "industry": ts_to_ind.get(ts), "n": len(rs),
        "wr40": wr, "avg40": avg,
        "etf_count": len(held_in),
    })
stock_wr40.sort(key=lambda x: -x["wr40"])

lines.append("| # | 代码 | 名称 | 行业 | ETF 数 | 样本 | **D+40 胜率** | D+40 涨幅 |\n")
lines.append("|---|---|---|---|---|---|---|---|\n")
for i, st in enumerate(stock_wr40[:20]):
    lines.append(f"| {i+1} | {st['ts_code']} | {st['name']} | {st['industry']} | {st['etf_count']} | {st['n']:,} | **{st['wr40']:.1f}%** | {st['avg40']:+.2f}% |\n")
lines.append("\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"\n报告: {OUT}")
print(f"持仓股池: {len(held_stocks)}, 持仓样本: {len(held_samples):,}")
if stock_wr40:
    print(f"持仓股 D+40 胜率最高: {stock_wr40[0]['name']} ({stock_wr40[0]['ts_code']}) {stock_wr40[0]['wr40']:.1f}%")
