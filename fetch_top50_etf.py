"""扩展到 Top 50 ETF (按全期年化收益排序)。

复用 nav 缓存, 不重新拉。补拉缺失的 ETF 持仓数据。
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import tushare as ts
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

ETF_DIR = ROOT / "output" / "etf_analysis"
NAV_DIR = ETF_DIR / "nav"
HOLD_DIR = ETF_DIR / "holdings"
LIST_FILE = ETF_DIR / "etf_list.json"
TOP50_FILE = ETF_DIR / "top50.json"

with open(LIST_FILE, encoding="utf-8") as f:
    etf_list = json.load(f)

print(f"[1/3] 计算所有 ETF 年化收益...")
ranked = []
for etf in etf_list:
    code = etf["ts_code"]
    cache = NAV_DIR / f"{code}.json"
    if not cache.exists(): continue
    try:
        rows = json.loads(cache.read_text(encoding="utf-8"))
    except: continue
    if len(rows) < 100: continue
    valid = [r for r in rows if (r.get("adj_nav") or r.get("unit_nav"))]
    if len(valid) < 100: continue
    first_v = valid[0].get("adj_nav") or valid[0].get("unit_nav")
    last_v = valid[-1].get("adj_nav") or valid[-1].get("unit_nav")
    first_d = valid[0]["nav_date"]; last_d = valid[-1]["nav_date"]
    d1 = (int(last_d[:4]) - int(first_d[:4])) + (int(last_d[4:6]) - int(first_d[4:6]))/12
    if d1 < 0.5: continue
    annual = (last_v / first_v) ** (1.0 / d1) - 1
    total = last_v / first_v - 1

    # Q1 2026 涨幅
    Q1S, Q1E = "20260101", "20260331"
    q1_s = q1_e = None
    for r in valid:
        nv = r.get("adj_nav") or r.get("unit_nav")
        if r["nav_date"] >= Q1S and q1_s is None: q1_s = nv
        if r["nav_date"] <= Q1E: q1_e = nv
    q1_ret = (q1_e/q1_s - 1) if (q1_s and q1_e and q1_s > 0) else None

    # 2024-09-24 起的 924 行情段
    Q924, Q924E = "20240924", "20260331"
    g_s = g_e = None; g_sd = g_ed = None
    for r in valid:
        nv = r.get("adj_nav") or r.get("unit_nav")
        if r["nav_date"] >= Q924 and g_s is None: g_s = nv; g_sd = r["nav_date"]
        if r["nav_date"] <= Q924E: g_e = nv; g_ed = r["nav_date"]
    g_ret = (g_e/g_s - 1) if (g_s and g_e and g_s > 0) else None

    ranked.append({
        **etf,
        "first_date": first_d, "last_date": last_d, "years": round(d1, 2),
        "total_return_pct": round(total*100, 2),
        "annual_return_pct": round(annual*100, 2),
        "q1_2026_return_pct": round(q1_ret*100, 2) if q1_ret else None,
        "since_924_return_pct": round(g_ret*100, 2) if g_ret else None,
    })

ranked.sort(key=lambda x: -x["annual_return_pct"])
top50 = ranked[:50]

TOP50_FILE.write_text(json.dumps(top50, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[2/3] Top 50 (按年化排序):")
for i, c in enumerate(top50):
    q1 = f"{c['q1_2026_return_pct']:+.1f}%" if c['q1_2026_return_pct'] is not None else "-"
    g = f"{c['since_924_return_pct']:+.1f}%" if c['since_924_return_pct'] is not None else "-"
    print(f"  #{i+1:2d} {c['ts_code']:12s} {c['name'][:28]:28s} 年化 {c['annual_return_pct']:>+6.1f}% | Q1 {q1:>7s} | 924 {g:>7s}")

# 补拉 Top 50 中缺失的 holdings
print(f"\n[3/3] 补拉 Top 50 持仓数据...")
new_fetch = 0
for i, etf in enumerate(top50):
    code = etf["ts_code"]
    cache = HOLD_DIR / f"{code}.json"
    if cache.exists():
        try:
            existing = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                continue
        except: pass
    try:
        df = pro.fund_portfolio(ts_code=code)
        time.sleep(0.5)
        if df is None or df.empty:
            cache.write_text("[]", encoding="utf-8"); continue
        df = df[df["end_date"] >= "20230101"]
        rows = [{
            "end_date": r.end_date, "ann_date": r.ann_date,
            "symbol": r.symbol,
            "mkv": float(r.mkv) if r.mkv else None,
            "amount": float(r.amount) if r.amount else None,
            "stk_mkv_ratio": float(r.stk_mkv_ratio) if r.stk_mkv_ratio else None,
        } for r in df.itertuples()]
        cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        new_fetch += 1
        if (i+1) % 10 == 0:
            print(f"  补拉进度 {i+1}/{len(top50)}, 新增 {new_fetch}")
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg: time.sleep(8); continue
        cache.write_text("[]", encoding="utf-8")

print(f"\nDONE. Top 50 写入 {TOP50_FILE}, 新拉持仓 {new_fetch} 只")

# 统计有持仓数据的 ETF 数量
n_with_holdings = sum(1 for etf in top50
                       if (HOLD_DIR / f"{etf['ts_code']}.json").exists()
                       and json.loads((HOLD_DIR / f"{etf['ts_code']}.json").read_text(encoding="utf-8")))
print(f"Top 50 中有股票持仓数据的: {n_with_holdings}")
