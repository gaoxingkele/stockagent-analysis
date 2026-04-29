"""给 3 年样本补 total_mv/pe/pe_ttm/pb 字段。

策略: 复用 1 年的 daily_basic_cache (覆盖 2025-04~2026-04 部分日期),
      只补缺失日期。
"""
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
import tushare as ts
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

DIR_3Y = Path("output/backtest_3y_2023_2026")
GR_DIR = DIR_3Y / "group_results"
CACHE_3Y = DIR_3Y / "daily_basic_cache.json"
CACHE_1Y = Path("output/backtest_factors_2026_04/daily_basic_cache.json")

# ─── Step 1: 收集所有 trade_dates ───
print("[1/3] 扫所有样本收集 trade_dates...")
all_dates = set()
for gf in sorted(GR_DIR.glob("group_*.jsonl")):
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
                if s.get("trade_date"):
                    all_dates.add(s["trade_date"])
            except: pass
all_dates = sorted(all_dates)
print(f"  {len(all_dates)} 个交易日 ({all_dates[0]} ~ {all_dates[-1]})")

# 加载已有缓存 (1y + 3y)
cache = {}
if CACHE_3Y.exists():
    cache = json.load(open(CACHE_3Y, encoding="utf-8"))
    print(f"  3y 缓存已有 {len(cache)} 天")
if CACHE_1Y.exists():
    cache_1y = json.load(open(CACHE_1Y, encoding="utf-8"))
    for d, v in cache_1y.items():
        if d not in cache and v:
            cache[d] = v
    print(f"  合并 1y 缓存后: {len(cache)} 天")

todo = [d for d in all_dates if d not in cache]
print(f"[2/3] 拉缺失 {len(todo)} 天 daily_basic...")
RATE = 0.4

for i, td in enumerate(todo):
    try:
        df = pro.daily_basic(trade_date=td, fields="ts_code,trade_date,total_mv,pe,pe_ttm,pb")
        if df is None or df.empty:
            cache[td] = {}
        else:
            cache[td] = {r.ts_code: {
                "total_mv": float(r.total_mv) if r.total_mv else None,
                "pe": float(r.pe) if r.pe else None,
                "pe_ttm": float(r.pe_ttm) if r.pe_ttm else None,
                "pb": float(r.pb) if r.pb else None,
            } for r in df.itertuples()}
        time.sleep(RATE)
        if (i + 1) % 50 == 0:
            with open(CACHE_3Y, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"  进度 {i+1}/{len(todo)}, 累计 {sum(len(v) for v in cache.values()):,} 条")
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "频率" in msg:
            time.sleep(8); continue
        cache[td] = {}

with open(CACHE_3Y, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)
print(f"  缓存写入: {sum(len(v) for v in cache.values()):,} 条")

# ─── Step 3: 给样本补字段 ───
print("[3/3] 给样本补字段...")
total = 0; enriched = 0
for gf in sorted(GR_DIR.glob("group_*.jsonl")):
    out_lines = []
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
            except: continue
            ts_code = s.get("ts_code"); td = s.get("trade_date")
            day = cache.get(td, {}).get(ts_code)
            if day:
                s["total_mv"] = day.get("total_mv")
                s["pe"] = day.get("pe")
                s["pe_ttm"] = day.get("pe_ttm")
                s["pb"] = day.get("pb")
                enriched += 1
            total += 1
            out_lines.append(json.dumps(s, ensure_ascii=False))
    with open(gf, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out_lines) + "\n")
print(f"DONE: {enriched}/{total} ({enriched/total*100:.1f}%) 样本已补字段")
