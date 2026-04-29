"""按交易日批量拉 daily_basic, 给现有 samples 补 total_mv 和 pe 字段。

策略: 按 trade_date 批量调 (一天 1 call, 拿全市场), 比按 ts_code 逐个快 20+ 倍。
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import tushare as ts
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

OUTPUT_DIR = ROOT / "output" / "backtest_factors_2026_04"
GR_DIR = OUTPUT_DIR / "group_results"
CACHE_FILE = OUTPUT_DIR / "daily_basic_cache.json"

# ─── Step 1: 收集所有需要的 trade_date ───
print("[1/3] 扫所有样本收集 trade_dates...")
all_dates = set()
for gf in sorted(GR_DIR.glob("group_*.jsonl")):
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
                if s.get("trade_date"):
                    all_dates.add(s["trade_date"])
            except Exception:
                pass
all_dates = sorted(all_dates)
print(f"  共 {len(all_dates)} 个交易日")

# ─── Step 2: 按日批量拉 daily_basic, 缓存 ───
cache = {}
if CACHE_FILE.exists():
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"[2/3] 缓存已有 {len(cache)} 天")

todo = [d for d in all_dates if d not in cache]
print(f"[2/3] 待拉 {len(todo)} 天")
RATE_SLEEP = 0.4
for i, td in enumerate(todo):
    try:
        df = pro.daily_basic(trade_date=td,
                             fields="ts_code,trade_date,total_mv,pe,pe_ttm,pb")
        if df is None or df.empty:
            cache[td] = {}
        else:
            d = {}
            for r in df.itertuples():
                d[r.ts_code] = {
                    "total_mv": float(r.total_mv) if r.total_mv else None,
                    "pe": float(r.pe) if r.pe else None,
                    "pe_ttm": float(r.pe_ttm) if r.pe_ttm else None,
                    "pb": float(r.pb) if r.pb else None,
                }
            cache[td] = d
        if (i + 1) % 20 == 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"  进度 {i+1}/{len(todo)}, 累计 {sum(len(v) for v in cache.values()):,} 条")
        time.sleep(RATE_SLEEP)
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "limit" in msg or "频率" in msg:
            time.sleep(8)
            try:
                df = pro.daily_basic(trade_date=td,
                                    fields="ts_code,trade_date,total_mv,pe,pe_ttm,pb")
                if df is not None and not df.empty:
                    d = {r.ts_code: {
                        "total_mv": float(r.total_mv) if r.total_mv else None,
                        "pe": float(r.pe) if r.pe else None,
                        "pe_ttm": float(r.pe_ttm) if r.pe_ttm else None,
                        "pb": float(r.pb) if r.pb else None,
                    } for r in df.itertuples()}
                    cache[td] = d
            except Exception:
                cache[td] = {}
        else:
            print(f"  [err] {td}: {e}")
            cache[td] = {}

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)
print(f"[2/3] daily_basic 缓存写入: {sum(len(v) for v in cache.values()):,} 条")

# ─── Step 3: 给所有 samples 补 total_mv / pe 字段 ───
print("[3/3] 给样本补 total_mv / pe ...")
total_samples = 0
enriched = 0
for gf in sorted(GR_DIR.glob("group_*.jsonl")):
    out_lines = []
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
            except Exception:
                continue
            ts_code = s.get("ts_code")
            td = s.get("trade_date")
            day_data = cache.get(td, {}).get(ts_code)
            if day_data:
                s["total_mv"] = day_data.get("total_mv")
                s["pe"] = day_data.get("pe")
                s["pe_ttm"] = day_data.get("pe_ttm")
                s["pb"] = day_data.get("pb")
                enriched += 1
            total_samples += 1
            out_lines.append(json.dumps(s, ensure_ascii=False))
    with open(gf, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out_lines) + "\n")
print(f"DONE: {enriched}/{total_samples} ({enriched/total_samples*100:.1f}%) 样本已补字段")
