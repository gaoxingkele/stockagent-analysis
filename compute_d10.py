"""补算 D+10 收益与回撤, 用现有 samples + raw_data 即可, 不重拉数据。"""
import json
from pathlib import Path

ROOT = Path("output/backtest_factors_2026_04")
RAW_DIR = ROOT / "raw_data"
GR_DIR = ROOT / "group_results"

# 加载 universe 拿 group 划分
with open(ROOT / "universe.json", encoding="utf-8") as f:
    uni = json.load(f)
stock_to_group = {s["ts_code"]: s["group_id"] for s in uni["stocks"]}

# 缓存: ts_code -> {date: (close, idx)} for D+10 计算
def load_stock_daily(ts_code):
    f = RAW_DIR / f"{ts_code}.json"
    if not f.exists(): return None
    with open(f, encoding="utf-8") as fp:
        d = json.load(fp)
    daily = (d.get("ts") or {}).get("daily") or []
    return [(r.get("trade_date"), float(r.get("close") or 0)) for r in daily]


def safe_float(v):
    try:
        f = float(v)
        if f != f or f == float('inf') or f == float('-inf'):
            return None
        return f
    except (TypeError, ValueError):
        return None


def compute_d10(daily_closes, target_date):
    """返回 (r10, dd10) 或 (None, None)。"""
    idx_t = None
    for i, (d, c) in enumerate(daily_closes):
        if d == target_date:
            idx_t = i; break
    if idx_t is None: return None, None
    if idx_t + 10 >= len(daily_closes): return None, None
    close_t = daily_closes[idx_t][1]
    if close_t <= 0: return None, None
    future = [c for d, c in daily_closes[idx_t+1:idx_t+11] if c > 0]
    if len(future) < 10: return None, None
    end_close = future[-1]
    r10 = (end_close - close_t) / close_t * 100
    peak = close_t; max_dd = 0.0
    for c in future:
        peak = max(peak, c)
        dd = (c - peak) / peak * 100
        if dd < max_dd: max_dd = dd
    return r10, max_dd


# 处理每组, 写新文件
total = 0
written = 0
for gf in sorted(GR_DIR.glob("group_*.jsonl")):
    out_lines = []
    daily_cache = {}
    with open(gf, encoding="utf-8") as fp:
        for line in fp:
            try:
                s = json.loads(line)
            except Exception:
                continue
            ts_code = s.get("ts_code")
            td = s.get("trade_date")
            if ts_code not in daily_cache:
                daily_cache[ts_code] = load_stock_daily(ts_code) or []
            dc = daily_cache[ts_code]
            r10, dd10 = compute_d10(dc, td)
            if r10 is not None:
                s["r10"] = r10; s["dd10"] = dd10
                written += 1
            total += 1
            out_lines.append(json.dumps(s, ensure_ascii=False))
    # 覆盖原文件
    with open(gf, "w", encoding="utf-8") as fp:
        fp.write("\n".join(out_lines) + "\n")
    print(f"{gf.name}: {len(out_lines)} 行, D+10 命中 {written/total*100:.1f}% (累计)")

print(f"\nDONE: {written}/{total} 样本补充 D+10 ({written/total*100:.1f}%)")
