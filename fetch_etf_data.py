"""ETF 数据拉取与筛选 (年化收益 ≥50%)。

Step 1: 拉 ETF 列表, 过滤
  - 上市日 ≤ 2023-01-04 (能跑全期)
  - 股票型或指数型 (排除货币、债券、QDII)
Step 2: 拉每只 ETF 的 fund_nav, 计算年化收益 (2023-01 ~ 2026-03)
Step 3: 筛年化 ≥50% 的, 拉持仓 (fund_portfolio)
Step 4: 输出 候选 ETF 清单 + 持仓时序
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

OUT_DIR = ROOT / "output" / "etf_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LIST_FILE = OUT_DIR / "etf_list.json"
NAV_DIR = OUT_DIR / "nav"; NAV_DIR.mkdir(exist_ok=True)
HOLD_DIR = OUT_DIR / "holdings"; HOLD_DIR.mkdir(exist_ok=True)
WINNERS_FILE = OUT_DIR / "winners.json"

START = "20230101"
END = "20260331"
TARGET_ANNUAL_RETURN = 0.50

pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

# ─── Step 1: 拉 ETF 列表 ───
print("[1/4] 拉 ETF 列表...")
df = pro.fund_basic(market="E", status="L")
print(f"  原始 ETF 总数: {len(df)}")

# 过滤
df = df[df["list_date"].notna()]
df = df[df["list_date"] <= "20230104"]   # 能跑全期
print(f"  上市日 ≤ 20230104: {len(df)}")

def _safe_str(v):
    if v is None: return ""
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return ""
    except: pass
    return str(v)

# 投资类型: 排除货币 / 债券 / QDII
def is_equity_or_index(row):
    inv = _safe_str(row.get("invest_type"))
    typ = _safe_str(row.get("fund_type"))
    name = _safe_str(row.get("name"))
    if "货币" in inv or "货币" in name: return False
    if "债券" in inv or "债" in inv: return False
    if "QDII" in inv: return False
    if "ETF" not in name and "etf" not in name: return False
    return True

df["keep"] = df.apply(is_equity_or_index, axis=1)
df = df[df["keep"]]
print(f"  股票/指数型 ETF: {len(df)}")

etf_list = df[["ts_code", "name", "management", "list_date",
                "fund_type", "invest_type", "benchmark"]].to_dict(orient="records")
LIST_FILE.write_text(json.dumps(etf_list, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  写入 {LIST_FILE}")


# ─── Step 2: 拉每只 ETF 的 fund_nav ───
print(f"\n[2/4] 拉 {len(etf_list)} 只 ETF 净值时序...")
candidates = []
for i, etf in enumerate(etf_list):
    code = etf["ts_code"]
    cache = NAV_DIR / f"{code}.json"
    if cache.exists():
        try:
            with open(cache, encoding="utf-8") as f:
                rows = json.load(f)
        except Exception:
            rows = []
    else:
        try:
            navdf = pro.fund_nav(ts_code=code, start_date=START, end_date=END)
            time.sleep(0.4)
            if navdf is None or navdf.empty:
                rows = []
            else:
                navdf = navdf.sort_values("nav_date")
                # adj_nav 是累计净值(含分红再投), 没有就用 unit_nav
                rows = []
                for r in navdf.itertuples():
                    rows.append({
                        "nav_date": r.nav_date,
                        "unit_nav": float(r.unit_nav) if r.unit_nav else None,
                        "adj_nav": float(r.adj_nav) if r.adj_nav else None,
                        "accum_nav": float(r.accum_nav) if hasattr(r, 'accum_nav') and r.accum_nav else None,
                    })
            cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "限" in msg or "频率" in msg:
                time.sleep(8)
                continue
            print(f"  [err] {code}: {e}")
            cache.write_text("[]", encoding="utf-8")
            rows = []

    # 计算年化
    if len(rows) >= 100:
        # 用 adj_nav 优先, 否则 unit_nav
        valid = [r for r in rows if (r.get("adj_nav") or r.get("unit_nav"))]
        if len(valid) >= 100:
            first_v = valid[0].get("adj_nav") or valid[0].get("unit_nav")
            last_v = valid[-1].get("adj_nav") or valid[-1].get("unit_nav")
            first_d = valid[0]["nav_date"]
            last_d = valid[-1]["nav_date"]
            try:
                d1 = (int(last_d[:4]) - int(first_d[:4])) + (int(last_d[4:6]) - int(first_d[4:6]))/12
                if d1 < 0.5: continue  # 数据不足
                annual = (last_v / first_v) ** (1.0 / d1) - 1
                total_ret = last_v / first_v - 1
                candidates.append({
                    **etf,
                    "first_date": first_d, "last_date": last_d,
                    "first_nav": first_v, "last_nav": last_v,
                    "years": round(d1, 2),
                    "total_return": round(total_ret * 100, 2),
                    "annual_return": round(annual * 100, 2),
                })
            except Exception:
                pass

    if (i + 1) % 50 == 0:
        winners = [c for c in candidates if c.get("annual_return", 0) >= TARGET_ANNUAL_RETURN * 100]
        print(f"  进度 {i+1}/{len(etf_list)}, 已算 {len(candidates)}, 候选 (年化≥50%) {len(winners)}")

# ─── Step 3: 筛选 winners ───
candidates.sort(key=lambda c: -c.get("annual_return", 0))
winners = [c for c in candidates if c.get("annual_return", 0) >= TARGET_ANNUAL_RETURN * 100]

print(f"\n[3/4] 筛选 winners")
print(f"  全部能算年化的 ETF: {len(candidates)}")
print(f"  年化 ≥ {TARGET_ANNUAL_RETURN*100:.0f}% 的 winners: {len(winners)}")

WINNERS_FILE.write_text(json.dumps(winners, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"  写入 {WINNERS_FILE}")

# 全部年化分布
top20 = candidates[:20]
print(f"\nTop 20 年化收益:")
for c in top20:
    print(f"  {c['ts_code']:12s} {c['name'][:30]:30s} 年化 {c['annual_return']:+.1f}% (全期 {c['total_return']:+.1f}%)")


# ─── Step 4: 拉 winners 的持仓 ───
print(f"\n[4/4] 拉 {len(winners)} 只 winner ETF 的持仓 (fund_portfolio)...")
for i, etf in enumerate(winners):
    code = etf["ts_code"]
    cache = HOLD_DIR / f"{code}.json"
    if cache.exists(): continue
    try:
        df = pro.fund_portfolio(ts_code=code)
        time.sleep(0.5)
        if df is None or df.empty:
            cache.write_text("[]", encoding="utf-8")
            continue
        # 只取 2023-01 之后的报告期
        df = df[df["end_date"] >= "20230101"]
        # symbol: A 股代码, mkv: 持仓市值, amount: 持股数
        rows = []
        for r in df.itertuples():
            rows.append({
                "end_date": r.end_date, "ann_date": r.ann_date,
                "symbol": r.symbol, "mkv": float(r.mkv) if r.mkv else None,
                "amount": float(r.amount) if r.amount else None,
                "stk_mkv_ratio": float(r.stk_mkv_ratio) if r.stk_mkv_ratio else None,
                "stk_float_ratio": float(r.stk_float_ratio) if r.stk_float_ratio else None,
            })
        cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "限" in msg:
            time.sleep(10)
            continue
        print(f"  [err] {code}: {e}")
        cache.write_text("[]", encoding="utf-8")
    if (i + 1) % 20 == 0:
        print(f"  持仓进度 {i+1}/{len(winners)}")

print(f"\nDONE. winners 持仓写入 {HOLD_DIR}")
