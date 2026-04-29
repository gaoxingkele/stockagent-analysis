"""拉 24 只 winner ETF 的持仓时序 (fund_portfolio)。"""
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

ETF_DIR = Path(__file__).parent / "output" / "etf_analysis"
HOLD_DIR = ETF_DIR / "holdings"
HOLD_DIR.mkdir(exist_ok=True)

with open(ETF_DIR / "winners_relaxed.json", encoding="utf-8") as f:
    winners = json.load(f)
print(f"拉 {len(winners)} 只 winner 的持仓...")

for i, etf in enumerate(winners):
    code = etf["ts_code"]
    cache = HOLD_DIR / f"{code}.json"
    if cache.exists():
        try:
            existing = json.loads(cache.read_text(encoding="utf-8"))
            if existing:
                print(f"  [{i+1}/{len(winners)}] {code} 已缓存 ({len(existing)} 行)")
                continue
        except: pass
    try:
        df = pro.fund_portfolio(ts_code=code)
        time.sleep(0.5)
        if df is None or df.empty:
            cache.write_text("[]", encoding="utf-8")
            print(f"  [{i+1}/{len(winners)}] {code} 空")
            continue
        df = df[df["end_date"] >= "20230101"]
        rows = []
        for r in df.itertuples():
            rows.append({
                "end_date": r.end_date, "ann_date": r.ann_date,
                "symbol": r.symbol,
                "mkv": float(r.mkv) if r.mkv else None,
                "amount": float(r.amount) if r.amount else None,
                "stk_mkv_ratio": float(r.stk_mkv_ratio) if r.stk_mkv_ratio else None,
                "stk_float_ratio": float(r.stk_float_ratio) if r.stk_float_ratio else None,
            })
        cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"  [{i+1}/{len(winners)}] {code} {etf['name'][:25]} {len(rows)} 行")
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "频率" in msg:
            print(f"  限频, sleep 10s...")
            time.sleep(10)
            continue
        print(f"  [err] {code}: {e}")
        cache.write_text("[]", encoding="utf-8")

print(f"\nDONE. 持仓数据写入 {HOLD_DIR}")
