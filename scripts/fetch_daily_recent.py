"""拉最近一周 daily 数据 (用于验证 5/15 之前持仓的 r5 label)."""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
import tushare as ts

pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
CACHE = ROOT / "output" / "tushare_cache" / "daily"

START, END = "20260518", "20260522"
trade_cal = pro.trade_cal(exchange="SSE", start_date=START, end_date=END, is_open="1")
dates = trade_cal["cal_date"].tolist()
print(f"交易日: {dates}", flush=True)

t0 = time.time()
for d in dates:
    out_p = CACHE / f"{d}.parquet"
    if out_p.exists():
        print(f"  {d}: 已存在, 跳过", flush=True); continue
    try:
        df = pro.daily(trade_date=d)
        if df is None or df.empty:
            print(f"  {d}: 空", flush=True); continue
        df.to_parquet(out_p, index=False)
        print(f"  {d}: {len(df)} stocks OK", flush=True)
        time.sleep(0.5)
    except Exception as e:
        print(f"  {d} 失败: {e}", flush=True)

print(f"\n耗时 {time.time()-t0:.0f}s")
