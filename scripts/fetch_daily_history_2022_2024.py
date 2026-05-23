"""拉 daily 历史数据 2022-01 至 2024-01 (补 504d 窗口空缺).

当前 cache 仅 20240102 起, 504d 窗口完全没历史数据.
拉 2022-01 至 2024-01 的 daily, 让 504d 因子在训练样本里有足够覆盖.

时长估计: ~500 交易日 × Tushare daily 速率 ~3 sec/day = 25 min
"""
from __future__ import annotations
import os, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
import tushare as ts

TOKEN = os.getenv("TUSHARE_TOKEN")
pro = ts.pro_api(TOKEN)

CACHE = ROOT / "output" / "tushare_cache" / "daily"
CACHE.mkdir(parents=True, exist_ok=True)

START = "20220104"
END = "20240101"


def main():
    t0 = time.time()
    print(f"\n=== daily 历史拉取 {START}-{END} ===\n", flush=True)
    # 获取交易日历
    trade_cal = pro.trade_cal(exchange="SSE", start_date=START, end_date=END,
                                is_open="1")
    dates = trade_cal["cal_date"].tolist()
    print(f"交易日数: {len(dates)}", flush=True)

    done = sum(1 for d in dates if (CACHE / f"{d}.parquet").exists())
    print(f"已缓存: {done}/{len(dates)}", flush=True)

    for i, d in enumerate(dates):
        out_p = CACHE / f"{d}.parquet"
        if out_p.exists():
            continue
        try:
            df = pro.daily(trade_date=d)
            if df is None or df.empty:
                print(f"  {d}: 空, 跳过", flush=True); continue
            df.to_parquet(out_p, index=False)
            if (i + 1) % 20 == 0:
                print(f"  进度 {i+1}/{len(dates)} ({d}: {len(df)} 股, "
                       f"已用 {time.time()-t0:.0f}s)", flush=True)
            time.sleep(0.4)  # rate limit
        except Exception as e:
            print(f"  {d} 失败: {e}", flush=True)
            time.sleep(2)

    final = sum(1 for d in dates if (CACHE / f"{d}.parquet").exists())
    print(f"\n完成: {final}/{len(dates)} 日, 耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
