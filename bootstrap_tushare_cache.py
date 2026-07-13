# -*- coding: utf-8 -*-
"""从零 bootstrap: 下载全市场 daily + moneyflow 到 output/tushare_cache/{daily,moneyflow}/{YYYYMMDD}.parquet

- 交易日历: 20240101 → 最新可用交易日 (SSE, is_open=1)
- 每交易日 1 次 pro.daily + 1 次 pro.moneyflow, 原始列直接落 parquet (与 daily_review 期望一致)
- 断点续传: 已存在的日期文件跳过
- 限速: 每次调用后 sleep, 失败重试 3 次
用法: python bootstrap_tushare_cache.py [START=20240101] [END=auto]
"""
from __future__ import annotations
import os, sys, time, datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
import tushare as ts
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

DAILY = ROOT / "output/tushare_cache/daily"
MF = ROOT / "output/tushare_cache/moneyflow"
DAILY.mkdir(parents=True, exist_ok=True)
MF.mkdir(parents=True, exist_ok=True)

SLEEP = 0.25          # 每次成功调用后休眠
RETRY = 3


def log(s):
    sys.stdout.buffer.write((f"[{datetime.datetime.now():%H:%M:%S}] {s}\n").encode("utf-8"))
    sys.stdout.flush()


def latest_trade_date() -> str:
    for k in range(8):
        d = (datetime.datetime.now() - datetime.timedelta(days=k)).strftime("%Y%m%d")
        try:
            x = pro.daily(trade_date=d)
            if x is not None and len(x) > 0:
                return d
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError("no available trade date")


def call(fn, **kw):
    for i in range(RETRY):
        try:
            r = fn(**kw)
            time.sleep(SLEEP)
            return r
        except Exception as e:
            wait = 2 + i * 3
            log(f"  ! {fn.__name__}{kw} err={str(e)[:60]} retry in {wait}s")
            time.sleep(wait)
    return None


def main():
    args = [a for a in sys.argv[1:] if a.isdigit()]
    start = args[0] if len(args) >= 1 else "20240101"
    end = args[1] if len(args) >= 2 else latest_trade_date()
    log(f"目标区间 {start} ~ {end}")

    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
    days = sorted(cal["cal_date"].astype(str))
    log(f"交易日总数 {len(days)}")

    have_d = set(p.stem for p in DAILY.glob("*.parquet"))
    have_m = set(p.stem for p in MF.glob("*.parquet"))
    todo = [d for d in days if d not in have_d or d not in have_m]
    log(f"已有 daily {len(have_d)} / moneyflow {len(have_m)}; 待下载 {len(todo)}")

    t0 = time.time()
    n_d = n_m = 0
    for i, d in enumerate(todo, 1):
        if d not in have_d:
            df = call(pro.daily, trade_date=d)
            if df is not None and len(df):
                df.to_parquet(DAILY / f"{d}.parquet", index=False)
                n_d += 1
        if d not in have_m:
            mf = call(pro.moneyflow, trade_date=d)
            if mf is not None and len(mf):
                mf.to_parquet(MF / f"{d}.parquet", index=False)
                n_m += 1
        if i % 25 == 0 or i == len(todo):
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            log(f"  {i}/{len(todo)}  daily+{n_d} mf+{n_m}  用时 {el:.0f}s ETA {eta:.0f}s")
    log(f"完成: daily 写 {n_d}, moneyflow 写 {n_m}, 总用时 {time.time()-t0:.0f}s")
    log(f"最终 daily 文件 {len(list(DAILY.glob('*.parquet')))}, moneyflow {len(list(MF.glob('*.parquet')))}")


if __name__ == "__main__":
    main()
