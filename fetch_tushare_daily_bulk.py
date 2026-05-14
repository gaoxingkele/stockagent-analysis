#!/usr/bin/env python3
"""按 trade_date 维度拉全市场 daily, 一日一 parquet 落地.

策略:
  - pro.daily(trade_date=YYYYMMDD) 一次返回当日全市场 (~5000 行)
  - 2024-01-01 → 2026-05-07 大约 580 个交易日, 限速 200/min ≈ 3 分钟
  - 增量: output/tushare_cache/daily/YYYYMMDD.parquet 已存在则跳过
  - 交易日历来自 pro.trade_cal(exchange='SSE')
"""
from __future__ import annotations
import os, time, logging
from pathlib import Path
import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "tushare_cache" / "daily"
OUT.mkdir(parents=True, exist_ok=True)

START = "20240101"
END   = "20260513"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("daily_bulk")


def setup():
    if not os.environ.get("TUSHARE_TOKEN"):
        env = ROOT / ".env"
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("TUSHARE_TOKEN="):
                os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
                break
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    return ts.pro_api()


def get_trade_dates(pro, start, end):
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
    return sorted(cal["cal_date"].astype(str).tolist())


def main():
    pro = setup()
    dates = get_trade_dates(pro, START, END)
    log.info(f"交易日 {START} → {END}: {len(dates)} 个")

    todo = [d for d in dates if not (OUT / f"{d}.parquet").exists()]
    log.info(f"已缓存 {len(dates)-len(todo)}, 待拉 {len(todo)}")
    if not todo:
        log.info("全部已缓存, 退出")
        return

    t0 = time.time()
    n_ok = n_empty = n_fail = 0
    for i, d in enumerate(todo, 1):
        try:
            df = pro.daily(trade_date=d)
            if df is None or df.empty:
                n_empty += 1
            else:
                df.to_parquet(OUT / f"{d}.parquet", index=False)
                n_ok += 1
        except Exception as e:
            n_fail += 1
            msg = str(e)
            if "每分钟" in msg or "RATE" in msg.upper() or "频率" in msg:
                log.warning(f"  限速触发 {d}, sleep 10s")
                time.sleep(10)
            else:
                log.warning(f"  失败 {d}: {msg[:80]}")

        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / el * 60
            eta = (len(todo) - i) / max(rate / 60, 0.1)
            log.info(f"  [{i}/{len(todo)}] {el:.0f}s, {rate:.0f}/min, ETA {eta:.0f}s, "
                     f"ok={n_ok}, empty={n_empty}, fail={n_fail}")

    log.info(f"完成: ok={n_ok}, empty={n_empty}, fail={n_fail}, 耗时 {time.time()-t0:.0f}s")
    log.info(f"输出: {OUT}/")


if __name__ == "__main__":
    main()
