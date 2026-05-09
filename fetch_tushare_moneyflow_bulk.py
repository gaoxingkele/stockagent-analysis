#!/usr/bin/env python3
"""按 trade_date 拉全市场 moneyflow 增量, 缓存到 output/tushare_cache/moneyflow/.

策略:
  - pro.moneyflow(trade_date=YYYYMMDD) 一次返回当日全市场资金流
  - 仅拉 04-21 → 05-07 增量段 (12 个交易日)
  - 已有则跳过
"""
from __future__ import annotations
import os, time, logging
from pathlib import Path
import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "tushare_cache" / "moneyflow"
OUT.mkdir(parents=True, exist_ok=True)

START = "20260421"
END   = "20260508"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("mf_bulk")


def setup():
    if not os.environ.get("TUSHARE_TOKEN"):
        env = ROOT / ".env"
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("TUSHARE_TOKEN="):
                os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
                break
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    return ts.pro_api()


def main():
    pro = setup()
    cal = pro.trade_cal(exchange="SSE", start_date=START, end_date=END, is_open="1")
    dates = sorted(cal["cal_date"].astype(str).tolist())
    log.info(f"交易日 {START} → {END}: {len(dates)} 个")

    todo = [d for d in dates if not (OUT / f"{d}.parquet").exists()]
    log.info(f"已缓存 {len(dates)-len(todo)}, 待拉 {len(todo)}")

    t0 = time.time()
    n_ok = n_empty = n_fail = 0
    for d in todo:
        try:
            df = pro.moneyflow(trade_date=d)
            if df is None or df.empty:
                n_empty += 1
                log.warning(f"  {d} 空")
            else:
                df.to_parquet(OUT / f"{d}.parquet", index=False)
                n_ok += 1
                log.info(f"  {d} {len(df)} 行")
        except Exception as e:
            n_fail += 1
            msg = str(e)
            if "每分钟" in msg or "RATE" in msg.upper() or "频率" in msg:
                log.warning(f"  限速触发 {d}, sleep 10s")
                time.sleep(10)
            else:
                log.warning(f"  失败 {d}: {msg[:80]}")

    log.info(f"完成: ok={n_ok}, empty={n_empty}, fail={n_fail}, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
