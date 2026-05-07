#!/usr/bin/env python3
"""增量更新 Tushare moneyflow 到最新 (2026-01-27 → 2026-04-20).

策略:
  - 5072 股每只检查 cache, 如果 cache 末日 < 2026-04-20, 增量拉取
  - Tushare moneyflow 接口限速 200/min (免费版)
  - 每股增量 ~60 天数据, 单股调用 1 次
"""
from __future__ import annotations
import os, time, logging
from pathlib import Path
import pandas as pd
import sys
import tushare as ts

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "output" / "moneyflow" / "cache"

UPDATE_TO = "20260420"  # TDX 数据末日

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("mf_update")


def setup_tushare():
    if not os.environ.get("TUSHARE_TOKEN"):
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("TUSHARE_TOKEN="):
                    os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip().strip('"\'')
                    break
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    return ts.pro_api()


def main():
    pro = setup_tushare()
    files = sorted(CACHE_DIR.glob("*.parquet"))
    log.info(f"检查 {len(files)} 个 cache 文件...")

    need_update = []
    for p in files:
        try:
            df = pd.read_parquet(p, columns=["trade_date"])
            df["trade_date"] = df["trade_date"].astype(str)
            last_date = df["trade_date"].max()
            if last_date < UPDATE_TO:
                need_update.append((p, last_date))
        except Exception as e:
            log.warning(f"  读取 {p.name} 失败: {e}")

    log.info(f"需要更新: {len(need_update)} 股 (cache 末日 < {UPDATE_TO})")

    if not need_update:
        log.info("全部已最新, 无需更新")
        return

    t0 = time.time()
    n_ok = n_fail = n_no_new = 0
    for i, (p, last_date) in enumerate(need_update):
        ts_code = p.stem  # 如 600519.SH
        # 起始日 = cache 末日 + 1 天 (保守, 用 cache 末日)
        start_date = last_date
        try:
            new = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=UPDATE_TO)
            if new is None or new.empty:
                n_no_new += 1
                continue
            new = new[new["trade_date"] > last_date]  # 严格 >, 不重复
            if new.empty:
                n_no_new += 1
                continue
            old = pd.read_parquet(p)
            old["trade_date"] = old["trade_date"].astype(str)
            new["trade_date"] = new["trade_date"].astype(str)
            merged = pd.concat([old, new], ignore_index=True)
            merged = merged.sort_values("trade_date").drop_duplicates(subset=["trade_date"], keep="last")
            merged.to_parquet(p, index=False)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if "每分钟" in str(e) or "RATE" in str(e).upper():
                log.warning(f"  限速触发, sleep 10s")
                time.sleep(10)
            elif n_fail < 5:
                log.warning(f"  {ts_code} 失败: {str(e)[:100]}")

        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed * 60
            eta = (len(need_update) - (i+1)) / max(rate/60, 0.1)
            log.info(f"  [{i+1}/{len(need_update)}] {elapsed:.0f}s, "
                      f"{rate:.0f}/min, ETA {eta:.0f}s, "
                      f"ok={n_ok}, fail={n_fail}, no_new={n_no_new}")

    log.info(f"完成: ok={n_ok}, fail={n_fail}, no_new={n_no_new}, 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
