"""全市场 1H 数据批量拉取 (2 年, 2024-01-01 → 2026-05-15).

输出: output/tushare_cache/1h/{ts_code}.parquet (每股一个 parquet)

checkpoint: 已存 parquet 跳过, 中断可续

预估: 5000 股 × 0.3s/股 + 限速 200/min = 25-30 分钟
"""
from __future__ import annotations
import os, time, logging
from pathlib import Path
import pandas as pd
import tushare as ts

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "tushare_cache" / "1h"
OUT.mkdir(parents=True, exist_ok=True)

START = "20240101 09:00:00"
END = "20260515 15:00:00"

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("1h_bulk")


def get_all_codes():
    """获取全市场 A 股代码 (含北交所)."""
    basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    df = pd.read_parquet(basic_p)
    return df["ts_code"].tolist()


def main():
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    codes = get_all_codes()
    log.info(f"目标股票: {len(codes)} 只")

    # 跳过已存
    done = set(p.stem for p in OUT.glob("*.parquet"))
    todo = [c for c in codes if c not in done]
    log.info(f"已缓存: {len(done)}, 待拉: {len(todo)}")
    if not todo:
        log.info("全部已缓存, 退出"); return

    t0 = time.time()
    n_ok = n_empty = n_fail = 0
    for i, code in enumerate(todo, 1):
        try:
            df = ts.pro_bar(ts_code=code, freq="60min",
                             start_date=START, end_date=END)
            if df is None or df.empty:
                n_empty += 1
            else:
                # 保留必要字段
                df = df[["trade_time", "open", "high", "low", "close", "vol", "amount", "trade_date"]]
                df.to_parquet(OUT / f"{code}.parquet", index=False)
                n_ok += 1
        except Exception as e:
            n_fail += 1
            msg = str(e)
            if "每分钟" in msg or "RATE" in msg.upper() or "频率" in msg:
                log.warning(f"  限速 {code}, sleep 30s")
                time.sleep(30)
            else:
                log.warning(f"  失败 {code}: {msg[:100]}")
        if i % 100 == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / el * 60
            eta = (len(todo) - i) / max(rate / 60, 0.1)
            log.info(f"  [{i}/{len(todo)}] {el:.0f}s, {rate:.0f}/min, ETA {eta:.0f}s, "
                     f"ok={n_ok}, empty={n_empty}, fail={n_fail}")
        time.sleep(0.05)   # 缓速 (避免触发严格限速)

    log.info(f"完成: ok={n_ok}, empty={n_empty}, fail={n_fail}, 耗时 {time.time()-t0:.0f}s")
    log.info(f"输出: {OUT}/")


if __name__ == "__main__":
    main()
