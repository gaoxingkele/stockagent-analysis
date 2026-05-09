#!/usr/bin/env python3
"""通过 Tushare Pro daily API 增量更新 TDX .day 文件到 2026-05-07.

策略:
  1. 5072 股 × Tushare daily 拉 2026-04-21 → 2026-05-07
  2. 转 TDX 二进制格式 (32 bytes/record)
  3. append 到对应 .day 文件末尾

TDX .day 格式 (32 字节/记录, little-endian uint32):
  [0] date (YYYYMMDD int)
  [1] open × 100
  [2] high × 100
  [3] low × 100
  [4] close × 100
  [5] amount (元, raw)         ← Tushare amount * 1000 (千元 → 元)
  [6] volume (股, raw)          ← Tushare vol * 100 (手 → 股)
  [7] reserved (0)

Tushare daily 限速 200/min (免费版).
工时估计: 5072 股 / 200 ≈ 25 分钟
"""
from __future__ import annotations
import os, struct, time, logging
from pathlib import Path
import pandas as pd
import tushare as ts

ROOT = Path(__file__).resolve().parent
TDX = Path(os.getenv("TDX_DIR", "D:/tdx"))

START_DATE = "20260421"  # TDX 现有末日 +1
END_DATE   = "20260507"  # 今天

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("tdx_update")


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


def code_market(ts_code):
    if "." in ts_code:
        c, ex = ts_code.split(".")
        return c, ex.lower()
    return ts_code, "sz"


def get_existing_last_date(market, code):
    """读 TDX 文件末日, 用于校验 (防重复 append)."""
    p = TDX / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    try:
        data = p.read_bytes()
    except: return None
    n = len(data) // 32
    if n == 0: return None
    last = struct.unpack_from("<8I", data, (n-1)*32)
    return last[0]


def append_records(market, code, records):
    """把记录 append 到 .day 文件 (二进制).
    records: list of (date_int, open, high, low, close, amount_yuan, volume_shares)
    """
    if not records: return 0
    p = TDX / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return 0

    # 校验: 去除已存在的日期 (防重复)
    existing_last = get_existing_last_date(market, code)
    if existing_last is None: return 0
    new_records = [r for r in records if r[0] > existing_last]
    if not new_records: return 0

    # pack 为 32 字节/记录
    binary = b""
    for r in new_records:
        date_int, o, h, l, c, amt, vol = r
        binary += struct.pack("<8I",
                                date_int,
                                int(round(o * 100)),
                                int(round(h * 100)),
                                int(round(l * 100)),
                                int(round(c * 100)),
                                int(amt),  # 元
                                int(vol),  # 股
                                0)
    # append
    with open(p, "ab") as f:
        f.write(binary)
    return len(new_records)


def main():
    pro = setup_tushare()
    log.info(f"Tushare 增量拉取 {START_DATE} → {END_DATE}")

    # 获取 5072 股列表 (从 moneyflow cache 推导)
    cache_dir = ROOT / "output" / "moneyflow" / "cache"
    ts_codes = sorted([p.stem for p in cache_dir.glob("*.parquet")])
    log.info(f"5072 股待更新: {len(ts_codes)}")

    t0 = time.time()
    n_ok = n_skip = n_no_data = n_fail = 0
    total_appended = 0

    for i, ts_code in enumerate(ts_codes):
        try:
            df = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
            if df is None or df.empty:
                n_no_data += 1
                continue
            # Tushare 返回是 newest-first, 反转为 oldest-first
            df = df.sort_values("trade_date").reset_index(drop=True)
            records = []
            for _, row in df.iterrows():
                date_int = int(row["trade_date"])
                # Tushare amount 单位千元, vol 单位手
                amt_yuan = row["amount"] * 1000  # 千元 → 元
                vol_shares = row["vol"] * 100    # 手 → 股
                records.append((
                    date_int, row["open"], row["high"], row["low"], row["close"],
                    amt_yuan, vol_shares
                ))

            code, market = code_market(ts_code)
            n_appended = append_records(market, code, records)
            if n_appended > 0:
                total_appended += n_appended
                n_ok += 1
            else:
                n_skip += 1

        except Exception as e:
            n_fail += 1
            err_msg = str(e)
            if "每分钟" in err_msg or "RATE" in err_msg.upper() or "频率" in err_msg:
                log.warning(f"  限速触发, sleep 10s")
                time.sleep(10)

        if (i+1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed * 60
            eta = (len(ts_codes) - (i+1)) / max(rate/60, 0.1)
            log.info(f"  [{i+1}/{len(ts_codes)}] {elapsed:.0f}s, {rate:.0f}/min, ETA {eta:.0f}s, "
                     f"ok={n_ok}, skip={n_skip}, no_data={n_no_data}, fail={n_fail}, "
                     f"+{total_appended} 行")

    log.info(f"完成: ok={n_ok}, skip={n_skip}, no_data={n_no_data}, fail={n_fail}, "
              f"总 append {total_appended} 行, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
