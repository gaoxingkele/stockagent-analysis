#!/usr/bin/env python3
"""扩 labels 到 daily cache 能算的最新日期 (用 Tushare cache).

目标输出 (与现有 labels schema 对齐):
  - labels_10d_ext.parquet: r10/r20/max_gain_10/max_dd_10 (d 末日 ≈ 04-23, r20 留 forward 20)
  - max_gain_labels_ext.parquet: max_gain_20/max_dd_20/r20_close/is_clean (d 末日 ≈ 04-09)

策略: 增量段, 从现有 labels 末日 +1 → 最新可算日

幂等: 重跑只会覆盖 ext parquet
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY_CACHE = ROOT / "output" / "tushare_cache" / "daily"
OUT_LABELS_10D = ROOT / "output" / "cogalpha_features" / "labels_10d_ext.parquet"
OUT_LABELS_20  = ROOT / "output" / "labels" / "max_gain_labels_ext.parquet"

LABELS_10D_EXISTING = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20_EXISTING  = ROOT / "output" / "labels" / "max_gain_labels.parquet"

CLEAN_GAIN = 20.0
CLEAN_DD   = -3.0


def load_daily_by_code():
    files = sorted(DAILY_CACHE.glob("*.parquet"))
    parts = [pd.read_parquet(f) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return {ts: g.reset_index(drop=True) for ts, g in big.groupby("ts_code")}


def main():
    t0 = time.time()
    print("=== 扩 labels 增量到最新可算日 ===\n", flush=True)

    print("加载现有 labels 末日 ...", flush=True)
    l10 = pd.read_parquet(LABELS_10D_EXISTING, columns=["trade_date"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    l10_end = l10["trade_date"].max()
    l20 = pd.read_parquet(LABELS_20_EXISTING, columns=["trade_date"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    l20_end = l20["trade_date"].max()
    print(f"  labels_10d 末日: {l10_end}", flush=True)
    print(f"  max_gain_labels 末日: {l20_end}", flush=True)

    print("\n加载 Tushare daily cache ...", flush=True)
    daily_by_code = load_daily_by_code()
    print(f"  {len(daily_by_code)} 股, 耗时 {time.time()-t0:.0f}s", flush=True)

    rows_10 = []
    rows_20 = []
    ts_codes = sorted(daily_by_code.keys())
    print(f"\n算 forward labels (start_d > 现有末日, 留足 forward N) ...", flush=True)
    t = time.time()
    for i, ts in enumerate(ts_codes, 1):
        d = daily_by_code[ts]
        n = len(d)
        if n < 25: continue
        opens = d["open"].values
        highs = d["high"].values
        lows  = d["low"].values
        closes = d["close"].values
        dates = d["trade_date"].values

        # labels_10d: 需要 forward 20 (因为含 r20 列), d+20 < n
        # max_gain_labels: 同样 forward 20
        for idx in range(n - 20):
            td = dates[idx]
            entry_open = opens[idx + 1]
            if entry_open <= 0: continue
            # 10 日窗口
            f10_h = highs[idx+1: idx+11]
            f10_l = lows [idx+1: idx+11]
            close_10 = closes[idx + 10]
            r10 = (close_10 / entry_open - 1) * 100
            mg10 = (f10_h.max() / entry_open - 1) * 100
            md10 = (f10_l.min() / entry_open - 1) * 100

            # 20 日窗口
            f20_h = highs[idx+1: idx+21]
            f20_l = lows [idx+1: idx+21]
            close_20 = closes[idx + 20]
            r20 = (close_20 / entry_open - 1) * 100
            mg20 = (f20_h.max() / entry_open - 1) * 100
            md20 = (f20_l.min() / entry_open - 1) * 100

            if td > l10_end:
                rows_10.append({
                    "trade_date": td, "entry_open": float(entry_open),
                    "r10": round(r10, 4),
                    "max_gain_10": round(mg10, 4),
                    "max_dd_10": round(md10, 4),
                    "r20": round(r20, 4),
                    "ts_code": ts,
                })
            if td > l20_end:
                rows_20.append({
                    "ts_code": ts, "trade_date": td,
                    "entry_open": float(entry_open),
                    "max_gain_20": round(mg20, 4),
                    "max_dd_20":   round(md20, 4),
                    "r20_close":   round(r20, 4),
                    "is_clean": bool((mg20 >= CLEAN_GAIN) and (md20 >= CLEAN_DD)),
                })

        if i % 1000 == 0:
            print(f"  [{i}/{len(ts_codes)}] {time.time()-t:.0f}s, "
                  f"l10={len(rows_10):,} l20={len(rows_20):,}", flush=True)

    df10 = pd.DataFrame(rows_10)
    df20 = pd.DataFrame(rows_20)
    if not df10.empty:
        df10.to_parquet(OUT_LABELS_10D, index=False)
        print(f"\nlabels_10d_ext: {len(df10):,} 行 末日 {df10['trade_date'].max()} → {OUT_LABELS_10D}", flush=True)
    if not df20.empty:
        df20.to_parquet(OUT_LABELS_20, index=False)
        print(f"max_gain_labels_ext: {len(df20):,} 行 末日 {df20['trade_date'].max()} → {OUT_LABELS_20}", flush=True)
    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
