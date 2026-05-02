#!/usr/bin/env python3
"""为全部 parquet 样本计算 max_gain_20 + max_dd_20 标签 (从 TDX OHLC).

输出: output/labels/max_gain_labels.parquet
列: ts_code, trade_date, entry_open, max_gain_20, max_dd_20, r20_close, is_clean
覆盖范围: 2024-04 → 2026-01 (训练 + OOS 测试期)
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OUT_DIR = ROOT / "output" / "labels"
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "max_gain_labels.parquet"

TDX = os.getenv("TDX_DIR", "D:/tdx")
START = "20240101"
END   = "20260126"
LOOKAHEAD = 20

# 干净走势阈值 (复用)
CLEAN_GAIN = 20.0
CLEAN_DD   = -3.0


def read_tdx(market: str, code: str):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    try: data = p.read_bytes()
    except: return None
    n = len(data) // 32
    if n == 0: return None
    out = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        out.append((d.strftime("%Y%m%d"), f[1]/100.0, f[2]/100.0, f[3]/100.0, f[4]/100.0))
    return out


def code_market(ts: str):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def main():
    t0 = time.time()
    print("加载 parquet keys (ts_code, trade_date)...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code", "trade_date"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        parts.append(df)
    keys = pd.concat(parts, ignore_index=True).drop_duplicates()
    print(f"总 keys: {len(keys)}, 唯一股票: {keys['ts_code'].nunique()}")

    # 按 ts_code 缓存 TDX 数据
    print("扫 TDX 算 max_gain_20 + max_dd_20 ...")
    rows = []
    ts_codes = keys["ts_code"].unique()
    n_total = 0
    for i, ts in enumerate(ts_codes):
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(ts_codes)}] 已记 {n_total} 条")
        c, m = code_market(ts)
        ohlc = read_tdx(m, c)
        if not ohlc:
            continue
        # 索引: trade_date → idx
        idx_map = {x[0]: j for j, x in enumerate(ohlc)}
        # 对该股所有日期算
        sub = keys[keys["ts_code"] == ts]
        for td in sub["trade_date"].values:
            idx = idx_map.get(td)
            if idx is None or idx + LOOKAHEAD >= len(ohlc): continue
            entry_open = ohlc[idx + 1][1]
            if entry_open <= 0: continue
            future = ohlc[idx + 1: idx + 1 + LOOKAHEAD]
            max_h = max(x[2] for x in future)
            min_l = min(x[3] for x in future)
            close_end = ohlc[idx + LOOKAHEAD][4]
            mg = (max_h / entry_open - 1) * 100
            md = (min_l / entry_open - 1) * 100
            rows.append({
                "ts_code": ts, "trade_date": td,
                "entry_open": entry_open,
                "max_gain_20": round(mg, 4),
                "max_dd_20":   round(md, 4),
                "r20_close":   round((close_end / entry_open - 1) * 100, 4),
                "is_clean": (mg >= CLEAN_GAIN) and (md >= CLEAN_DD),
            })
            n_total += 1

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"\n写出 {OUT} ({len(df)} 行, {time.time()-t0:.1f}s)")

    # 分布概览
    print("\n=== max_gain_20 分布 ===")
    print(df["max_gain_20"].describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(2))
    print("\n=== 干净样本数: %d (%.1f%%) ===" % (df["is_clean"].sum(),
                                                  df["is_clean"].mean()*100))
    print("max_gain ≥ 15%: %d (%.1f%%)" % ((df["max_gain_20"]>=15).sum(),
                                           (df["max_gain_20"]>=15).mean()*100))
    print("max_gain ≥ 20%: %d (%.1f%%)" % ((df["max_gain_20"]>=20).sum(),
                                           (df["max_gain_20"]>=20).mean()*100))
    print("max_gain ≥ 30%: %d (%.1f%%)" % ((df["max_gain_20"]>=30).sum(),
                                           (df["max_gain_20"]>=30).mean()*100))


if __name__ == "__main__":
    main()
