#!/usr/bin/env python3
"""把已打分样本和 TDX OHLC 算出的 max_gain_20/max_dd_20 合并, 重新分析仓位精度.

读: output/diagnose_3d/scored_samples.parquet (含 sparse/lgbm/clean/position)
写: output/diagnose_3d/scored_with_maxgain.parquet (加 max_gain_20/max_dd_20)
报: 各仓位档/分桶的真实 max_gain 表现
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCORED = ROOT / "output" / "diagnose_3d" / "scored_samples.parquet"
OUT = ROOT / "output" / "diagnose_3d" / "scored_with_maxgain.parquet"
TDX = os.getenv("TDX_DIR", "D:/tdx")
LOOKAHEAD = 20

def read_tdx(market, code):
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

def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"

def main():
    t0 = time.time()
    print("加载 scored_samples.parquet ...")
    df = pd.read_parquet(SCORED)
    print(f"行数: {len(df)}, 唯一股票: {df['ts_code'].nunique()}")

    # 缓存每股 daily
    daily_cache = {}
    print("扫 TDX 算 max_gain_20 / max_dd_20 ...")
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        if (i+1) % 20000 == 0:
            print(f"  [{i+1}/{len(df)}]")
        ts = r["ts_code"]
        if ts not in daily_cache:
            c, m = code_market(ts)
            ohlc = read_tdx(m, c)
            daily_cache[ts] = ohlc
        ohlc = daily_cache[ts]
        if not ohlc:
            rows.append({"ts_code": ts, "trade_date": r["trade_date"],
                         "max_gain_20": None, "max_dd_20": None, "entry_open": None})
            continue
        # 找 trade_date 索引
        td = r["trade_date"]
        idx = next((j for j, x in enumerate(ohlc) if x[0] == td), None)
        if idx is None or idx + LOOKAHEAD >= len(ohlc):
            rows.append({"ts_code": ts, "trade_date": td,
                         "max_gain_20": None, "max_dd_20": None, "entry_open": None})
            continue
        entry_open = ohlc[idx + 1][1]
        if entry_open <= 0:
            rows.append({"ts_code": ts, "trade_date": td,
                         "max_gain_20": None, "max_dd_20": None, "entry_open": None})
            continue
        future = ohlc[idx + 1: idx + 1 + LOOKAHEAD]
        max_h = max(x[2] for x in future)
        min_l = min(x[3] for x in future)
        rows.append({
            "ts_code": ts, "trade_date": td,
            "max_gain_20": (max_h / entry_open - 1) * 100,
            "max_dd_20":   (min_l / entry_open - 1) * 100,
            "entry_open": entry_open,
        })

    aug = pd.DataFrame(rows)
    df = df.merge(aug, on=["ts_code", "trade_date"], how="left")
    df.to_parquet(OUT, index=False)
    print(f"写出 {OUT} ({time.time()-t0:.1f}s)")

    # ── 报告 ──
    have = df.dropna(subset=["max_gain_20", "real_r20"])
    print(f"\n有效样本: {len(have)} / {len(df)}")

    print("\n=== 仓位档 vs 真实表现 ===")
    have["pos_bucket"] = pd.cut(have["position_pct"].fillna(0),
                                 bins=[-1, 5, 15, 30, 60, 100],
                                 labels=["0-5%","5-15%","15-30%","30-60%","60-100%"])
    summary = have.groupby("pos_bucket", observed=True).agg(
        n=("real_r20", "count"),
        r20_close=("real_r20", "mean"),
        max_gain_avg=("max_gain_20", "mean"),
        max_gain_med=("max_gain_20", "median"),
        max_dd_avg=("max_dd_20", "mean"),
        max_dd_med=("max_dd_20", "median"),
        clean_hit_rate=("max_gain_20", lambda x: ((x >= 20) & (have.loc[x.index, "max_dd_20"] >= -3)).mean() * 100),
        gain_15_rate=("max_gain_20", lambda x: (x >= 15).mean() * 100),
        gain_20_rate=("max_gain_20", lambda x: (x >= 20).mean() * 100),
    ).round(2)
    print(summary.to_string())

    print("\n=== 高仓位组 (60-100%) 的 MV×ETF 切片 ===")
    high = have[have["position_pct"] >= 60]
    cuts = high.groupby(["mv_seg", "etf_held"], observed=True).agg(
        n=("real_r20", "count"),
        r20_close=("real_r20", "mean"),
        max_gain_avg=("max_gain_20", "mean"),
        gain_15_rate=("max_gain_20", lambda x: (x >= 15).mean() * 100),
        gain_20_rate=("max_gain_20", lambda x: (x >= 20).mean() * 100),
        clean_hit_rate=("max_gain_20", lambda x: ((x >= 20) & (high.loc[x.index, "max_dd_20"] >= -3)).mean() * 100),
    ).round(2)
    print(cuts.to_string())

    summary.to_csv(ROOT / "output" / "diagnose_3d" / "position_with_maxgain.csv", encoding="utf-8-sig")
    cuts.to_csv(ROOT / "output" / "diagnose_3d" / "high_pos_mv_etf.csv", encoding="utf-8-sig")
    print(f"\n总耗时: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
