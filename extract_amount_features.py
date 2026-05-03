#!/usr/bin/env python3
"""提取绝对成交额特征 (TDX OHLC × volume).

新增 6 个特征 (现有 153 因子缺失):
  amount_1d:         当日成交额 (亿元)
  amount_5d_avg:     5 日平均成交额
  amount_20d_avg:    20 日平均成交额
  amount_log:        log(amount_1d+1) — 长尾分布平滑
  amount_zscore_60d: 60 日 z-score (极端放量识别)
  amount_breakout:   当日成交额是否突破前 20 日 max (1/0)

输出: output/amount_features/amount_features.parquet
覆盖: 2023-01 → 2026-01
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OUT_DIR = ROOT / "output" / "amount_features"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

START = "20230101"
END   = "20260126"


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
        out.append((d.strftime("%Y%m%d"),
                     f[1]/100.0, f[2]/100.0, f[3]/100.0, f[4]/100.0,
                     float(f[6])))
    return out


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def compute_amount_features(daily: list, start: str, end: str) -> pd.DataFrame:
    if len(daily) < 30: return pd.DataFrame()
    closes = np.array([r[4] for r in daily])
    vols   = np.array([r[5] for r in daily])
    dates  = [r[0] for r in daily]
    # amount = volume × close (用单位股, 已是手数所以再 × 100 得到金额单位)
    amount = vols * closes / 1e8   # 转亿元 (假设 vol 是 100股×N)
    # 实际 TDX 的 vol 是手数 (1手=100股), amount = vol × 100 × close / 1e8 = vol × close / 1e6
    # 让我们用统一单位: 把它换成亿元
    amount = vols * closes / 1e6  # 亿元 (vol 是手数, ×100股 ×close 元 / 1e8)

    n = len(amount)
    a5  = pd.Series(amount).rolling(5,  min_periods=3).mean().values
    a20 = pd.Series(amount).rolling(20, min_periods=10).mean().values
    a60_mean = pd.Series(amount).rolling(60, min_periods=30).mean().values
    a60_std  = pd.Series(amount).rolling(60, min_periods=30).std().values
    a60_z    = (amount - a60_mean) / np.where(a60_std > 1e-9, a60_std, 1)
    breakout = pd.Series(amount).rolling(20).max().values
    breakout_flag = (amount >= breakout).astype(int)

    rows = []
    for i in range(n):
        td = dates[i]
        if td < start or td > end: continue
        if np.isnan(a20[i]): continue
        rows.append({
            "trade_date":         td,
            "amount_1d":          round(float(amount[i]), 4),
            "amount_5d_avg":      round(float(a5[i]) if not np.isnan(a5[i]) else 0, 4),
            "amount_20d_avg":     round(float(a20[i]), 4),
            "amount_log":         round(float(np.log1p(amount[i])), 4),
            "amount_zscore_60d":  round(float(a60_z[i]) if not np.isnan(a60_z[i]) else 0, 4),
            "amount_breakout":    int(breakout_flag[i]),
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print("加载 parquet 股票列表...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code","trade_date"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        parts.append(df)
    all_keys = pd.concat(parts).drop_duplicates()
    ts_codes = all_keys["ts_code"].unique()
    print(f"股票: {len(ts_codes)}")

    all_amount = []
    for i, ts in enumerate(ts_codes):
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(ts_codes)}] {int(time.time()-t0)}s")
        c, m = code_market(ts)
        daily = read_tdx(m, c)
        if not daily: continue
        feat = compute_amount_features(daily, START, END)
        if feat.empty: continue
        feat["ts_code"] = ts
        all_amount.append(feat)

    df = pd.concat(all_amount, ignore_index=True)
    print(f"\n总样本: {len(df):,} 行")

    # 分布
    print("\n=== 各特征分布 ===")
    print(df[["amount_1d","amount_5d_avg","amount_zscore_60d","amount_breakout"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_string())
    print(f"\nbreakout 比例: {df['amount_breakout'].mean()*100:.1f}%")

    df.to_parquet(OUT_DIR / "amount_features.parquet", index=False)
    print(f"\n写出 {OUT_DIR / 'amount_features.parquet'} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
