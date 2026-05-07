#!/usr/bin/env python3
"""V7 P1+P3 因子: pyr_velocity 多变种 + f1/f2 阈值条件化.

V7 P1: pyr_velocity 多变种
  pyr_velocity_5_60   = pyr_5d - pyr_60d
  pyr_velocity_10_30  = pyr_10d - pyr_30d
  pyr_velocity_20_60  = pyr_20d - pyr_60d
  (pyr_velocity_10_60 已有 from V6.5)

V7 P3: f1/f2 阈值条件化
  f1/f2 默认阈值 ±3% (大跌/大涨)
  新建: f1_neg1, f1_neg5, f1_neg8 (阈值变种)
  新建: f2_pos1, f2_pos5, f2_pos8

输入: output/moneyflow/cache/{ts_code}.parquet + TDX OHLC
输出: output/v7_extras/features.parquet
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "output" / "moneyflow" / "cache"
OUT_DIR = ROOT / "output" / "v7_extras"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")
START = "20230101"
END   = "20260420"
THOUSAND_TO_YI = 1.0 / 1e5


def read_tdx(market, code):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    data = p.read_bytes()
    n = len(data) // 32
    rows = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append((d.strftime("%Y%m%d"), f[4]/100.0))
    return rows


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def compute_v7(mf: pd.DataFrame, ohlc) -> pd.DataFrame:
    if mf.empty or not ohlc: return pd.DataFrame()
    mf = mf.sort_values("trade_date").reset_index(drop=True)
    px = pd.DataFrame(ohlc, columns=["trade_date","close"])
    df = mf.merge(px, on="trade_date", how="left")
    if df.empty: return pd.DataFrame()

    main = ((df["buy_lg_amount"] + df["buy_elg_amount"]
             - df["sell_lg_amount"] - df["sell_elg_amount"]) * THOUSAND_TO_YI)
    inst_total = ((df["buy_lg_amount"] + df["buy_elg_amount"]
                    + df["sell_lg_amount"] + df["sell_elg_amount"]) * THOUSAND_TO_YI)
    retail_total = ((df["buy_sm_amount"] + df["buy_md_amount"]
                      + df["sell_sm_amount"] + df["sell_md_amount"]) * THOUSAND_TO_YI)
    ret_pct = df["close"].pct_change() * 100

    out = pd.DataFrame({"ts_code": mf["ts_code"], "trade_date": df["trade_date"]})

    # ── pyr_velocity 变种 (新加 3 个, 配合 V6.5 的 _10_60) ──
    for w in [5, 10, 20, 30, 60]:
        inst_w = inst_total.rolling(w, min_periods=max(3, w//3)).sum()
        retail_w = retail_total.rolling(w, min_periods=max(3, w//3)).sum()
        out[f"_pyr_{w}d"] = inst_w / (retail_w + 1e-3)

    out["pyr_velocity_5_60"]  = out["_pyr_5d"]  - out["_pyr_60d"]
    out["pyr_velocity_10_30"] = out["_pyr_10d"] - out["_pyr_30d"]
    out["pyr_velocity_20_60"] = out["_pyr_20d"] - out["_pyr_60d"]
    out["pyr_velocity_5_30"]  = out["_pyr_5d"]  - out["_pyr_30d"]

    # 删除中间列
    for c in [c for c in out.columns if c.startswith("_pyr_")]:
        del out[c]

    # ── f1/f2 阈值变种 ──
    for thr in [1, 5, 8]:
        is_red = (ret_pct < -thr).astype(float)
        is_green = (ret_pct > thr).astype(float)
        out[f"f1_neg{thr}"] = (main * is_red).rolling(5, min_periods=3).sum()
        out[f"f2_pos{thr}"] = (-main * is_green).rolling(5, min_periods=3).sum()

    out = out[(out["trade_date"] >= START) & (out["trade_date"] <= END)].copy()
    return out


def main():
    t0 = time.time()
    cache_files = sorted(CACHE_DIR.glob("*.parquet"))
    print(f"moneyflow cache: {len(cache_files)} 文件", flush=True)

    all_rows = []
    n_ok = n_skip = 0
    for i, p in enumerate(cache_files):
        if (i+1) % 1000 == 0:
            print(f"  [{i+1}/{len(cache_files)}] {int(time.time()-t0)}s ok={n_ok}, rows={len(all_rows):,}",
                  flush=True)
        ts = p.stem
        c, m = code_market(ts)
        ohlc = read_tdx(m, c)
        if not ohlc: n_skip += 1; continue
        try:
            mf = pd.read_parquet(p)
            mf["trade_date"] = mf["trade_date"].astype(str)
        except: n_skip += 1; continue
        if mf.empty: n_skip += 1; continue
        feat = compute_v7(mf, ohlc)
        if feat.empty: n_skip += 1; continue
        all_rows.append(feat)
        n_ok += 1

    full = pd.concat(all_rows, ignore_index=True)
    full.to_parquet(OUT_DIR / "features.parquet", index=False)
    print(f"\nfeatures: {len(full):,} 行 × {full['ts_code'].nunique()} 股", flush=True)
    feat_cols = [c for c in full.columns if c not in ("ts_code","trade_date")]
    print(f"\n=== 因子分布 (10 个新因子) ===", flush=True)
    print(full[feat_cols].describe(percentiles=[0.05,0.5,0.95]).round(4).to_string(), flush=True)
    print(f"\n总耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
