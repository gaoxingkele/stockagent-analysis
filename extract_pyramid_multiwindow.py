#!/usr/bin/env python3
"""V6 P2: mfk_pyramid_top_heavy 多窗口变种 (4 窗口 + 2 派生).

V6 P1 实证: mfk_pyramid_top_heavy (5d) 是最强单过滤因子, vs hs300 α 比 baseline 提升 0.29pp.
现扩展 4 窗口 + 2 派生, 看哪个时间尺度最强:
  pyr_5d   : (现有 mfk_pyramid_top_heavy 重命名)
  pyr_10d  : 中短期
  pyr_20d  : 中期
  pyr_60d  : 长期 (3 月)
  pyr_acc  : 5d / 20d 比值 (机构占比加速度)
  pyr_velocity : 10d - 60d 差值 (机构占比趋势)

输出: output/pyramid_v2/features.parquet
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "output" / "moneyflow" / "cache"
OUT_DIR = ROOT / "output" / "pyramid_v2"
OUT_DIR.mkdir(exist_ok=True)
START = "20230101"
END   = "20260126"
THOUSAND_TO_YI = 1.0 / 1e5


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def compute_pyramid_v2(mf: pd.DataFrame) -> pd.DataFrame:
    if mf.empty: return pd.DataFrame()
    mf = mf.sort_values("trade_date").reset_index(drop=True)

    # 4 层资金 (亿元)
    inst_total = (mf["buy_lg_amount"] + mf["buy_elg_amount"]
                   + mf["sell_lg_amount"] + mf["sell_elg_amount"]) * THOUSAND_TO_YI
    retail_total = (mf["buy_sm_amount"] + mf["buy_md_amount"]
                     + mf["sell_sm_amount"] + mf["sell_md_amount"]) * THOUSAND_TO_YI

    out = pd.DataFrame({"ts_code": mf["ts_code"], "trade_date": mf["trade_date"]})

    # ── 4 个窗口 ──
    for w in [5, 10, 20, 60]:
        inst_w = inst_total.rolling(w, min_periods=max(3, w//3)).sum()
        retail_w = retail_total.rolling(w, min_periods=max(3, w//3)).sum()
        out[f"pyr_{w}d"] = inst_w / (retail_w + 1e-3)

    # ── 派生 ──
    # 加速度: 5d / 20d (短期机构占比 / 长期, >1 = 加仓中, <1 = 减仓中)
    out["pyr_acc_5_20"] = out["pyr_5d"] / (out["pyr_20d"] + 1e-3)
    # 趋势: 10d - 60d
    out["pyr_velocity_10_60"] = out["pyr_10d"] - out["pyr_60d"]

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
        try:
            mf = pd.read_parquet(p)
            mf["trade_date"] = mf["trade_date"].astype(str)
        except: n_skip += 1; continue
        if mf.empty: n_skip += 1; continue
        feat = compute_pyramid_v2(mf)
        if feat.empty: n_skip += 1; continue
        all_rows.append(feat)
        n_ok += 1

    if not all_rows:
        print("没数据!"); return
    full = pd.concat(all_rows, ignore_index=True)
    full.to_parquet(OUT_DIR / "features.parquet", index=False)
    print(f"\nfeatures: {len(full):,} 行 × {full['ts_code'].nunique()} 股", flush=True)

    print(f"\n=== 因子分布 ===", flush=True)
    feat_cols = [c for c in full.columns if c.startswith("pyr_")]
    print(full[feat_cols].describe(percentiles=[0.05, 0.5, 0.95]).round(4).to_string(), flush=True)
    print(f"\n总耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
