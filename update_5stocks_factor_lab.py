#!/usr/bin/env python3
"""阶段 3: 仅对 5 只指定股增量更新 factor_lab 153 因子.

策略:
  - 读 5 只股 TDX OHLC + Tushare daily basic
  - 跑 factor_lab.compute_factors() 算 153 因子
  - merge 到现有 factor_lab_3y parquet (按 ts_code+trade_date)
  - 仅这 5 股的 2026-01-27 → 2026-04-20 是新行
  - 全市场其他股保持 2026-01-26 末日 (5 股本身 V7c 推理够用)
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from factor_lab import compute_factors

TDX = os.getenv("TDX_DIR", "D:/tdx")
START = "20240101"  # factor_lab 用近 2 年, 节省时间
END   = "20260420"

STOCKS = [
    ("300567", "sz", "精测电子"),
    ("688037", "sh", "芯源微"),
    ("688577", "sh", "浙海德曼"),
    ("688409", "sh", "富创精密"),
    ("688720", "sh", "艾森股份"),
]


def read_tdx_full(market, code, end_inclusive):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    data = p.read_bytes()
    n = len(data) // 32
    rows = []
    end_int = int(end_inclusive)
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        if di > end_int: continue
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append({
            "trade_date": d.strftime("%Y%m%d"),
            "open": f[1]/100.0, "high": f[2]/100.0,
            "low": f[3]/100.0, "close": f[4]/100.0,
            "vol": float(f[6]),
            "amount": float(f[7]) if len(f) > 7 else float(f[6]) * f[4]/100.0,
        })
    df = pd.DataFrame(rows)
    df["pre_close"] = df["close"].shift(1).bfill()
    df["change"] = df["close"] - df["pre_close"]
    df["pct_chg"] = df["change"] / df["pre_close"] * 100
    return df[df["trade_date"] >= START].reset_index(drop=True)


def main():
    t0 = time.time()
    print(f"=== 阶段 3: 5 股 factor_lab 153 因子增量更新 ===\n", flush=True)

    # 读现有 factor_lab parquet (作为 industry/pe/total_mv 模板)
    PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
    NEW_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups_5stocks"
    NEW_DIR.mkdir(exist_ok=True)

    # 提取 5 股的最新一行 (用作模板获取 industry/pe 等基础列)
    all_parts = []
    target_codes = [f"{c}.{m.upper()}" for c, m, _ in STOCKS]
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df = df[df["ts_code"].isin(target_codes)]
        if not df.empty: all_parts.append(df)
    if not all_parts:
        print("⚠ 5 只股不在 factor_lab parquet"); return
    existing = pd.concat(all_parts, ignore_index=True)
    existing["trade_date"] = existing["trade_date"].astype(str)
    print(f"现有 5 股数据: {len(existing):,} 行, 末日 {existing['trade_date'].max()}", flush=True)

    # 模板: 5 股各自最后一行作为 industry/pe/total_mv 的来源
    template = existing.sort_values(["ts_code","trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
    print(f"模板列: {list(template.columns)[:20]}...共 {len(template.columns)} 列", flush=True)

    # 对每只股算 factor_lab 153 因子, 拼接基础列, 输出
    new_rows_all = []
    for code, market, name in STOCKS:
        ts_code = f"{code}.{market.upper()}"
        print(f"\n  {ts_code} {name} ", flush=True)
        daily = read_tdx_full(market, code, END)
        if daily is None or len(daily) < 30:
            print(f"    OHLC 数据不足"); continue

        # factor_lab 153 因子
        factors = compute_factors(daily)
        factors["ts_code"] = ts_code
        # 仅取 2026-01-27 → 2026-04-20 新增段
        factors_new = factors[factors["trade_date"] > "20260126"].copy()
        if factors_new.empty:
            print(f"    无新数据"); continue

        # merge 模板的基础列 (industry/pe/total_mv 等不变的列)
        tpl = template[template["ts_code"] == ts_code].iloc[0]
        # 找出 factors 没有但 template 有的列 (基础列)
        base_cols = [c for c in template.columns if c not in factors_new.columns]
        for col in base_cols:
            factors_new[col] = tpl[col]

        # 还需要 r5/r10/r20/r30/r40 + dd5/dd10/.../mv_bucket/pe_bucket 等
        # 这些是 forward labels, 新数据没有 (要等 20+ 天后), 用 NaN
        for fwd in ["r5","r10","r20","r30","r40","dd5","dd10","dd20","dd30","dd40"]:
            if fwd not in factors_new.columns:
                factors_new[fwd] = pd.NA

        # 列对齐 (跟 existing 完全一样)
        for col in existing.columns:
            if col not in factors_new.columns:
                factors_new[col] = pd.NA
        factors_new = factors_new[existing.columns]
        new_rows_all.append(factors_new)
        print(f"    {len(factors_new)} 行新数据 (2026-01-27 → 2026-04-20)", flush=True)

    if not new_rows_all:
        print("\n⚠ 无新数据"); return
    new_df = pd.concat(new_rows_all, ignore_index=True)
    out_path = NEW_DIR / "5stocks_2026q1q2.parquet"
    new_df.to_parquet(out_path, index=False)
    print(f"\n输出 {out_path}: {len(new_df)} 行", flush=True)
    print(f"日期范围: {new_df['trade_date'].min()} → {new_df['trade_date'].max()}", flush=True)
    print(f"\n总耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
