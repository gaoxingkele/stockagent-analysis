#!/usr/bin/env python3
"""factor_lab 153 因子全市场增量 2026-04-21 → 2026-05-07, 数据源 Tushare cache.

策略:
  - 读 output/tushare_cache/daily/*.parquet (564 天全市场), concat 成大表
  - 复用 factor_groups/ 的 ts_code 分组规则 (52 group × 100 股)
  - 5 worker 并行: 每 group 内对每股取历史 → compute_factors → 取增量段
  - 输出 factor_groups_extension/group_XXX_ext2.parquet (04-21 → 05-07)
  - 对应 group_XXX_ext.parquet 已覆盖 01-27 → 04-20, 两者拼起来即 01-27 → 05-07
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from factor_lab import compute_factors

DAILY_CACHE = ROOT / "output" / "tushare_cache" / "daily"
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
EXT_DIR     = ROOT / "output" / "factor_lab_3y" / "factor_groups_extension"
EXT_DIR.mkdir(exist_ok=True)

START      = "20240101"
END        = "20260513"
NEW_START  = "20260421"
NEW_END    = "20260513"


def load_all_daily():
    files = sorted(DAILY_CACHE.glob("*.parquet"))
    parts = []
    for f in files:
        if f.stem < START or f.stem > END:
            continue
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts, ignore_index=True)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def process_group(group_path, ts_codes, templates, base_cols, daily_by_code):
    rows = []
    for ts in ts_codes:
        d = daily_by_code.get(ts)
        if d is None or len(d) < 60:
            continue
        try:
            factors = compute_factors(d)
        except Exception:
            continue
        factors["ts_code"] = ts
        new = factors[(factors["trade_date"] >= NEW_START) &
                      (factors["trade_date"] <= NEW_END)].copy()
        if new.empty:
            continue
        tpl = templates.get(ts, {})
        for col in base_cols:
            if col not in new.columns:
                new[col] = tpl.get(col, pd.NA)
        rows.append(new)
    if not rows:
        return 0
    out = pd.concat(rows, ignore_index=True)
    out_path = EXT_DIR / f"{group_path.stem}_ext2.parquet"
    out.to_parquet(out_path, index=False)
    return len(out)


def main():
    t0 = time.time()
    print("=== factor_lab Tushare 增量 04-21 → 05-07 ===\n", flush=True)

    print("1) 加载 Tushare daily cache ...", flush=True)
    big = load_all_daily()
    print(f"   {len(big):,} 行, {big['ts_code'].nunique()} 股, "
          f"日期 {big['trade_date'].min()} → {big['trade_date'].max()}", flush=True)

    print("\n2) 按 ts_code 切分到 dict (一次性, 后续 worker 读)...", flush=True)
    daily_by_code = {ts: g.reset_index(drop=True) for ts, g in big.groupby("ts_code")}
    print(f"   {len(daily_by_code)} 股", flush=True)

    print("\n3) 串行跑 52 个 group ...", flush=True)
    parquet_files = sorted(PARQUET_DIR.glob("group_*.parquet"))
    total = 0
    for i, p in enumerate(parquet_files, 1):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        last_rows = df.sort_values(["ts_code","trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
        templates = {row["ts_code"]: row.to_dict() for _, row in last_rows.iterrows()}
        ts_codes = sorted(templates.keys())
        base_cols = list(templates[ts_codes[0]].keys())
        n = process_group(p, ts_codes, templates, base_cols, daily_by_code)
        total += n
        print(f"   [{i}/{len(parquet_files)}] {p.stem}: {n} 行 ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n=== 完成 ===", flush=True)
    print(f"  新增: {total:,} 行 (5072 股 × 12 交易日 ≈ 60K)", flush=True)
    print(f"  输出: {EXT_DIR}/group_XXX_ext2.parquet", flush=True)
    print(f"  耗时: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
