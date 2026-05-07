#!/usr/bin/env python3
"""全市场 factor_lab 153 因子增量更新到 2026-04-20.

策略:
  - 读现有 factor_lab_3y/factor_groups/*.parquet (5072 股 × 153 因子, 末日 2026-01-26)
  - 5 worker 并行: 每股 read_tdx → compute_factors → 取 2026-01-27 → 2026-04-20 增量
  - 用现有 parquet 最后一行作为基础列模板 (industry/pe/total_mv 等)
  - 输出: output/factor_lab_3y/factor_groups_extension/{group}.parquet (新增段)
  - 加载时 union 现有 + 扩展即可

工时估计: 5 worker × 5072 股 / 5 ≈ 8-12 分钟
"""
from __future__ import annotations
import os, struct, time, json, datetime as dt
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from factor_lab import compute_factors

TDX = os.getenv("TDX_DIR", "D:/tdx")
START = "20240101"  # factor_lab 用 2 年滚动窗口
END   = "20260420"
NEW_START = "20260127"  # 增量起始
NEW_END   = "20260420"

PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
EXT_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups_extension"
EXT_DIR.mkdir(exist_ok=True)


def read_tdx_full(market, code, end_inclusive):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    try: data = p.read_bytes()
    except: return None
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
    if not rows: return None
    df = pd.DataFrame(rows)
    df["pre_close"] = df["close"].shift(1).bfill()
    df["change"] = df["close"] - df["pre_close"]
    df["pct_chg"] = df["change"] / df["pre_close"] * 100
    return df[df["trade_date"] >= START].reset_index(drop=True)


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def process_one_stock(args):
    """单股处理 (worker 函数)."""
    ts_code, template_row, base_cols = args
    code, market = code_market(ts_code)
    daily = read_tdx_full(market, code, END)
    if daily is None or len(daily) < 60:
        return None
    try:
        factors = compute_factors(daily)
    except Exception:
        return None
    factors["ts_code"] = ts_code
    factors_new = factors[(factors["trade_date"] >= NEW_START) &
                            (factors["trade_date"] <= NEW_END)].copy()
    if factors_new.empty:
        return None
    # 填模板基础列
    for col in base_cols:
        if col not in factors_new.columns:
            factors_new[col] = template_row.get(col, pd.NA)
    return factors_new


def worker_group(args):
    """单 group 处理器."""
    group_path, ts_codes, templates, base_cols = args
    rows = []
    for ts in ts_codes:
        tpl = templates.get(ts, {})
        result = process_one_stock((ts, tpl, base_cols))
        if result is not None:
            rows.append(result)
    if not rows:
        return group_path.stem, 0
    new_df = pd.concat(rows, ignore_index=True)
    out_path = EXT_DIR / f"{group_path.stem}_ext.parquet"
    new_df.to_parquet(out_path, index=False)
    return group_path.stem, len(new_df)


def main():
    t0 = time.time()
    print("=== 全市场 factor_lab 153 因子增量更新 ===\n", flush=True)

    # 1. 读现有 parquet, 提取每股最新一行作为模板 (industry/pe 等基础列)
    print("准备模板 (5072 股 industry/pe/total_mv 等基础列)...", flush=True)
    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
    print(f"  factor_groups: {len(parquet_files)} 个文件", flush=True)

    # 2. 准备 worker 任务
    tasks = []
    sample_cols = None
    for p in parquet_files:
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        if sample_cols is None:
            sample_cols = df.columns.tolist()
        # 每股最新一行
        last_rows = df.sort_values(["ts_code","trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
        templates = {row["ts_code"]: row.to_dict() for _, row in last_rows.iterrows()}
        ts_codes = sorted(templates.keys())
        # 基础列 = 模板列 - factor_lab compute_factors 输出列 (前者多)
        # 不知道 compute_factors 完整输出列, 用 'tradedate' + 'ts_code' 排除其他即可
        # 简化: 把 compute_factors 没输出的列都视为 base_cols
        # 实际更简单: 模板有 industry/pe/total_mv 等, factor_new 没有, 直接全填模板
        base_cols = list(templates[ts_codes[0]].keys())
        tasks.append((p, ts_codes, templates, base_cols))
    n_total = sum(len(t[1]) for t in tasks)
    print(f"  共 {n_total} 股待处理 ({len(tasks)} 个 group)", flush=True)

    # 3. 5 worker 并行
    print(f"\n启动 5 worker 并行...", flush=True)
    with mp.Pool(processes=5) as pool:
        results = []
        for i, (group_id, n_rows) in enumerate(pool.imap_unordered(worker_group, tasks), 1):
            results.append((group_id, n_rows))
            elapsed = time.time() - t0
            print(f"  [{i}/{len(tasks)}] {group_id}: {n_rows} 行 ({elapsed:.0f}s)", flush=True)

    total_new = sum(r[1] for r in results)
    print(f"\n=== 完成 ===", flush=True)
    print(f"  总新增数据: {total_new:,} 行 (5072 股 × 60 交易日 ≈ 300K)", flush=True)
    print(f"  输出目录: {EXT_DIR}/", flush=True)
    print(f"  总耗时: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
