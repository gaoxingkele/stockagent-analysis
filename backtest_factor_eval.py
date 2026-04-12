# -*- coding: utf-8 -*-
"""单因子独立评估 — 每个因子独立打分，按分数段统计20日收益和胜率（多空双向视角）。"""
import sys, os, time, argparse, statistics
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
    sys.path.insert(0, ROOT)

import numpy as np
from pathlib import Path
from backtest_agents import (
    load_tdx_daily, rolling_indicators,
    score_trend_momentum, score_capital_liquidity, score_divergence,
    score_chanlun, score_resonance, score_ichimoku, _f_amt_ratio,
)
from stockagent_analysis.channel_reversal import compute_channel, detect_phases

FACTORS = {
    "channel_reversal": None,  # 特殊处理
    "chanlun":          score_chanlun,
    "divergence":       score_divergence,
    "trend_momentum":   score_trend_momentum,
    "capital_liquidity": score_capital_liquidity,
    "f_amt_ratio":      lambda r: max(10, min(90, _f_amt_ratio(r))),
    "ichimoku":         score_ichimoku,
    "resonance":        score_resonance,
}

BINS = [(0,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,85),(85,90),(90,100)]

def compute_channel_scores(df):
    try:
        df2 = compute_channel(df.copy())
        df2 = detect_phases(df2)
        return df2["cr_score"].values
    except Exception:
        return np.full(len(df), 50.0)

def analyze_factor(pairs):
    scores = np.array([p[0] for p in pairs])
    r20 = np.array([p[1] for p in pairs])
    m20 = ~np.isnan(r20)
    ic20 = float(np.corrcoef(scores[m20], r20[m20])[0,1]) if m20.sum()>30 else 0
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    buckets = []
    for lo, hi in BINS:
        mask = (scores >= lo) & (scores < hi)
        cnt = int(mask.sum())
        if cnt == 0: continue
        sr20 = r20[mask & m20]
        avg20 = float(np.mean(sr20)) if len(sr20) else None
        wr20 = float((sr20>0).mean()*100) if len(sr20) else None
        buckets.append({"lo": lo, "hi": hi, "count": cnt, "avg_20d": avg20, "wr_20d": wr20})
    return {"ic20": ic20, "mean": mu, "std": sigma, "total": len(pairs), "buckets": buckets}

def print_factor_report(name, result):
    total = result["total"]
    print(f"\n{'='*80}")
    print(f"  因子: {name}  |  σ={result['std']:.2f}  |  IC(20d)={result['ic20']:+.4f}  |  样本={total:,}")
    print(f"{'='*80}")
    print(f"  {'分数段':>8} | {'占比':>5} | {'样本':>7} | {'操作':>10} | {'做多20d':>8} | {'做多胜率':>8} | {'做空20d':>8} | {'做空胜率':>8}")
    print(f"  {'-'*76}")
    for b in result["buckets"]:
        lo = b["lo"]
        pct = b["count"] / total * 100
        a20 = b["avg_20d"]
        w20 = b["wr_20d"]
        if lo < 35:
            # 做空区
            op = "融券做空"
            long_r = "—"
            long_w = "—"
            short_r = f"{-a20:+.2f}%" if a20 is not None else "N/A"
            short_w = f"{100-w20:.1f}%" if w20 is not None else "N/A"
        elif lo < 45:
            op = "空仓"
            long_r = "—"
            long_w = "—"
            short_r = "—"
            short_w = "—"
        elif lo < 65:
            op = "观察"
            long_r = "—"
            long_w = "—"
            short_r = "—"
            short_w = "—"
        else:
            op = "做多"
            long_r = f"{a20:+.2f}%" if a20 is not None else "N/A"
            long_w = f"{w20:.1f}%" if w20 is not None else "N/A"
            short_r = "—"
            short_w = "—"
        rng = f"{lo}-{b['hi']}"
        print(f"  {rng:>8} | {pct:>4.1f}% | {b['count']:>7,} | {op:>10} | {long_r:>8} | {long_w:>8} | {short_r:>8} | {short_w:>8}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=200)
    args = parser.parse_args()

    pool_file = Path("backtest_stock_pool.txt")
    symbols = [l.strip() for l in pool_file.read_text().splitlines() if l.strip()][:args.max]
    print(f"单因子独立评估 — {len(symbols)}只股票", flush=True)

    # 一次性计算所有数据
    all_rows = []  # [(row_dict, cr_score_val, r20), ...]
    t0 = time.time()
    for si, sym in enumerate(symbols):
        if (si+1) % 50 == 0:
            print(f"  进度: {si+1}/{len(symbols)}...", flush=True)
        try:
            df = load_tdx_daily(sym)
            if df is None or len(df) < 200:
                continue
            cr_scores = compute_channel_scores(df)
            ind = rolling_indicators(df, min_bars=120)
            if ind.empty:
                continue
            close_arr = df["close"].values
            for _, r in ind.iterrows():
                row = r.to_dict()
                idx = int(row["idx"])
                r20 = (close_arr[idx+20]/close_arr[idx]-1)*100 if idx+20<len(close_arr) else np.nan
                cr_val = float(cr_scores[idx]) if idx<len(cr_scores) else 50.0
                all_rows.append((row, cr_val, r20))
        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"数据计算完成: {len(all_rows):,}样本, {elapsed:.0f}s", flush=True)

    # 逐因子评估
    for fname, score_fn in FACTORS.items():
        if fname == "channel_reversal":
            pairs = [(row[1], row[2]) for row in all_rows]  # cr_val, r20
        else:
            pairs = [(score_fn(row[0]), row[2]) for row in all_rows]
        result = analyze_factor(pairs)
        print_factor_report(fname, result)

if __name__ == "__main__":
    main()
