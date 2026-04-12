# -*- coding: utf-8 -*-
"""因子相关性矩阵分析 — 计算8因子两两Pearson相关 + 与20日收益的相关。"""
import sys, os, time

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
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


FACTORS = [
    ("channel_reversal", None),
    ("chanlun",          score_chanlun),
    ("divergence",       score_divergence),
    ("trend_momentum",   score_trend_momentum),
    ("capital_liquidity", score_capital_liquidity),
    ("f_amt_ratio",      lambda r: max(10, min(90, _f_amt_ratio(r)))),
    ("ichimoku",         score_ichimoku),
    ("resonance_rev",    lambda r: 100.0 - score_resonance(r)),
]
NAMES = [f[0] for f in FACTORS]


def compute_channel_scores(df):
    try:
        df2 = compute_channel(df.copy())
        df2 = detect_phases(df2)
        return df2["cr_score"].values
    except Exception:
        return np.full(len(df), 50.0)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=400)
    args = parser.parse_args()

    pool = Path("backtest_stock_pool.txt")
    symbols = [l.strip() for l in pool.read_text().splitlines() if l.strip()][:args.max]
    print(f"因子相关性分析 — {len(symbols)} 只股票", flush=True)

    n_factors = len(FACTORS)
    all_scores = [[] for _ in range(n_factors)]
    all_r20 = []
    t0 = time.time()

    for si, sym in enumerate(symbols):
        if (si + 1) % 50 == 0:
            print(f"  进度: {si+1}/{len(symbols)}...", flush=True)
        try:
            df = load_tdx_daily(sym)
            if df is None or len(df) < 200:
                continue
            cr_arr = compute_channel_scores(df)
            ind = rolling_indicators(df, min_bars=120)
            if ind.empty:
                continue
            close_arr = df["close"].values

            for _, r in ind.iterrows():
                row = r.to_dict()
                idx = int(row["idx"])
                r20 = (close_arr[idx+20] / close_arr[idx] - 1) * 100 if idx + 20 < len(close_arr) else np.nan

                for fi, (fname, fn) in enumerate(FACTORS):
                    if fname == "channel_reversal":
                        all_scores[fi].append(float(cr_arr[idx]) if idx < len(cr_arr) else 50.0)
                    else:
                        all_scores[fi].append(fn(row))
                all_r20.append(r20)
        except Exception:
            pass

    elapsed = time.time() - t0
    n = len(all_r20)
    print(f"\n样本: {n:,}  用时: {elapsed:.0f}s\n", flush=True)

    # 转numpy
    mat = np.array(all_scores)  # (n_factors, n_samples)
    r20 = np.array(all_r20)
    m20 = ~np.isnan(r20)

    # ── 因子间相关矩阵 ──
    corr = np.corrcoef(mat)
    print("=" * 90)
    print("因子间 Pearson 相关矩阵")
    print("=" * 90)

    # header
    header = f"{'':>20}" + "".join(f"{n:>10}" for n in NAMES)
    print(header)

    for i, name in enumerate(NAMES):
        row_str = f"{name:>20}"
        for j in range(n_factors):
            val = corr[i, j]
            if i == j:
                row_str += f"{'1.000':>10}"
            else:
                marker = " **" if abs(val) > 0.3 else ""
                row_str += f"{val:>+7.3f}{marker}"

        print(row_str)

    # ── 因子与r20的IC ──
    print(f"\n{'='*60}")
    print("因子 vs 20日收益 IC  +  因子σ")
    print(f"{'='*60}")
    for i, name in enumerate(NAMES):
        scores_i = mat[i][m20]
        r20_valid = r20[m20]
        ic = float(np.corrcoef(scores_i, r20_valid)[0, 1])
        sigma = float(np.std(mat[i]))
        # 信号率: |score - 50| > 5 的比例
        sig_rate = float(np.mean(np.abs(mat[i] - 50) > 5)) * 100
        print(f"  {name:>20} | IC(20d)={ic:+.4f} | σ={sigma:>5.1f} | 信号率={sig_rate:>5.1f}%")

    # ── 高相关因子对分析 ──
    print(f"\n{'='*60}")
    print("相关系数 > 0.3 的因子对 (冗余风险)")
    print(f"{'='*60}")
    pairs = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            if abs(corr[i, j]) > 0.3:
                pairs.append((NAMES[i], NAMES[j], corr[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))
    if pairs:
        for a, b, c in pairs:
            print(f"  {a:>20} ~ {b:<20} ρ={c:+.3f}")
    else:
        print("  无 (所有因子对相关系数均 ≤ 0.3)")

    # ── 冗余因子剔除模拟 ──
    print(f"\n{'='*60}")
    print("冗余剔除模拟: 逐个去掉一个因子, 看剩余因子等权综合IC")
    print(f"{'='*60}")

    # 基准: 全因子等权
    composite_all = mat.mean(axis=0)
    ic_all = float(np.corrcoef(composite_all[m20], r20[m20])[0, 1])
    print(f"  {'全部8因子':>20} | IC(20d)={ic_all:+.4f}")

    for drop_i in range(n_factors):
        mask_f = [k for k in range(n_factors) if k != drop_i]
        composite_drop = mat[mask_f].mean(axis=0)
        ic_drop = float(np.corrcoef(composite_drop[m20], r20[m20])[0, 1])
        delta = ic_drop - ic_all
        marker = " ← 剔除后IC提升!" if delta > 0.001 else ""
        print(f"  去掉 {NAMES[drop_i]:>20} | IC(20d)={ic_drop:+.4f} (Δ={delta:+.4f}){marker}")


if __name__ == "__main__":
    main()
