# -*- coding: utf-8 -*-
"""综合评分 V2 — 基于单因子IC重新设计权重。

相对 V1 (HEAD baseline) 的改动:
1. resonance 100-score 反转后加入加权 (原本单因子IC=-0.0624→反转后+0.0624, 最强)
2. chanlun 降权 0.18 → 0.05 (单因子IC仅+0.0112, 性价比最低)
3. divergence 降权 0.18 → 0.10 (单因子IC仅+0.0288, 信号过度集中在中性)
4. ichimoku 升权 0.09 → 0.15 (单因子IC=+0.0429, 被低估)
5. 其余按IC比例微调

用法:
    python backtest_composite_v2.py [--max N]
"""
import sys, os, glob, time, statistics

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_agents import (
    load_tdx_daily, rolling_indicators,
    score_trend_momentum, score_capital_liquidity, score_divergence,
    score_chanlun, score_resonance, score_ichimoku, _f_amt_ratio,
)
from stockagent_analysis.channel_reversal import compute_channel, detect_phases


# ── 新权重 (V2) — 按单因子IC(20d)比例分配 ────────────────────

WEIGHTS_V2 = {
    "resonance_rev":    0.22,  # IC=+0.0624 (反转后), σ=26.21
    "trend_momentum":   0.16,  # IC=+0.0438, σ=37.70
    "ichimoku":         0.15,  # IC=+0.0429, σ=33.40
    "capital_liquidity":0.11,  # IC=+0.0314, σ=10.21
    "f_amt_ratio":      0.11,  # IC=+0.0303, σ=23.08
    "channel_reversal": 0.10,  # IC=+0.0293, σ=17.52
    "divergence":       0.10,  # IC=+0.0288, σ=7.76
    "chanlun":          0.05,  # IC=+0.0112, σ=22.37 (最弱)
}

_KEY_DIMS = {
    "channel_reversal": 0.12,
    "divergence":       0.10,
    "chanlun":          0.06,
    "capital_liquidity":0.08,
    "resonance_rev":    0.10,
}


# ── 股票池 ────────────────────────────────────────────────────

def compute_channel_scores(df):
    try:
        df2 = compute_channel(df.copy())
        df2 = detect_phases(df2)
        return df2["cr_score"].values
    except Exception:
        return np.full(len(df), 50.0)


def key_dim_bonus(scores):
    bonus = 0.0
    for dim, pull in _KEY_DIMS.items():
        s = scores.get(dim, 50.0)
        if s > 75:
            bonus += (s - 75) * pull
        elif s < 25:
            bonus -= (25 - s) * pull
    return bonus


def composite_score_v2(row_scores):
    raw = sum(row_scores.get(k, 50.0) * w for k, w in WEIGHTS_V2.items())
    return max(0.0, min(100.0, raw + key_dim_bonus(row_scores)))


# ── 展宽 (与 v1 相同) ───────────────────────────────────────────

_STRETCH_SIGMA_THRESHOLD = 12.0
_STRETCH_TARGET_SIGMA = 18.0
_STRETCH_TARGET_MEAN = 55.0


def batch_stretch(records):
    if len(records) < 30:
        return records
    scores = [r[0] for r in records]
    mu = statistics.mean(scores)
    sigma = statistics.stdev(scores)
    print(f"[展宽] 原始: μ={mu:.2f}, σ={sigma:.2f}", flush=True)
    if sigma >= _STRETCH_SIGMA_THRESHOLD:
        return records
    out = []
    for rec in records:
        z = (rec[0] - mu) / max(sigma, 0.01)
        new = max(5.0, min(95.0, _STRETCH_TARGET_MEAN + z * _STRETCH_TARGET_SIGMA))
        out.append((new,) + rec[1:])
    new_scores = [r[0] for r in out]
    print(f"[展宽] 后: μ={statistics.mean(new_scores):.2f}, σ={statistics.stdev(new_scores):.2f}", flush=True)
    return out


# ── 回测 ──────────────────────────────────────────────────────

def run_backtest(symbols):
    records = []
    total = len(symbols)
    for si, sym in enumerate(symbols):
        if (si + 1) % 50 == 0:
            print(f"  进度: {si+1}/{total} (样本={len(records)})...", flush=True)
        try:
            df = load_tdx_daily(sym)
            if df is None or len(df) < 200:
                continue
            cr_arr = compute_channel_scores(df)
            ind = rolling_indicators(df, min_bars=120)
            if ind.empty:
                continue
            close_arr = df["close"].values
            rows = [r.to_dict() for _, r in ind.iterrows()]

            for row in rows:
                idx = int(row["idx"])
                r5  = (close_arr[idx+5]  / close_arr[idx] - 1)*100 if idx+5  < len(close_arr) else np.nan
                r10 = (close_arr[idx+10] / close_arr[idx] - 1)*100 if idx+10 < len(close_arr) else np.nan
                r20 = (close_arr[idx+20] / close_arr[idx] - 1)*100 if idx+20 < len(close_arr) else np.nan

                # resonance_rev = 100 - resonance (反转因子, 单因子IC +0.0624)
                res_raw = score_resonance(row)
                res_rev = 100.0 - res_raw

                agent_scores = {
                    "trend_momentum":    score_trend_momentum(row),
                    "capital_liquidity": score_capital_liquidity(row),
                    "divergence":        score_divergence(row),
                    "chanlun":           score_chanlun(row),
                    "f_amt_ratio":       max(10, min(90, _f_amt_ratio(row))),
                    "ichimoku":          score_ichimoku(row),
                    "resonance_rev":     res_rev,
                    "channel_reversal":  float(cr_arr[idx]) if idx < len(cr_arr) else 50.0,
                }
                score = composite_score_v2(agent_scores)
                records.append((score, r5, r10, r20))
        except Exception as e:
            print(f"  {sym} 失败: {e}", flush=True)
    return records


def analyze(pairs):
    if not pairs:
        return {}
    scores = np.array([p[0] for p in pairs])
    r5 = np.array([p[1] for p in pairs])
    r10 = np.array([p[2] for p in pairs])
    r20 = np.array([p[3] for p in pairs])
    m5, m10, m20 = ~np.isnan(r5), ~np.isnan(r10), ~np.isnan(r20)

    ic5  = float(np.corrcoef(scores[m5],  r5[m5])[0,1])  if m5.sum()>30  else 0.0
    ic10 = float(np.corrcoef(scores[m10], r10[m10])[0,1]) if m10.sum()>30 else 0.0
    ic20 = float(np.corrcoef(scores[m20], r20[m20])[0,1]) if m20.sum()>30 else 0.0

    bins = [(0,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),
            (65,70),(70,75),(75,80),(80,85),(85,90),(90,100)]
    buckets = []
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        sr5 = r5[mask & m5]; sr10 = r10[mask & m10]; sr20 = r20[mask & m20]
        buckets.append({
            "range": f"{lo}-{hi}",
            "count": cnt,
            "avg_20d": float(np.mean(sr20)) if len(sr20) else None,
            "wr_20d": float((sr20 > 0).mean() * 100) if len(sr20) else None,
        })
    return {
        "total": len(pairs),
        "ic5": ic5, "ic10": ic10, "ic20": ic20,
        "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        "buckets": buckets,
    }


def print_result(label, result):
    ic_avg = (result["ic5"] + result["ic10"] + result["ic20"]) / 3
    print(f"\n{'='*85}")
    print(f"{label}  样本={result['total']:,}  μ={result['mean']:.2f}  σ={result['std']:.2f}")
    print(f"IC(5d)={result['ic5']:+.4f}  IC(10d)={result['ic10']:+.4f}  IC(20d)={result['ic20']:+.4f}  均IC={ic_avg:+.4f}")
    print(f"{'='*85}")
    for b in result["buckets"]:
        pct = b["count"] / result["total"] * 100
        lo = int(b["range"].split("-")[0])
        flag = " ◀" if lo >= 65 else ""
        a20 = f"{b['avg_20d']:>+7.2f}%" if b["avg_20d"] is not None else "    N/A"
        w20 = f"{b['wr_20d']:>6.1f}%" if b["wr_20d"] is not None else "   N/A"
        print(f"  {b['range']:>8} | {b['count']:>6,} ({pct:>4.1f}%) | 20d均涨={a20} | 胜率={w20}{flag}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    pool_file = Path("backtest_stock_pool.txt")
    symbols = [l.strip() for l in pool_file.read_text().splitlines() if l.strip()]
    if args.max > 0:
        symbols = symbols[:args.max]

    print(f"综合评分 V2 回测 — {len(symbols)} 只股票", flush=True)
    print(f"权重: {WEIGHTS_V2}", flush=True)
    t0 = time.time()

    pairs_raw = run_backtest(symbols)
    print(f"\n样本: {len(pairs_raw):,}  用时: {time.time()-t0:.0f}s", flush=True)

    res_raw = analyze(pairs_raw)
    print_result("V2 原始", res_raw)

    pairs_str = batch_stretch(pairs_raw)
    res_str = analyze(pairs_str)
    print_result("V2 展宽后", res_str)


if __name__ == "__main__":
    main()
