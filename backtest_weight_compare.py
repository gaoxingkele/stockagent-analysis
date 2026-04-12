# -*- coding: utf-8 -*-
"""权重方案对比 — V4(当前) vs V5(三因子融合) vs V6(融合+中性零权重)。"""
import sys, os, time, statistics

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


# ── V4 权重 (当前已提交版本) ──
W_V4 = {
    "channel_reversal": 0.19,
    "chanlun":          0.18,
    "divergence":       0.18,
    "trend_momentum":   0.14,
    "capital_liquidity":0.09,
    "f_amt_ratio":      0.09,
    "ichimoku":         0.05,
    "resonance_rev":    0.08,
}
KEY_V4 = {
    "chanlun": 0.15, "channel_reversal": 0.12,
    "divergence": 0.10, "capital_liquidity": 0.08,
}

# ── V5: 三因子融合为 mean_reversion ──
# trend_momentum(ρ=0.79 ichimoku, 0.73 res_rev) + ichimoku + resonance_rev → 均值
# 释放的权重分给独立因子(capital_liquidity IC=+0.051最高)
W_V5 = {
    "channel_reversal": 0.20,
    "divergence":       0.20,
    "capital_liquidity":0.15,
    "mean_reversion":   0.15,   # 三合一
    "f_amt_ratio":      0.12,
    "chanlun":          0.18,
}
KEY_V5 = {
    "channel_reversal": 0.12, "divergence": 0.10,
    "capital_liquidity": 0.10, "chanlun": 0.10,
}

# ── V5b: 同V5但capital_liquidity更激进 ──
W_V5b = {
    "capital_liquidity":0.20,
    "channel_reversal": 0.20,
    "divergence":       0.20,
    "mean_reversion":   0.15,
    "f_amt_ratio":      0.12,
    "chanlun":          0.13,
}
KEY_V5b = KEY_V5.copy()

# ── V6: V5 + 中性零权重 ──
# (动态计算, 不需要静态权重表, 在composite函数中实现)
W_V6_BASE = W_V5.copy()
KEY_V6 = KEY_V5.copy()
NEUTRAL_THRESHOLD = 5.0  # |score - 50| <= 5 → 权重渐减


def key_bonus(scores, key_dims):
    b = 0.0
    for dim, pull in key_dims.items():
        s = scores.get(dim, 50.0)
        if s > 75: b += (s - 75) * pull
        elif s < 25: b -= (25 - s) * pull
    return b


def composite_static(scores, weights, key_dims):
    raw = sum(scores.get(k, 50.0) * w for k, w in weights.items())
    return max(0.0, min(100.0, raw + key_bonus(scores, key_dims)))


def composite_dynamic(scores, base_weights, key_dims, threshold=5.0):
    """中性零权重: |score-50| <= threshold 时权重渐减到0, 释放的权重重分配。"""
    eff_weights = {}
    for k, w in base_weights.items():
        s = scores.get(k, 50.0)
        dev = abs(s - 50.0)
        if dev >= threshold:
            eff_weights[k] = w
        else:
            # 渐进: dev=0→0%, dev=threshold→100%
            eff_weights[k] = w * (dev / threshold)

    total_w = sum(eff_weights.values())
    if total_w < 0.01:
        return 50.0

    # 归一化
    raw = sum(scores.get(k, 50.0) * (ew / total_w) for k, ew in eff_weights.items())
    return max(0.0, min(100.0, raw + key_bonus(scores, key_dims)))


def compute_channel_scores(df):
    try:
        df2 = compute_channel(df.copy())
        df2 = detect_phases(df2)
        return df2["cr_score"].values
    except Exception:
        return np.full(len(df), 50.0)


def stretch(records):
    if len(records) < 30:
        return records
    scores = [r[0] for r in records]
    mu = statistics.mean(scores)
    sigma = statistics.stdev(scores)
    if sigma >= 12.0:
        return records
    out = []
    for rec in records:
        z = (rec[0] - mu) / max(sigma, 0.01)
        new = max(5.0, min(95.0, 55.0 + z * 18.0))
        out.append((new,) + rec[1:])
    return out


def analyze(pairs):
    scores = np.array([p[0] for p in pairs])
    r20 = np.array([p[1] for p in pairs])
    m = ~np.isnan(r20)
    ic = float(np.corrcoef(scores[m], r20[m])[0, 1]) if m.sum() > 30 else 0
    mu = float(np.mean(scores))
    sigma = float(np.std(scores))

    bins = [(0, 30), (30, 45), (45, 55), (55, 65), (65, 75), (75, 85), (85, 100)]
    bks = []
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi) & m
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        sr = r20[mask]
        bks.append({
            "range": f"{lo}-{hi}", "count": cnt,
            "avg": float(np.mean(sr)), "wr": float((sr > 0).mean() * 100),
        })
    return ic, mu, sigma, bks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0)
    args = parser.parse_args()

    pool = Path("backtest_stock_pool.txt")
    symbols = [l.strip() for l in pool.read_text().splitlines() if l.strip()]
    if args.max > 0:
        symbols = symbols[:args.max]

    print(f"V4/V5/V5b/V6 对比 — {len(symbols)} 只股票", flush=True)
    t0 = time.time()

    all_data = {k: [] for k in ("V4", "V5", "V5b", "V6")}

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
                r20 = (close_arr[idx + 20] / close_arr[idx] - 1) * 100 if idx + 20 < len(close_arr) else np.nan

                tm = score_trend_momentum(row)
                ich = score_ichimoku(row)
                res_rev = 100.0 - score_resonance(row)
                # 三因子融合: 简单均值
                mr = (tm + ich + res_rev) / 3.0

                base = {
                    "channel_reversal": float(cr_arr[idx]) if idx < len(cr_arr) else 50.0,
                    "chanlun":          score_chanlun(row),
                    "divergence":       score_divergence(row),
                    "capital_liquidity": score_capital_liquidity(row),
                    "f_amt_ratio":      max(10, min(90, _f_amt_ratio(row))),
                }

                # V4: 当前版本 (分开三因子)
                s4 = dict(base)
                s4["trend_momentum"] = tm
                s4["ichimoku"] = ich
                s4["resonance_rev"] = res_rev
                all_data["V4"].append((composite_static(s4, W_V4, KEY_V4), r20))

                # V5/V5b: 三合一
                s5 = dict(base)
                s5["mean_reversion"] = mr
                all_data["V5"].append((composite_static(s5, W_V5, KEY_V5), r20))
                all_data["V5b"].append((composite_static(s5, W_V5b, KEY_V5b), r20))

                # V6: V5 + 中性零权重
                all_data["V6"].append((composite_dynamic(s5, W_V6_BASE, KEY_V6, NEUTRAL_THRESHOLD), r20))

        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"\n样本: {len(all_data['V4']):,}  用时: {elapsed:.0f}s", flush=True)

    labels = {
        "V4": "V4 当前(三因子分开, 27%权重冗余)",
        "V5": "V5 三因子融合(mean_reversion 15%)",
        "V5b": "V5b 融合+capital_liquidity加权至20%",
        "V6": "V6 V5+中性零权重(|s-50|≤5渐减)",
    }
    for name in ("V4", "V5", "V5b", "V6"):
        data_s = stretch(all_data[name])
        ic, mu, sigma, bks = analyze(data_s)
        print(f"\n{'=' * 70}")
        print(f"{labels[name]}  IC(20d)={ic:+.4f}  μ={mu:.2f}  σ={sigma:.2f}")
        for b in bks:
            pct = b["count"] / len(all_data[name]) * 100
            flag = " ◀" if int(b["range"].split("-")[0]) >= 65 else ""
            print(f"  {b['range']:>8} | {b['count']:>6,} ({pct:>4.1f}%) | 20d={b['avg']:+.2f}% | wr={b['wr']:.1f}%{flag}")


if __name__ == "__main__":
    main()
