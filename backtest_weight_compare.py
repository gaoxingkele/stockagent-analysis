# -*- coding: utf-8 -*-
"""V1 vs V2 权重方案对比 — 一次遍历数据同时计算两套权重的综合分。"""
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

# ── V1 权重 (HEAD baseline) ──
W_V1 = {
    "channel_reversal": 0.21,
    "chanlun":          0.19,
    "divergence":       0.19,
    "trend_momentum":   0.15,
    "capital_liquidity":0.10,
    "f_amt_ratio":      0.10,
    "ichimoku":         0.05,
    "resonance":        0.01,
}
KEY_V1 = {
    "chanlun": 0.15, "channel_reversal": 0.12,
    "divergence": 0.10, "capital_liquidity": 0.08,
}

# ── V2 权重 (IC-proportional + resonance_rev heavy) ──
W_V2 = {
    "resonance_rev":    0.22,
    "trend_momentum":   0.16,
    "ichimoku":         0.15,
    "capital_liquidity":0.11,
    "f_amt_ratio":      0.11,
    "channel_reversal": 0.10,
    "divergence":       0.10,
    "chanlun":          0.05,
}
KEY_V2 = {
    "channel_reversal": 0.12, "divergence": 0.10,
    "resonance_rev": 0.10, "capital_liquidity": 0.08, "chanlun": 0.06,
}

# ── V3 权重 (保守调整: chanlun降权+ichimoku升权+resonance_rev适度) ──
W_V3 = {
    "channel_reversal": 0.18,
    "trend_momentum":   0.17,
    "divergence":       0.15,
    "ichimoku":         0.12,
    "chanlun":          0.10,
    "capital_liquidity":0.10,
    "f_amt_ratio":      0.10,
    "resonance_rev":    0.08,
}
KEY_V3 = {
    "channel_reversal": 0.12, "divergence": 0.10,
    "chanlun": 0.10, "capital_liquidity": 0.08,
}

# ── V4 权重 (V1+仅添加resonance_rev 0.08, 其余等比缩减) ──
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
KEY_V4 = KEY_V1.copy()


def key_bonus(scores, key_dims):
    b = 0.0
    for dim, pull in key_dims.items():
        s = scores.get(dim, 50.0)
        if s > 75: b += (s - 75) * pull
        elif s < 25: b -= (25 - s) * pull
    return b


def composite(scores, weights, key_dims):
    raw = sum(scores.get(k, 50.0) * w for k, w in weights.items())
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

    bins = [(0,30),(30,45),(45,55),(55,65),(65,75),(75,85),(85,100)]
    bks = []
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi) & m
        cnt = int(mask.sum())
        if cnt == 0: continue
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

    print(f"V1 vs V2 对比 — {len(symbols)} 只股票", flush=True)
    t0 = time.time()

    schemes = {
        "V1": (W_V1, KEY_V1),
        "V2": (W_V2, KEY_V2),
        "V3": (W_V3, KEY_V3),
        "V4": (W_V4, KEY_V4),
    }
    all_data = {k: [] for k in schemes}

    for si, sym in enumerate(symbols):
        if (si + 1) % 50 == 0:
            print(f"  进度: {si+1}/{len(symbols)}...", flush=True)
        try:
            df = load_tdx_daily(sym)
            if df is None or len(df) < 200: continue
            cr_arr = compute_channel_scores(df)
            ind = rolling_indicators(df, min_bars=120)
            if ind.empty: continue
            close_arr = df["close"].values

            for _, r in ind.iterrows():
                row = r.to_dict()
                idx = int(row["idx"])
                r20 = (close_arr[idx+20] / close_arr[idx] - 1) * 100 if idx + 20 < len(close_arr) else np.nan

                res_raw = score_resonance(row)
                base = {
                    "trend_momentum":    score_trend_momentum(row),
                    "capital_liquidity": score_capital_liquidity(row),
                    "divergence":        score_divergence(row),
                    "chanlun":           score_chanlun(row),
                    "f_amt_ratio":       max(10, min(90, _f_amt_ratio(row))),
                    "ichimoku":          score_ichimoku(row),
                    "channel_reversal":  float(cr_arr[idx]) if idx < len(cr_arr) else 50.0,
                }

                for name, (w, kd) in schemes.items():
                    s = dict(base)
                    if "resonance" in w:
                        s["resonance"] = res_raw
                    if "resonance_rev" in w:
                        s["resonance_rev"] = 100.0 - res_raw
                    all_data[name].append((composite(s, w, kd), r20))

        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"\n样本: {len(all_data['V1']):,}  用时: {elapsed:.0f}s", flush=True)

    labels = {
        "V1": "V1 (HEAD baseline)",
        "V2": "V2 (IC权重+resonance_rev重)",
        "V3": "V3 (保守调整: chanlun↓ ichimoku↑ res_rev适度)",
        "V4": "V4 (V1+仅添加res_rev 0.08)",
    }
    for name in ("V1", "V2", "V3", "V4"):
        data_s = stretch(all_data[name])
        ic, mu, sigma, bks = analyze(data_s)
        print(f"\n{'='*70}")
        print(f"{labels[name]}  IC(20d)={ic:+.4f}  μ={mu:.2f}  σ={sigma:.2f}")
        for b in bks:
            pct = b["count"] / len(all_data[name]) * 100
            flag = " ◀" if int(b["range"].split("-")[0]) >= 65 else ""
            print(f"  {b['range']:>8} | {b['count']:>6,} ({pct:>4.1f}%) | 20d={b['avg']:+.2f}% | wr={b['wr']:.1f}%{flag}")

if __name__ == "__main__":
    main()
