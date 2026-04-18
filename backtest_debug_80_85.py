# -*- coding: utf-8 -*-
"""分析综合评分 80-85 分段的162个样本，输出每个agent的分数分布。"""
import sys, os

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_agents import (
    load_tdx_daily, rolling_indicators,
    score_trend_momentum, score_capital_liquidity, score_divergence,
    score_chanlun, score_pattern, score_sentiment_flow,
    score_volume_structure, score_resonance, score_kline_vision_fallback,
)
from stockagent_analysis.channel_reversal import compute_channel, detect_phases
from backtest_composite import WEIGHTS, _KEY_DIMS, key_dim_dominance, composite_score


def compute_channel_scores(df):
    try:
        df2 = compute_channel(df.copy())
        df2 = detect_phases(df2)
        return df2["cr_score"].values, df2["phase"].values
    except Exception:
        return np.full(len(df), 50.0), np.full(len(df), "?", dtype=object)


AGENT_FUNCS = {
    "trend_momentum":    score_trend_momentum,
    "capital_liquidity": score_capital_liquidity,
    "divergence":        score_divergence,
    "chanlun":           score_chanlun,
    "pattern":           score_pattern,
    "sentiment_flow":    score_sentiment_flow,
    "volume_structure":  score_volume_structure,
    "resonance":         score_resonance,
    "kline_vision":      score_kline_vision_fallback,
}


def main():
    pool_file = Path("backtest_stock_pool.txt")
    symbols = [line.strip() for line in pool_file.read_text().splitlines() if line.strip()]
    print(f"扫描 {len(symbols)} 只股票，收集80-85分段样本...", flush=True)

    records = []
    total = len(symbols)

    for si, sym in enumerate(symbols):
        if (si + 1) % 100 == 0:
            print(f"  进度: {si+1}/{total} (收集到 {len(records)} 条)...", flush=True)
        try:
            df = load_tdx_daily(sym)
            if df is None or len(df) < 200:
                continue

            cr_scores, cr_phases = compute_channel_scores(df)
            ind = rolling_indicators(df, min_bars=120)
            if ind.empty:
                continue

            close_arr = df["close"].values
            rows = [r.to_dict() for _, r in ind.iterrows()]

            for row in rows:
                idx = int(row["idx"])
                agent_scores = {}
                for k, fn in AGENT_FUNCS.items():
                    agent_scores[k] = fn(row)
                agent_scores["fundamental"] = 50.0
                agent_scores["channel_reversal"] = float(cr_scores[idx]) if idx < len(cr_scores) else 50.0

                score = composite_score(agent_scores)

                if 80 <= score < 85:
                    r5 = (close_arr[idx+5] / close_arr[idx] - 1) * 100 if idx + 5 < len(close_arr) else np.nan
                    r20 = (close_arr[idx+20] / close_arr[idx] - 1) * 100 if idx + 20 < len(close_arr) else np.nan
                    phase = cr_phases[idx] if idx < len(cr_phases) else "?"
                    rec = {
                        "symbol": sym,
                        "composite": round(score, 1),
                        "r5": round(r5, 2) if not np.isnan(r5) else None,
                        "r20": round(r20, 2) if not np.isnan(r20) else None,
                        "cr_phase": phase,
                        "key_bonus": round(key_dim_dominance(agent_scores), 2),
                    }
                    for k in WEIGHTS:
                        rec[k] = round(agent_scores[k], 1)
                    records.append(rec)

        except Exception as e:
            pass

    print(f"\n收集到 {len(records)} 个80-85分段样本\n", flush=True)
    if not records:
        return

    df_rec = pd.DataFrame(records)

    # 各agent分数统计
    agent_cols = list(WEIGHTS.keys())
    print("=" * 80)
    print("各Agent分数统计 (80-85分段)")
    print("=" * 80)
    print(f"{'Agent':>22} | {'均值':>6} | {'中位数':>6} | {'std':>6} | {'min':>5} | {'max':>5} | {'权重':>5}")
    print("-" * 80)
    for col in agent_cols:
        vals = df_rec[col]
        print(f"{col:>22} | {vals.mean():>6.1f} | {vals.median():>6.1f} | {vals.std():>6.1f} | {vals.min():>5.0f} | {vals.max():>5.0f} | {WEIGHTS[col]:>5.2f}")

    print(f"\n{'key_bonus':>22} | {df_rec['key_bonus'].mean():>6.2f} | {df_rec['key_bonus'].median():>6.2f} | {df_rec['key_bonus'].std():>6.2f}")

    # Phase分布
    print(f"\nChannel Reversal Phase 分布:")
    phase_counts = df_rec["cr_phase"].value_counts()
    for phase, cnt in phase_counts.items():
        pct = cnt / len(df_rec) * 100
        sub = df_rec[df_rec["cr_phase"] == phase]
        r5_mean = sub["r5"].dropna().mean()
        r20_mean = sub["r20"].dropna().mean()
        print(f"  {phase:>4}: {cnt:>4} ({pct:>5.1f}%) | 5日均涨={r5_mean:+.2f}% | 20日均涨={r20_mean:+.2f}%")

    # 前20个样本明细
    print(f"\n前20个样本明细:")
    print(f"{'sym':>6} {'comp':>5} {'r5':>7} {'r20':>7} {'phase':>5} | {'ch_rev':>6} {'div':>5} {'trend':>5} {'chanlun':>5} {'cap_liq':>5} {'pattern':>5} {'bonus':>6}")
    print("-" * 100)
    for _, r in df_rec.head(20).iterrows():
        r5_s = f"{r['r5']:+.2f}%" if r['r5'] is not None else "  N/A"
        r20_s = f"{r['r20']:+.2f}%" if r['r20'] is not None else "  N/A"
        print(f"{r['symbol']:>6} {r['composite']:>5.1f} {r5_s:>7} {r20_s:>7} {r['cr_phase']:>5} | {r['channel_reversal']:>6.1f} {r['divergence']:>5.1f} {r['trend_momentum']:>5.1f} {r['chanlun']:>5.1f} {r['capital_liquidity']:>5.1f} {r['pattern']:>5.1f} {r['key_bonus']:>+6.2f}")


if __name__ == "__main__":
    main()
