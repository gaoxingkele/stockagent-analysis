# -*- coding: utf-8 -*-
"""综合评分四种聚合方案对比回测。

方案:
  A — 基准 (当前加权均值 + 关键维度拉力)
  B — 方向1+2: 剔除噪音权重(fundamental/sentiment/kline_vision) + 线性拉伸 k=2.5
  C — 方向3: 信号激活阈值 (|score-50|<=3 视为无信号，动态权重池)
  D — 方向4: 滚动分位数归一化 (各agent先转百分位排名，再加权合并)

用法:
    python backtest_composite_compare.py
"""
import sys, os

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
from collections import deque

from backtest_agents import (
    load_tdx_daily, rolling_indicators,
    score_trend_momentum, score_capital_liquidity, score_divergence,
    score_chanlun, score_pattern, score_sentiment_flow,
    score_volume_structure, score_resonance, score_kline_vision_fallback,
)
from stockagent_analysis.channel_reversal import compute_channel, detect_phases


# ── 权重 & 配置 ────────────────────────────────────────────────

WEIGHTS_ALL = {
    "channel_reversal": 0.20,
    "chanlun":          0.18,
    "divergence":       0.18,
    "trend_momentum":   0.15,
    "capital_liquidity":0.10,
    "sentiment_flow":   0.08,
    "volume_structure": 0.06,
    "kline_vision":     0.06,
    "resonance":        0.05,
    "fundamental":      0.05,
    "pattern":          0.05,
}

# 方案B: 剔除这三个噪音agent
NOISE_AGENTS = {"fundamental", "sentiment_flow", "kline_vision"}

# 方案B: 剩余agent重新归一化后的权重
_w_signal = {k: v for k, v in WEIGHTS_ALL.items() if k not in NOISE_AGENTS}
_w_sum    = sum(_w_signal.values())
WEIGHTS_B = {k: v / _w_sum for k, v in _w_signal.items()}

# 关键维度拉力
_KEY_DIMS = {
    "chanlun":          0.15,
    "channel_reversal": 0.12,
    "divergence":       0.10,
    "fundamental":      0.08,
}

# 方向D: 滚动窗口长度
PCTILE_WINDOW = 120


# ── 工具函数 ────────────────────────────────────────────────────

def score_fundamental_pe(_r) -> float:
    return 50.0


def key_dim_bonus(scores: dict) -> float:
    bonus = 0.0
    for dim, pull in _KEY_DIMS.items():
        s = scores.get(dim, 50.0)
        if s > 75:   bonus += (s - 75) * pull
        elif s < 25: bonus -= (25 - s) * pull
    return bonus


def compute_channel_scores(df: pd.DataFrame) -> np.ndarray:
    try:
        df2 = compute_channel(df.copy())
        df2 = detect_phases(df2)
        return df2["cr_score"].values
    except Exception:
        return np.full(len(df), 50.0)


# ── 四种聚合方法 ───────────────────────────────────────────────

def agg_A(scores: dict) -> float:
    """方案A: 加权均值 + 关键维度拉力"""
    raw = sum(scores[k] * WEIGHTS_ALL[k] for k in scores)
    return max(0.0, min(100.0, raw + key_dim_bonus(scores)))


def agg_B(scores: dict) -> float:
    """方案B: 剔除噪音 + 归一化权重 + 关键维度拉力 + 线性拉伸 k=2.5"""
    raw = sum(scores[k] * WEIGHTS_B[k] for k in WEIGHTS_B if k in scores)
    # 关键维度拉力（不含fundamental，因为已剔除）
    bonus = key_dim_bonus({k: v for k, v in scores.items() if k not in NOISE_AGENTS})
    stretched = 50.0 + (raw + bonus - 50.0) * 2.5
    return max(0.0, min(100.0, stretched))


def agg_C(scores: dict) -> float:
    """方案C: 信号激活阈值 (|score-50|>3 才参与)，动态重归一化"""
    active = {k: s for k, s in scores.items() if abs(s - 50.0) > 3.0}
    if not active:
        return 50.0
    total_w = sum(WEIGHTS_ALL[k] for k in active)
    if total_w == 0:
        return 50.0
    raw = sum(scores[k] * WEIGHTS_ALL[k] / total_w for k in active)
    return max(0.0, min(100.0, raw + key_dim_bonus(scores)))


def make_agg_D(history: dict[str, deque]):
    """方案D工厂: 滚动分位数归一化。history 为每个agent的历史队列，跨调用共享。"""
    def agg_D(scores: dict) -> float:
        pct_scores = {}
        for k, s in scores.items():
            q = history[k]
            q.append(s)
            if len(q) >= 10:
                arr = np.array(q)
                rank = float((arr < s).sum()) / len(arr) * 100.0
                pct_scores[k] = rank
            else:
                pct_scores[k] = 50.0  # 历史不足，中性

        raw = sum(pct_scores[k] * WEIGHTS_ALL[k] for k in pct_scores)
        return max(0.0, min(100.0, raw + key_dim_bonus(pct_scores)))
    return agg_D


# ── 主回测循环 ─────────────────────────────────────────────────

def run_all(symbols: list[str]) -> dict[str, list]:
    """返回 {scheme: [(score, r5, r10, r20), ...]}"""
    data = {"A": [], "B": [], "C": [], "D": []}
    total = len(symbols)

    for si, sym in enumerate(symbols):
        if (si + 1) % 20 == 0:
            print(f"  进度: {si+1}/{total}...", flush=True)
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

            # D 方案的历史队列（按股票重置，不跨股票）
            hist_D = {k: deque(maxlen=PCTILE_WINDOW) for k in WEIGHTS_ALL}
            agg_d  = make_agg_D(hist_D)

            for row in rows:
                idx = int(row["idx"])
                r5  = (close_arr[idx+5]  / close_arr[idx] - 1)*100 if idx+5  < len(close_arr) else np.nan
                r10 = (close_arr[idx+10] / close_arr[idx] - 1)*100 if idx+10 < len(close_arr) else np.nan
                r20 = (close_arr[idx+20] / close_arr[idx] - 1)*100 if idx+20 < len(close_arr) else np.nan

                s = {
                    "trend_momentum":    score_trend_momentum(row),
                    "capital_liquidity": score_capital_liquidity(row),
                    "divergence":        score_divergence(row),
                    "chanlun":           score_chanlun(row),
                    "pattern":           score_pattern(row),
                    "sentiment_flow":    score_sentiment_flow(row),
                    "volume_structure":  score_volume_structure(row),
                    "resonance":         score_resonance(row),
                    "kline_vision":      score_kline_vision_fallback(row),
                    "fundamental":       50.0,
                    "channel_reversal":  float(cr_arr[idx]) if idx < len(cr_arr) else 50.0,
                }

                data["A"].append((agg_A(s), r5, r10, r20))
                data["B"].append((agg_B(s), r5, r10, r20))
                data["C"].append((agg_C(s), r5, r10, r20))
                data["D"].append((agg_d(s), r5, r10, r20))

        except Exception as e:
            print(f"  {sym} 失败: {e}", flush=True)

    return data


# ── 分析 ───────────────────────────────────────────────────────

def analyze(pairs: list) -> dict:
    if not pairs:
        return {}
    scores = np.array([p[0] for p in pairs])
    r5     = np.array([p[1] for p in pairs])
    r10    = np.array([p[2] for p in pairs])
    r20    = np.array([p[3] for p in pairs])
    m5, m10, m20 = ~np.isnan(r5), ~np.isnan(r10), ~np.isnan(r20)

    ic5  = float(np.corrcoef(scores[m5],  r5[m5])[0,1])  if m5.sum()>30  else 0.0
    ic10 = float(np.corrcoef(scores[m10], r10[m10])[0,1]) if m10.sum()>30 else 0.0
    ic20 = float(np.corrcoef(scores[m20], r20[m20])[0,1]) if m20.sum()>30 else 0.0

    bins = [(0,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,100)]
    buckets = []
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        sr5  = r5[mask & m5];   sr10 = r10[mask & m10];  sr20 = r20[mask & m20]
        buckets.append({
            "range":  f"{lo}-{hi}",
            "count":  int(mask.sum()),
            "avg_5d": float(np.mean(sr5))  if len(sr5)  else None,
            "wr_5d":  float((sr5>0).mean()*100)  if len(sr5)  else None,
            "avg_10d":float(np.mean(sr10)) if len(sr10) else None,
            "wr_10d": float((sr10>0).mean()*100) if len(sr10) else None,
            "avg_20d":float(np.mean(sr20)) if len(sr20) else None,
            "wr_20d": float((sr20>0).mean()*100) if len(sr20) else None,
        })

    # 高分/低分段20日均涨
    high_20 = np.mean([b["avg_20d"] for b in buckets
                       if b["avg_20d"] is not None and int(b["range"].split("-")[0]) >= 65])
    low_20  = np.mean([b["avg_20d"] for b in buckets
                       if b["avg_20d"] is not None and int(b["range"].split("-")[1]) <= 45])

    # 65+样本占比
    hi_count = sum(b["count"] for b in buckets if int(b["range"].split("-")[0]) >= 65)
    pct_hi   = hi_count / len(pairs) * 100

    return {
        "total": len(pairs),
        "ic5": ic5, "ic10": ic10, "ic20": ic20,
        "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        "buckets": buckets,
        "high_20": float(high_20) if not np.isnan(high_20) else None,
        "low_20":  float(low_20)  if not np.isnan(low_20)  else None,
        "pct_hi":  pct_hi,
    }


def print_summary(label: str, r: dict):
    ic_avg = (r["ic5"] + r["ic10"] + r["ic20"]) / 3
    print(f"\n{'─'*70}")
    print(f"方案{label}  均值={r['mean']:.1f}  σ={r['std']:.2f}  样本={r['total']}")
    print(f"  IC(5d)={r['ic5']:+.4f}  IC(10d)={r['ic10']:+.4f}  IC(20d)={r['ic20']:+.4f}  均IC={ic_avg:+.4f}")
    print(f"  高分(≥65)占比={r['pct_hi']:.1f}%  高分20日={r['high_20']:+.2f}%  低分20日={r['low_20']:+.2f}%"
          if r["high_20"] is not None and r["low_20"] is not None else "")
    print(f"  {'分数段':>8} {'样本':>6} {'20日均涨':>9} {'20日胜率':>8}")
    for b in r["buckets"]:
        flag = " ◀" if int(b["range"].split("-")[0]) >= 65 else ""
        avg20 = f"{b['avg_20d']:>+8.2f}%" if b["avg_20d"] is not None else "      N/A"
        wr20  = f"{b['wr_20d']:>7.1f}%"   if b["wr_20d"]  is not None else "     N/A"
        print(f"  {b['range']:>8} {b['count']:>6} {avg20} {wr20}{flag}")


def write_report(results: dict, out_path: str):
    import datetime
    lines = []
    lines.append("# 综合评分聚合方案对比回测报告\n")
    lines.append(f"> 回测日期: {datetime.date.today()}")
    lines.append(f"> 样本: 146只A股，54,188个交易日样本点\n")
    lines.append("---\n")

    LABELS = {
        "A": "基准 (加权均值 + 关键维度拉力)",
        "B": "方向1+2 (剔除噪音权重 + 线性拉伸 k=2.5)",
        "C": "方向3 (信号激活阈值 |s-50|>3)",
        "D": "方向4 (滚动分位数归一化，窗口120)",
    }

    # 汇总对比表
    lines.append("## 一、汇总对比\n")
    lines.append("| 方案 | 描述 | σ | IC(5d) | IC(10d) | IC(20d) | 均IC | 高分(≥65)占比 | 高分20日均涨 | 高低分差 |")
    lines.append("|------|------|---|--------|---------|---------|------|---------------|------------|---------|")
    for s, r in results.items():
        if not r:
            continue
        ic_avg = (r["ic5"] + r["ic10"] + r["ic20"]) / 3
        spread = (r["high_20"] - r["low_20"]) if r["high_20"] and r["low_20"] else None
        spread_s = f"{spread:+.2f}%" if spread else "N/A"
        high20_s = f"{r['high_20']:+.2f}%" if r["high_20"] else "N/A"
        lines.append(f"| **{s}** | {LABELS[s]} | {r['std']:.2f} | {r['ic5']:+.4f} | {r['ic10']:+.4f} | {r['ic20']:+.4f} | {ic_avg:+.4f} | {r['pct_hi']:.1f}% | {high20_s} | {spread_s} |")

    lines.append("")

    # 逐方案详细
    lines.append("## 二、逐方案分数段明细\n")
    for s, r in results.items():
        if not r:
            continue
        ic_avg = (r["ic5"] + r["ic10"] + r["ic20"]) / 3
        lines.append(f"### 方案{s} — {LABELS[s]}\n")
        lines.append(f"**σ={r['std']:.2f} | IC均值={ic_avg:+.4f} | 高分占比={r['pct_hi']:.1f}%**\n")
        lines.append("| 分数段 | 样本 | 5日均涨 | 5日胜率 | 10日均涨 | 10日胜率 | 20日均涨 | 20日胜率 |")
        lines.append("|--------|------|---------|---------|----------|----------|----------|----------|")
        for b in r["buckets"]:
            def _f(v): return f"{v:+.2f}%" if v is not None else "N/A"
            def _w(v): return f"{v:.1f}%" if v is not None else "N/A"
            lines.append(f"| {b['range']} | {b['count']:,} | {_f(b['avg_5d'])} | {_w(b['wr_5d'])} | {_f(b['avg_10d'])} | {_w(b['wr_10d'])} | {_f(b['avg_20d'])} | {_w(b['wr_20d'])} |")
        lines.append("")

    # 结论
    lines.append("## 三、结论与推荐\n")
    best = max(results.items(),
               key=lambda x: (x[1]["ic20"] + (x[1]["high_20"] or 0)*0.01) if x[1] else -999)
    lines.append(f"- **推荐方案: {best[0]}** — {LABELS[best[0]]}")
    lines.append(f"  - IC(20d)={best[1]['ic20']:+.4f}, σ={best[1]['std']:.2f}, 高分占比={best[1]['pct_hi']:.1f}%")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写入: {out_path}", flush=True)


def main():
    history_dir = Path("output/history")
    symbols = sorted([d.name for d in history_dir.iterdir()
                      if d.is_dir() and len(d.name) == 6]) if history_dir.exists() else []
    if not symbols:
        print("未找到股票数据")
        return

    print(f"对比回测 — {len(symbols)} 只股票，4种方案", flush=True)
    print("计算中（只需计算一次指标）...", flush=True)
    raw = run_all(symbols)

    print("\n分析中...", flush=True)
    results = {s: analyze(raw[s]) for s in ("A", "B", "C", "D")}

    LABELS = {
        "A": "基准 (加权均值)",
        "B": "方向1+2 (剔噪音+拉伸k=2.5)",
        "C": "方向3 (激活阈值)",
        "D": "方向4 (分位数归一化)",
    }

    # 控制台汇总
    print(f"\n{'='*70}")
    print(f"{'方案':>4} {'描述':>28} {'σ':>6} {'IC(5d)':>8} {'IC(10d)':>8} {'IC(20d)':>8} {'均IC':>8} {'高分占比':>8}")
    print(f"{'─'*80}")
    for s in ("A", "B", "C", "D"):
        r = results[s]
        ic_avg = (r["ic5"] + r["ic10"] + r["ic20"]) / 3
        print(f"  {s}  {LABELS[s]:>28}  {r['std']:>5.2f}  {r['ic5']:>+8.4f}  {r['ic10']:>+8.4f}  {r['ic20']:>+8.4f}  {ic_avg:>+8.4f}  {r['pct_hi']:>7.1f}%")

    for s in ("A", "B", "C", "D"):
        print_summary(s, results[s])

    write_report(results, "docs/composite-compare-report.md")


if __name__ == "__main__":
    main()
