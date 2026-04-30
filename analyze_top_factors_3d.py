"""最有价值因子 × 市值 × PE × 行业 三维表现.

挑选标准:
  - 全市场 |IC(20d)| ≥ 0.025
  - 至少在一个分层维度上 Q5-Q1 胜率差 ≥ 12pp
  - 与其他因子相关性低 (避免重复)

每个因子输出:
  1. 全市场 5 分位胜率
  2. × 5 市值段
  3. × 6 PE 段
  4. × Top 10 行业
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("output/factor_lab")
HOLD = [5, 20, 40]
MV_LABELS = ["20-50亿", "50-100亿", "100-300亿", "300-1000亿", "1000亿+"]
PE_LABELS = ["亏损", "0-15", "15-30", "30-50", "50-100", "100+"]


def load_full() -> pd.DataFrame:
    files = sorted(OUT.glob("factor_groups/group_???.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def bucket_mv(v):
    if pd.isna(v):
        return None
    bil = v / 1e4
    for (lo, hi), lab in zip([(0, 50), (50, 100), (100, 300),
                              (300, 1000), (1000, 1e9)], MV_LABELS):
        if lo <= bil < hi:
            return lab
    return None


def bucket_pe(v):
    if pd.isna(v):
        return None
    if v < 0: return "亏损"
    if v < 15: return "0-15"
    if v < 30: return "15-30"
    if v < 50: return "30-50"
    if v < 100: return "50-100"
    return "100+"


def quantile_stats(sub: pd.DataFrame, factor: str, n_buckets: int = 5):
    x = sub[factor]
    valid = sub[x.notna()]
    if len(valid) < 500:
        return None
    try:
        ranks = valid[factor].rank(pct=True, method="first")
        bucket = pd.cut(ranks, bins=n_buckets, labels=[f"Q{i+1}" for i in range(n_buckets)],
                         include_lowest=True)
    except Exception:
        return None
    rows = []
    for q in [f"Q{i+1}" for i in range(n_buckets)]:
        b = valid[bucket == q]
        if len(b) == 0:
            continue
        r = b["r20"].dropna()
        if len(r) == 0:
            continue
        rows.append({
            "bucket": q,
            "n": len(b),
            "win": float((r > 0).mean()),
            "avg": float(r.mean()),
        })
    return pd.DataFrame(rows) if rows else None


def ic20(sub: pd.DataFrame, factor: str) -> float:
    x = sub[factor]
    r = sub["r20"]
    m = x.notna() & r.notna()
    if m.sum() < 300:
        return np.nan
    xv, yv = x[m].values, r[m].values
    if np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def q5_minus_q1(stats):
    if stats is None or len(stats) < 2:
        return None
    return (stats.iloc[-1]["win"] - stats.iloc[0]["win"]) * 100


def main():
    print("加载...")
    df = load_full()
    print(f"总样本 {len(df):,}")
    df["mv_bucket"] = df["total_mv"].apply(bucket_mv)
    df["pe_bucket"] = df["pe"].apply(bucket_pe)

    # 最有价值的 10 个因子 (基于前期分析人工挑选)
    TOP_FACTORS = [
        ("ht_trendmode", "希尔伯特趋势模式 (0=震荡 1=趋势)"),
        ("ma_ratio_60", "close/MA60-1 (60均线偏离)"),
        ("channel_pos_60", "60日通道位置 (0最低 1最高)"),
        ("sump_20", "20日累计涨幅%"),
        ("ma20_ma60", "MA20/MA60-1"),
        ("mfi_14", "资金流量指标 (大盘动量)"),
        ("rsi_24", "24日 RSI (大盘强势)"),
        ("trix", "三重指数移动平均 (小盘反转)"),
        ("atr_pct", "ATR/close (波动率,PE 反向)"),
        ("macd_hist", "MACD 柱状图"),
    ]

    base_win = (df["r20"] > 0).mean() * 100
    print(f"基准 D+20 胜率 {base_win:.1f}%")

    lines = ["# 10 个最有价值因子 × 市值×PE×行业 三维表现\n\n"]
    lines.append(f"> 总样本 {len(df):,}, 基准 D+20 胜率 **{base_win:.1f}%**\n")
    lines.append(f"> 每因子按值分 5 分位 (Q1=低/Q5=高), 看 Q1 vs Q5 D+20 胜率\n\n")

    # ── 总览表 ──
    lines.append("## 一、因子总览 (全市场)\n\n")
    lines.append("| 因子 | 含义 | 全市场 IC(20d) | 全市场 Q5-Q1 | 最强分层段 | 最强 Q5-Q1 |\n")
    lines.append("|---|---|---|---|---|---|\n")

    overview = []
    for fc, desc in TOP_FACTORS:
        if fc not in df.columns:
            continue
        ic_all = ic20(df, fc)
        stats_all = quantile_stats(df, fc)
        spread_all = q5_minus_q1(stats_all)

        # 找最强分层段
        max_spread, max_seg = 0, "-"
        for mv in MV_LABELS:
            sub = df[df["mv_bucket"] == mv]
            if len(sub) < 5000:
                continue
            s = quantile_stats(sub, fc)
            sp = q5_minus_q1(s)
            if sp is not None and abs(sp) > abs(max_spread):
                max_spread, max_seg = sp, f"市值 {mv}"
        for pe in PE_LABELS:
            sub = df[df["pe_bucket"] == pe]
            if len(sub) < 5000:
                continue
            s = quantile_stats(sub, fc)
            sp = q5_minus_q1(s)
            if sp is not None and abs(sp) > abs(max_spread):
                max_spread, max_seg = sp, f"PE {pe}"

        lines.append(f"| **{fc}** | {desc} | {ic_all:+.4f} | "
                     f"{spread_all:+.2f}pp | {max_seg} | **{max_spread:+.2f}pp** |\n")
        overview.append((fc, desc, ic_all, spread_all, max_seg, max_spread))
    lines.append("\n")

    # ── 每个因子完整 3D ──
    ind_counts = df["industry"].value_counts()
    main_inds = ind_counts[ind_counts >= 30000].index.tolist()[:10]

    for fc, desc in TOP_FACTORS:
        if fc not in df.columns:
            continue
        lines.append(f"## {fc}\n\n")
        lines.append(f"> {desc}\n\n")

        # 全市场
        stats = quantile_stats(df, fc)
        if stats is not None:
            lines.append("### 全市场 5 分位\n\n")
            lines.append("| 桶 | 样本 | D+20 胜率 | D+20 涨幅 |\n|---|---|---|---|\n")
            for _, r in stats.iterrows():
                lines.append(f"| {r['bucket']} | {r['n']:,} | "
                             f"{r['win']*100:.1f}% | {r['avg']:+.2f}% |\n")
            lines.append("\n")

        # 市值分层
        lines.append("### 市值分层 (Q1 vs Q5 胜率)\n\n")
        lines.append("| 市值段 | 样本 | IC(20d) | Q1 胜率 | Q5 胜率 | Q5-Q1 | 用法 |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for mv in MV_LABELS:
            sub = df[df["mv_bucket"] == mv]
            if len(sub) < 5000:
                continue
            ic = ic20(sub, fc)
            s = quantile_stats(sub, fc)
            if s is None or len(s) < 2:
                continue
            q1, q5 = s.iloc[0], s.iloc[-1]
            sp = (q5["win"] - q1["win"]) * 100
            usage = "买Q1" if sp <= -8 else ("买Q5" if sp >= 8 else "弱")
            lines.append(f"| {mv} | {len(sub):,} | {ic:+.4f} | "
                         f"{q1['win']*100:.1f}% | {q5['win']*100:.1f}% | "
                         f"**{sp:+.2f}pp** | {usage} |\n")
        lines.append("\n")

        # PE 分层
        lines.append("### PE 分层 (Q1 vs Q5 胜率)\n\n")
        lines.append("| PE 段 | 样本 | IC(20d) | Q1 胜率 | Q5 胜率 | Q5-Q1 | 用法 |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for pe in PE_LABELS:
            sub = df[df["pe_bucket"] == pe]
            if len(sub) < 5000:
                continue
            ic = ic20(sub, fc)
            s = quantile_stats(sub, fc)
            if s is None or len(s) < 2:
                continue
            q1, q5 = s.iloc[0], s.iloc[-1]
            sp = (q5["win"] - q1["win"]) * 100
            usage = "买Q1" if sp <= -8 else ("买Q5" if sp >= 8 else "弱")
            lines.append(f"| PE {pe} | {len(sub):,} | {ic:+.4f} | "
                         f"{q1['win']*100:.1f}% | {q5['win']*100:.1f}% | "
                         f"**{sp:+.2f}pp** | {usage} |\n")
        lines.append("\n")

        # 行业分层
        lines.append("### 行业分层 (Top 10 主要行业, Q1 vs Q5 胜率)\n\n")
        lines.append("| 行业 | 样本 | IC(20d) | Q1 胜率 | Q5 胜率 | Q5-Q1 | 用法 |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for ind in main_inds:
            sub = df[df["industry"] == ind]
            if len(sub) < 5000:
                continue
            ic = ic20(sub, fc)
            s = quantile_stats(sub, fc)
            if s is None or len(s) < 2:
                continue
            q1, q5 = s.iloc[0], s.iloc[-1]
            sp = (q5["win"] - q1["win"]) * 100
            usage = "买Q1" if sp <= -8 else ("买Q5" if sp >= 8 else "弱")
            lines.append(f"| {ind} | {len(sub):,} | {ic:+.4f} | "
                         f"{q1['win']*100:.1f}% | {q5['win']*100:.1f}% | "
                         f"**{sp:+.2f}pp** | {usage} |\n")
        lines.append("\n---\n\n")

    out_file = OUT / "report_top_factors_3d.md"
    out_file.write_text("".join(lines), encoding="utf-8")
    print(f"\n报告: {out_file}")
    print(f"行数: {len(lines)}")


if __name__ == "__main__":
    main()
