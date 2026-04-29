"""factor_lab 因子 5 分位胜率分析.

每因子按值排序分 5 桶 (Q1=低 / Q5=高), 每桶算:
  - 样本数
  - D+5 / D+20 / D+40 平均涨幅
  - D+5 / D+20 / D+40 胜率
  - max - min 胜率差 (区分力)

输出:
  output/factor_lab/report_winrate_buckets.md     全市场胜率
  output/factor_lab/report_winrate_layered.md     市值×因子分层胜率 (Top 因子)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("output/factor_lab")
HOLD = [5, 20, 40]
MV_LABELS = ["20-50亿", "50-100亿", "100-300亿", "300-1000亿", "1000亿+"]


def load_full(mode: str = "raw") -> pd.DataFrame:
    suf = "" if mode == "raw" else f"_{mode}"
    files = sorted(OUT.glob(f"factor_groups/group_???{suf}.parquet"))
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


def factor_quantile_stats(df: pd.DataFrame, factor: str, n_buckets: int = 5):
    """对单因子做 5 分位分桶, 返回每桶 D+5/20/40 胜率与涨幅."""
    x = df[factor]
    valid = df[x.notna()]
    if len(valid) < 5000:
        return None
    try:
        ranks = valid[factor].rank(pct=True, method="first")
        bucket = pd.cut(ranks, bins=n_buckets, labels=[f"Q{i+1}" for i in range(n_buckets)],
                         include_lowest=True)
    except Exception:
        return None
    rows = []
    for q in [f"Q{i+1}" for i in range(n_buckets)]:
        sub = valid[bucket == q]
        if len(sub) == 0:
            continue
        row = {"bucket": q, "n": len(sub),
                "factor_min": float(sub[factor].min()),
                "factor_max": float(sub[factor].max())}
        for h in HOLD:
            r = sub[f"r{h}"].dropna()
            if len(r) > 0:
                row[f"avg_{h}"] = float(r.mean())
                row[f"win_{h}"] = float((r > 0).mean())
            else:
                row[f"avg_{h}"] = np.nan
                row[f"win_{h}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_ic20(df: pd.DataFrame, factor: str) -> float:
    x = df[factor]
    r = df["r20"]
    m = x.notna() & r.notna()
    if m.sum() < 1000:
        return np.nan
    xv, yv = x[m].values, r[m].values
    if np.std(xv) < 1e-9 or np.std(yv) < 1e-9:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def bucket_pe(v):
    if pd.isna(v):
        return None
    if v < 0:
        return "亏损"
    if v < 15:
        return "0-15"
    if v < 30:
        return "15-30"
    if v < 50:
        return "30-50"
    if v < 100:
        return "50-100"
    return "100+"


PE_LABELS = ["亏损", "0-15", "15-30", "30-50", "50-100", "100+"]


def main():
    print("加载 raw parquet...")
    df = load_full("raw")
    print(f"总样本: {len(df):,}")
    df["mv_bucket"] = df["total_mv"].apply(bucket_mv)
    df["pe_bucket"] = df["pe"].apply(bucket_pe)

    excluded = {"ts_code", "trade_date", "industry", "name",
                "total_mv", "pe", "pe_ttm", "pb", "mv_bucket"} | \
        {f"r{h}" for h in (5, 10, 20, 30, 40)} | \
        {f"dd{h}" for h in (5, 10, 20, 30, 40)} | \
        {"market_score_adj", "adx", "winner_rate", "main_net", "holder_pct",
         "mf_divergence", "mf_strength", "mf_consecutive"}
    factor_cols = [c for c in df.columns if c not in excluded
                    and pd.api.types.is_numeric_dtype(df[c])]

    # ── 1) 计算所有因子 IC, 选 Top |IC| 30 ──
    print("计算 IC...")
    ic_data = []
    for fc in factor_cols:
        ic = compute_ic20(df, fc)
        if not np.isnan(ic):
            ic_data.append({"factor": fc, "ic_20": ic, "abs_ic": abs(ic)})
    ic_df = pd.DataFrame(ic_data).sort_values("abs_ic", ascending=False)
    top_factors = ic_df.head(30)["factor"].tolist()
    print(f"Top 30 因子准备分桶分析...")

    # ── 2) 每因子 5 分位桶 (全市场) ──
    lines = ["# 因子 5 分位桶胜率 (全市场)\n\n"]
    lines.append(f"> Top 30 因子 (按 |IC(20d)|), 总样本 {len(df):,}\n")
    lines.append(f"> Q1 = 因子值最低 20%, Q5 = 最高 20%\n\n")

    base_win5 = (df["r5"] > 0).mean()
    base_win20 = (df["r20"] > 0).mean()
    base_win40 = (df["r40"] > 0).mean()
    lines.append(f"**基准胜率**: D+5={base_win5*100:.1f}% / D+20={base_win20*100:.1f}% "
                 f"/ D+40={base_win40*100:.1f}%\n\n")

    for fc in top_factors:
        stats = factor_quantile_stats(df, fc)
        if stats is None or stats.empty:
            continue
        ic20 = ic_df[ic_df["factor"] == fc]["ic_20"].values[0]

        # 区分力 = Q5 - Q1 胜率差 (D+20)
        try:
            spread_5 = stats.iloc[-1]["win_5"] - stats.iloc[0]["win_5"]
            spread_20 = stats.iloc[-1]["win_20"] - stats.iloc[0]["win_20"]
            spread_40 = stats.iloc[-1]["win_40"] - stats.iloc[0]["win_40"]
        except Exception:
            spread_5 = spread_20 = spread_40 = 0

        lines.append(f"### {fc}  (IC20 = {ic20:+.4f})\n\n")
        lines.append(f"区分力 Q5-Q1: D+5={spread_5*100:+.2f}pp / "
                     f"D+20={spread_20*100:+.2f}pp / D+40={spread_40*100:+.2f}pp\n\n")
        lines.append("| 桶 | 样本 | 因子区间 | D+5 涨幅 | D+5 胜率 | "
                     "D+20 涨幅 | D+20 胜率 | D+40 涨幅 | D+40 胜率 |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in stats.iterrows():
            fmin = r["factor_min"]
            fmax = r["factor_max"]
            interval = f"[{fmin:.3f}, {fmax:.3f}]"
            lines.append(f"| {r['bucket']} | {r['n']:,} | {interval} | "
                         f"{r['avg_5']:+.2f}% | {r['win_5']*100:.1f}% | "
                         f"{r['avg_20']:+.2f}% | {r['win_20']*100:.1f}% | "
                         f"{r['avg_40']:+.2f}% | {r['win_40']*100:.1f}% |\n")
        lines.append("\n")

    out_file = OUT / "report_winrate_buckets.md"
    out_file.write_text("".join(lines), encoding="utf-8")
    print(f"全市场胜率: {out_file}")

    # ── 3) 跨市值分层胜率 (聚焦跨市值反转因子) ──
    flip_factors = [
        "ht_trendmode", "rsi_24", "mfi_14", "trix", "ppo",
        "channel_pos_60", "ma_ratio_60", "sump_20", "sumn_20", "ma20_ma60",
    ]
    lines2 = ["# 跨市值分层胜率 (重点反转因子)\n\n"]
    lines2.append(f"> 每因子 × 5 市值段 × 5 分位桶, 重点看 Q5(高) vs Q1(低)\n\n")

    for fc in flip_factors:
        if fc not in df.columns:
            continue
        lines2.append(f"## {fc}\n\n")
        lines2.append("| 市值段 | 样本 | Q1 D+20 胜率 | Q5 D+20 胜率 | Q5-Q1 差 | "
                      "Q1 D+20 涨幅 | Q5 D+20 涨幅 |\n")
        lines2.append("|---|---|---|---|---|---|---|\n")

        for mv_lab in MV_LABELS:
            sub = df[df["mv_bucket"] == mv_lab]
            if len(sub) < 5000:
                continue
            stats = factor_quantile_stats(sub, fc)
            if stats is None or stats.empty:
                continue
            try:
                q1 = stats.iloc[0]
                q5 = stats.iloc[-1]
                diff = (q5["win_20"] - q1["win_20"]) * 100
                lines2.append(f"| {mv_lab} | {len(sub):,} | "
                              f"{q1['win_20']*100:.1f}% | {q5['win_20']*100:.1f}% | "
                              f"**{diff:+.2f}pp** | "
                              f"{q1['avg_20']:+.2f}% | {q5['avg_20']:+.2f}% |\n")
            except Exception:
                continue
        lines2.append("\n")

    out_file2 = OUT / "report_winrate_layered.md"
    out_file2.write_text("".join(lines2), encoding="utf-8")
    print(f"分层胜率: {out_file2}")

    # ── 4) PE 分层胜率 ──
    pe_focus = [
        "natr_14", "atr_pct", "sump_20", "boll_width", "lr_angle_20",
        "macd_hist", "ht_trendmode", "ma_ratio_60",
    ]
    lines3 = ["# 跨 PE 分层胜率 (重点反转因子)\n\n"]
    lines3.append(f"> 每因子 × 5 PE 段 × 5 分位桶, 看 Q5(高) vs Q1(低)\n\n")
    lines3.append(f"**基准 D+20 胜率**: {(df['r20']>0).mean()*100:.1f}%\n\n")

    for fc in pe_focus:
        if fc not in df.columns:
            continue
        lines3.append(f"## {fc}\n\n")
        lines3.append("| PE 段 | 样本 | Q1 D+20 胜率 | Q5 D+20 胜率 | Q5-Q1 差 | "
                      "Q1 D+20 涨幅 | Q5 D+20 涨幅 |\n")
        lines3.append("|---|---|---|---|---|---|---|\n")

        for pe_lab in PE_LABELS:
            sub = df[df["pe_bucket"] == pe_lab]
            if len(sub) < 5000:
                continue
            stats = factor_quantile_stats(sub, fc)
            if stats is None or stats.empty:
                continue
            try:
                q1 = stats.iloc[0]
                q5 = stats.iloc[-1]
                diff = (q5["win_20"] - q1["win_20"]) * 100
                lines3.append(f"| PE {pe_lab} | {len(sub):,} | "
                              f"{q1['win_20']*100:.1f}% | {q5['win_20']*100:.1f}% | "
                              f"**{diff:+.2f}pp** | "
                              f"{q1['avg_20']:+.2f}% | {q5['avg_20']:+.2f}% |\n")
            except Exception:
                continue
        lines3.append("\n")

    out_file3 = OUT / "report_winrate_by_pe.md"
    out_file3.write_text("".join(lines3), encoding="utf-8")
    print(f"PE 分层胜率: {out_file3}")

    # ── 5) 行业分层胜率 (Top 10 行业 × 各自最强因子) ──
    print("行业分层胜率...")
    ind_counts = df["industry"].value_counts()
    main_inds = ind_counts[ind_counts >= 30000].index.tolist()[:10]

    lines4 = ["# 行业内最强因子胜率\n\n"]
    lines4.append(f"> Top 10 行业 (样本≥30000) 内单因子 IC 最强者的 5 分位胜率\n\n")

    for ind in main_inds:
        sub = df[df["industry"] == ind]
        if len(sub) < 5000:
            continue
        # 找该行业内 |IC20| 最强 3 个因子
        scores = []
        for fc in factor_cols:
            ic = compute_ic20(sub, fc)
            if not np.isnan(ic):
                scores.append((fc, ic))
        scores.sort(key=lambda r: abs(r[1]), reverse=True)
        top3 = scores[:3]
        if not top3:
            continue

        base_win = (sub['r20'] > 0).mean() * 100
        lines4.append(f"## {ind} (样本 {len(sub):,}, 基准 D+20 胜率 {base_win:.1f}%)\n\n")

        for fc, ic in top3:
            stats = factor_quantile_stats(sub, fc)
            if stats is None or stats.empty:
                continue
            lines4.append(f"### {fc} (IC20 = {ic:+.4f})\n\n")
            lines4.append("| 桶 | 样本 | 因子区间 | D+20 涨幅 | D+20 胜率 | "
                          "D+40 涨幅 | D+40 胜率 |\n")
            lines4.append("|---|---|---|---|---|---|---|\n")
            for _, r in stats.iterrows():
                interval = f"[{r['factor_min']:.3f}, {r['factor_max']:.3f}]"
                lines4.append(f"| {r['bucket']} | {r['n']:,} | {interval} | "
                              f"{r['avg_20']:+.2f}% | {r['win_20']*100:.1f}% | "
                              f"{r['avg_40']:+.2f}% | {r['win_40']*100:.1f}% |\n")
            lines4.append("\n")

    out_file4 = OUT / "report_winrate_by_industry.md"
    out_file4.write_text("".join(lines4), encoding="utf-8")
    print(f"行业胜率: {out_file4}")


if __name__ == "__main__":
    main()
