#!/usr/bin/env python3
"""打分准确性诊断 — 验证: 高分 → 5/20 天稳定涨幅?

输出:
  1. 5 分位 × {r5, r20} 平均涨幅 / 中位涨幅 / 胜率
  2. 单调性: Q1 → Q5 是否单调递增
  3. 稳定性: 多少 % 的周 Q5 > Q1 (而不是平均上 Q5 > Q1)
  4. 不同分数段的最适买入证据
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.stockagent_analysis.sparse_layered_score import (
    compute_sparse_layered_score, bucket_mv, bucket_pe,
)
from backtest_18m import (
    FACTOR_DIR, FACTOR_COLS_EXCLUDE, MIN_STOCKS,
    mf_state_from_row, regime_from_score, get_weekly_dates,
)

START = "20250501"
END   = "20260126"
MATRIX_PATH = "output/factor_lab_oos/validity_matrix.json"
OUT_DIR = Path("output/backtest_oos_B_avg")
N_BUCKETS = 5
USE_EB = True
SCORE_MODE = "avg"   # 改用平均涨幅而非胜率

def load_data():
    parts = []
    for p in sorted(FACTOR_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        if not df.empty:
            parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    full["trade_date"] = full["trade_date"].astype(str)
    return full

def get_factor_cols(df):
    return [c for c in df.columns if c not in FACTOR_COLS_EXCLUDE and c not in ("r30","dd30")]

def score_week(df_week, factor_cols, matrix):
    scores = {}
    for _, row in df_week.iterrows():
        features = {}
        for fc in factor_cols:
            v = row.get(fc)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                features[fc] = float(v)
        mv = row.get("total_mv"); pe = row.get("pe_ttm") or row.get("pe")
        context = {
            "mv_seg": bucket_mv(float(mv) if mv else None),
            "pe_seg": bucket_pe(float(pe) if pe else None),
            "industry": str(row.get("industry", "") or ""),
            "etf_held": False,
        }
        mf = mf_state_from_row(dict(row))
        ms = row.get("market_score_adj")
        regime = regime_from_score(float(ms) if ms else None)
        result = compute_sparse_layered_score(
            features, context, matrix=matrix, regime=regime, mf_state=mf,
            use_eb=USE_EB, use_k_weight=False, score_mode=SCORE_MODE,
        )
        scores[row["ts_code"]] = result["layered_score"]
    return pd.Series(scores)

def main():
    print(f"加载 {START} → {END} ...")
    df = load_data()
    factor_cols = get_factor_cols(df)
    matrix = json.loads(Path(MATRIX_PATH).read_text(encoding="utf-8"))
    weekly = get_weekly_dates(df)
    print(f"周数={len(weekly)}, 因子={len(factor_cols)}")

    bucket_records = []
    for i, wdate in enumerate(weekly):
        df_week = df[df["trade_date"] == wdate].copy()
        df_week = df_week[df_week["r5"].notna() & df_week["r20"].notna()]
        if len(df_week) < MIN_STOCKS:
            continue
        scores = score_week(df_week, factor_cols, matrix)
        df_week["_score"] = df_week["ts_code"].map(scores)
        df_week = df_week.dropna(subset=["_score"])
        try:
            df_week["_bucket"] = pd.qcut(df_week["_score"], N_BUCKETS,
                                         labels=False, duplicates="drop") + 1
        except Exception:
            continue

        for b in range(1, N_BUCKETS + 1):
            sub = df_week[df_week["_bucket"] == b]
            if len(sub) == 0:
                continue
            r5 = sub["r5"].values; r20 = sub["r20"].values
            bucket_records.append({
                "date": wdate,
                "bucket": int(b),
                "n": int(len(sub)),
                "score_mean": float(sub["_score"].mean()),
                "score_min":  float(sub["_score"].min()),
                "score_max":  float(sub["_score"].max()),
                "r5_avg":     float(np.mean(r5)),
                "r5_median":  float(np.median(r5)),
                "r5_winrate": float((r5 > 0).mean()),
                "r20_avg":    float(np.mean(r20)),
                "r20_median": float(np.median(r20)),
                "r20_winrate":float((r20 > 0).mean()),
            })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(weekly)}] {wdate}")

    res = pd.DataFrame(bucket_records)
    OUT_DIR.mkdir(exist_ok=True)
    res.to_parquet(OUT_DIR / "bucket_returns.parquet", index=False)

    # ── 报告 ────────────────────────────────────────────────────────
    n_weeks = res["date"].nunique()
    print("\n" + "=" * 78)
    print(f"打分准确性诊断 (eb 版本, {n_weeks} 周, {START}→{END})")
    print("=" * 78)

    # 1. 5 分位均值
    print("\n## 1. 5 分位平均表现")
    summary = res.groupby("bucket").agg(
        score=("score_mean", "mean"),
        r5_avg=("r5_avg", "mean"),
        r5_med=("r5_median", "mean"),
        r5_win=("r5_winrate", lambda x: x.mean() * 100),
        r20_avg=("r20_avg", "mean"),
        r20_med=("r20_median", "mean"),
        r20_win=("r20_winrate", lambda x: x.mean() * 100),
        n=("n", "mean"),
    ).round(3).reset_index()
    summary.columns = ["分位", "平均分", "r5均涨%", "r5中位%", "r5胜率%",
                       "r20均涨%", "r20中位%", "r20胜率%", "周均N"]
    print(summary.to_string(index=False))

    # 2. 单调性: Q1→Q5 是否递增
    print("\n## 2. 单调性 (Q1→Q5 是否递增)")
    means_r5  = summary["r5均涨%"].values
    means_r20 = summary["r20均涨%"].values
    wins_r5   = summary["r5胜率%"].values
    wins_r20  = summary["r20胜率%"].values
    print(f"  r5  涨幅: {' → '.join(f'{v:+.3f}' for v in means_r5)}  "
          f"{'✓ 单调递增' if all(means_r5[i] <= means_r5[i+1] for i in range(4)) else '✗ 非单调'}")
    print(f"  r20 涨幅: {' → '.join(f'{v:+.3f}' for v in means_r20)}  "
          f"{'✓ 单调递增' if all(means_r20[i] <= means_r20[i+1] for i in range(4)) else '✗ 非单调'}")
    print(f"  r5  胜率: {' → '.join(f'{v:.1f}%' for v in wins_r5)}  "
          f"{'✓ 单调递增' if all(wins_r5[i] <= wins_r5[i+1] for i in range(4)) else '✗ 非单调'}")
    print(f"  r20 胜率: {' → '.join(f'{v:.1f}%' for v in wins_r20)}  "
          f"{'✓ 单调递增' if all(wins_r20[i] <= wins_r20[i+1] for i in range(4)) else '✗ 非单调'}")

    # 3. Q5-Q1 价差 + 稳定性
    print("\n## 3. 高低分价差 (Q5 - Q1) 与稳定性")
    pivot_r5  = res.pivot(index="date", columns="bucket", values="r5_avg")
    pivot_r20 = res.pivot(index="date", columns="bucket", values="r20_avg")
    spread_r5  = (pivot_r5[5]  - pivot_r5[1]).dropna()
    spread_r20 = (pivot_r20[5] - pivot_r20[1]).dropna()
    print(f"  r5  Q5-Q1 价差均值: {spread_r5.mean()*100:+.4f}%  "
          f"std={spread_r5.std()*100:.3f}%  "
          f"周次 Q5>Q1: {(spread_r5 > 0).mean()*100:.1f}%")
    print(f"  r20 Q5-Q1 价差均值: {spread_r20.mean()*100:+.4f}%  "
          f"std={spread_r20.std()*100:.3f}%  "
          f"周次 Q5>Q1: {(spread_r20 > 0).mean()*100:.1f}%")

    # 4. 各分位每周胜率方差 (稳定性)
    print("\n## 4. Q5 高分组每周胜率分布")
    q5 = res[res["bucket"] == 5]
    print(f"  r5  胜率: 均{q5['r5_winrate'].mean()*100:.1f}%  "
          f"std={q5['r5_winrate'].std()*100:.1f}%  "
          f"min={q5['r5_winrate'].min()*100:.1f}%  "
          f"max={q5['r5_winrate'].max()*100:.1f}%")
    print(f"  r20 胜率: 均{q5['r20_winrate'].mean()*100:.1f}%  "
          f"std={q5['r20_winrate'].std()*100:.1f}%  "
          f"min={q5['r20_winrate'].min()*100:.1f}%  "
          f"max={q5['r20_winrate'].max()*100:.1f}%")

    # 5. 高分段细分 (Q5 内部再分 5 段)
    print("\n## 5. Q5 高分段内部细分 (按分数再分 5 段)")
    q5_only = res[res["bucket"] == 5]
    if len(q5_only) > 0:
        # 在每周 Q5 内部, 用 score_max 范围切分
        # 由于已是 Q5, 直接用各周的高分子集再分位
        # 用全局 score 范围近似
        all_q5_records = []
        for wdate in q5_only["date"].unique():
            wk = df[df["trade_date"] == wdate].copy()
            wk = wk[wk["r5"].notna() & wk["r20"].notna()]
            if len(wk) < MIN_STOCKS:
                continue
            scores = score_week(wk, factor_cols, matrix)
            wk["_score"] = wk["ts_code"].map(scores)
            wk = wk.dropna(subset=["_score"])
            # 取 top 20% 再分 5 段
            top_thresh = wk["_score"].quantile(0.80)
            top = wk[wk["_score"] >= top_thresh].copy()
            if len(top) < 20: continue
            try:
                top["_sub"] = pd.qcut(top["_score"], 5, labels=False, duplicates="drop") + 1
            except Exception:
                continue
            for b in range(1, 6):
                sub = top[top["_sub"] == b]
                if len(sub) == 0: continue
                all_q5_records.append({
                    "sub": b,
                    "score_mean": float(sub["_score"].mean()),
                    "r5_avg":  float(sub["r5"].mean()),
                    "r20_avg": float(sub["r20"].mean()),
                    "r5_win":  float((sub["r5"]>0).mean()),
                    "r20_win": float((sub["r20"]>0).mean()),
                })
        if all_q5_records:
            q5d = pd.DataFrame(all_q5_records)
            sub_summary = q5d.groupby("sub").agg(
                score=("score_mean","mean"),
                r5_avg=("r5_avg","mean"),
                r20_avg=("r20_avg","mean"),
                r5_win=("r5_win", lambda x: x.mean()*100),
                r20_win=("r20_win", lambda x: x.mean()*100),
            ).round(3).reset_index()
            sub_summary.columns=["Q5内子段","平均分","r5均涨%","r20均涨%","r5胜率%","r20胜率%"]
            print(sub_summary.to_string(index=False))

    summary.to_csv(OUT_DIR / "bucket_summary.csv", index=False)
    print(f"\n详细数据: {OUT_DIR}/bucket_returns.parquet")

if __name__ == "__main__":
    main()
