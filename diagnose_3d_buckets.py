#!/usr/bin/env python3
"""三维分桶诊断 (MV × PE × ETF) — 验证三引擎在不同股票类型上的精度差异.

输入: 2025-05 → 2026-01 OOS 测试集
对每行运行 sparse + LGBM r20 + clean 检测器, 按 (mv_seg, pe_seg, etf_held) 聚合:
  - 平均预测值
  - 真实 r20 平均/胜率
  - 关键指标: 高分组 (top 20%) 真实表现 vs 该桶基线

输出: output/diagnose_3d/bucket_results.csv
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.stockagent_analysis.sparse_layered_score import (
    compute_sparse_layered_score, bucket_mv, bucket_pe,
)
from backtest_18m import FACTOR_DIR, FACTOR_COLS_EXCLUDE

START = "20250501"
END   = "20260126"
MATRIX_PATH = "output/factor_lab_oos/validity_matrix.json"
ETF_PATH = "output/etf_analysis/stock_to_etfs.json"
OUT_DIR = Path("output/diagnose_3d")
OUT_DIR.mkdir(exist_ok=True)

# 抽样: 每周第一个交易日, 加快速度 (~16k 样本/月)
SAMPLE_DAYS = None   # None = 全量, 设数字=每隔 N 天采一次

def load_data():
    parts = []
    for p in sorted(FACTOR_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        if not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True)

def main():
    t0 = time.time()
    print(f"加载 OOS 数据 {START} → {END} ...")
    df = load_data()
    df = df.dropna(subset=["r20"])
    print(f"总行数: {len(df)}")

    factor_cols = [c for c in df.columns if c not in FACTOR_COLS_EXCLUDE
                   and c not in ("r30","dd30")]

    # 抽样
    all_dates = sorted(df["trade_date"].unique())
    if SAMPLE_DAYS:
        sampled = all_dates[::SAMPLE_DAYS]
        df = df[df["trade_date"].isin(sampled)]
        print(f"抽样后: {len(df)} 行 ({len(sampled)} 天)")
    else:
        # 默认每月第一周采样, 控制规模
        df["_ym"] = df["trade_date"].str[:6]
        sampled_dates = df.groupby("_ym")["trade_date"].apply(
            lambda x: sorted(x.unique())[:5]
        ).explode().tolist()
        df = df[df["trade_date"].isin(sampled_dates)]
        print(f"抽样后: {len(df)} 行 (每月前 5 天)")

    # ETF 持仓
    try:
        etf_data = json.loads(Path(ETF_PATH).read_text(encoding="utf-8"))
        etf_holders = set(etf_data.keys())
        print(f"ETF 持仓集合: {len(etf_holders)} 股")
    except Exception:
        etf_holders = set()

    matrix = json.loads(Path(MATRIX_PATH).read_text(encoding="utf-8"))

    print("\n开始打分...")
    records = []
    n_total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_total}] 已耗时 {int(elapsed)}s, "
                  f"预计剩 {int(elapsed/(i+1)*(n_total-i-1))}s")

        feats = {fc: float(row[fc]) for fc in factor_cols if pd.notna(row.get(fc))}
        raw = {k: row.get(k) for k in
               ("total_mv","pe","pe_ttm","market_score_adj",
                "mf_divergence","mf_strength","mf_consecutive")}
        ts_code = str(row["ts_code"])
        etf_held = ts_code in etf_holders
        ctx = {
            "mv_seg":   bucket_mv(row.get("total_mv")),
            "pe_seg":   bucket_pe(row.get("pe_ttm") or row.get("pe")),
            "industry": str(row.get("industry") or ""),
            "etf_held": etf_held,
            "_raw":     raw,
        }
        try:
            r = compute_sparse_layered_score(
                feats, ctx, matrix=matrix,
                use_eb=True, use_k_weight=False, score_mode="win",
            )
        except Exception:
            continue
        lgbm = r.get("lgbm") or {}
        clean = r.get("clean") or {}
        pos = r.get("position_suggestion") or {}
        records.append({
            "ts_code": ts_code,
            "trade_date": row["trade_date"],
            "mv_seg": ctx["mv_seg"],
            "pe_seg": ctx["pe_seg"],
            "etf_held": etf_held,
            "industry": ctx["industry"],
            "sparse": r["layered_score"],
            "lgbm_pred_r20": lgbm.get("pred_r20"),
            "lgbm_winprob":  lgbm.get("winprob"),
            "clean_prob":    clean.get("clean_prob"),
            "position_pct":  pos.get("position_pct"),
            "consistency":   pos.get("consistency"),
            "real_r20":   row["r20"],
            "real_dd20":  row.get("dd20"),
        })

    res = pd.DataFrame(records)
    res.to_parquet(OUT_DIR / "scored_samples.parquet", index=False)
    print(f"\n打分完成: {len(res)} 行 ({time.time()-t0:.1f}s)")

    # ── 三维分桶分析 ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("三维分桶 (MV × PE × ETF) 表现")
    print("=" * 80)

    # 整体相关性
    for col in ["sparse", "lgbm_pred_r20", "clean_prob"]:
        if col in res.columns:
            sub = res.dropna(subset=[col, "real_r20"])
            from scipy import stats as ss
            ic = ss.spearmanr(sub[col], sub["real_r20"])[0]
            print(f"\n[整体] {col} vs r20 Spearman IC: {ic:+.4f} (N={len(sub)})")

    # 三维 group: 取 top 20% 高分股, 看真实 r20 + clean_prob 命中率
    print("\n## 按 mv_seg × etf_held 分桶 (高分组 top 20%)")
    bucket_summary = []
    for mv in res["mv_seg"].dropna().unique():
        for etf in [True, False]:
            sub = res[(res["mv_seg"] == mv) & (res["etf_held"] == etf)]
            if len(sub) < 100: continue
            # 按 sparse top 20%
            top_sparse = sub[sub["sparse"] >= sub["sparse"].quantile(0.80)]
            top_clean  = sub[sub["clean_prob"] >= sub["clean_prob"].quantile(0.80)] if "clean_prob" in sub.columns else sub.head(0)
            base_r20 = sub["real_r20"].mean()
            base_winrate = (sub["real_r20"] > 0).mean() * 100
            row_out = {
                "mv": mv, "etf": etf, "n": len(sub),
                "base_r20":  round(base_r20, 3),
                "base_win%": round(base_winrate, 1),
                "top_sparse_r20": round(top_sparse["real_r20"].mean(), 3) if len(top_sparse) else None,
                "top_sparse_win%": round((top_sparse["real_r20"] > 0).mean() * 100, 1) if len(top_sparse) else None,
                "top_clean_r20": round(top_clean["real_r20"].mean(), 3) if len(top_clean) else None,
                "top_clean_win%": round((top_clean["real_r20"] > 0).mean() * 100, 1) if len(top_clean) else None,
            }
            bucket_summary.append(row_out)
    summary_df = pd.DataFrame(bucket_summary).sort_values(["mv", "etf"])
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUT_DIR / "mv_etf_summary.csv", index=False, encoding="utf-8-sig")

    # MV × PE 二维
    print("\n## 按 mv_seg × pe_seg 分桶 (高 sparse 组超额收益)")
    pivot = res.dropna(subset=["mv_seg","pe_seg","sparse","real_r20"]).copy()
    pivot["is_top"] = pivot.groupby(["mv_seg","pe_seg"])["sparse"].transform(
        lambda x: x >= x.quantile(0.80)
    )
    grp = pivot.groupby(["mv_seg","pe_seg"]).apply(
        lambda g: pd.Series({
            "n": len(g),
            "base_r20": g["real_r20"].mean(),
            "top_r20":  g.loc[g["is_top"], "real_r20"].mean() if g["is_top"].any() else np.nan,
            "alpha_r20": g.loc[g["is_top"], "real_r20"].mean() - g["real_r20"].mean()
                        if g["is_top"].any() else np.nan,
        })
    ).reset_index()
    print(grp.to_string(index=False))
    grp.to_csv(OUT_DIR / "mv_pe_summary.csv", index=False, encoding="utf-8-sig")

    # 仓位建议精度: 看每个仓位档实际表现
    print("\n## 仓位建议精度")
    res["pos_bucket"] = pd.cut(res["position_pct"].fillna(0),
                                bins=[-1, 5, 15, 30, 60, 100],
                                labels=["0-5%", "5-15%", "15-30%", "30-60%", "60-100%"])
    pos_perf = res.groupby("pos_bucket", observed=True).agg(
        n=("real_r20", "count"),
        avg_r20=("real_r20", "mean"),
        winrate=("real_r20", lambda x: (x > 0).mean() * 100),
        winrate_5pct=("real_r20", lambda x: (x > 5).mean() * 100),
    ).round(3)
    print(pos_perf.to_string())
    pos_perf.to_csv(OUT_DIR / "position_perf.csv", encoding="utf-8-sig")

    print(f"\n总耗时: {time.time()-t0:.1f}s")
    print(f"输出: {OUT_DIR}/")

if __name__ == "__main__":
    main()
