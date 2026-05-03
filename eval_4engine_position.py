#!/usr/bin/env python3
"""完整评估四引擎系统在 OOS 上的仓位精度.

输入: parquet (OOS 期 2025-05 → 2026-01) + max_gain_labels
输出: 各仓位档真实 max_gain / max_dd / gain≥15/20/30 命中率
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
MATRIX = "output/factor_lab_oos/validity_matrix.json"
LABELS = "output/labels/max_gain_labels.parquet"
ETF    = "output/etf_analysis/stock_to_etfs.json"
OUT_DIR = Path("output/eval_4engine")
OUT_DIR.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    print(f"加载 OOS 数据 {START} → {END} ...")
    parts = []
    for p in sorted(FACTOR_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    # 抽样: 每月前 5 天 (与之前 diagnose 一致)
    full["_ym"] = full["trade_date"].str[:6]
    sampled = full.groupby("_ym")["trade_date"].apply(
        lambda x: sorted(x.unique())[:5]
    ).explode().tolist()
    full = full[full["trade_date"].isin(sampled)]
    print(f"抽样后: {len(full)} 行")

    # 合并 max_gain 标签
    labels = pd.read_parquet(LABELS,
                              columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left")
    full = full.dropna(subset=["max_gain_20"])
    print(f"有标签样本: {len(full)}")

    factor_cols = [c for c in full.columns if c not in FACTOR_COLS_EXCLUDE
                   and c not in ("r30","dd30","entry_open","r20_close","is_clean")]

    matrix = json.loads(Path(MATRIX).read_text(encoding="utf-8"))
    etf_holders = set(json.loads(Path(ETF).read_text(encoding="utf-8")).keys())

    print("打分...")
    records = []
    for i, (_, row) in enumerate(full.iterrows()):
        if (i+1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(full)}] {int(elapsed)}s")
        feats = {fc: float(row[fc]) for fc in factor_cols if pd.notna(row.get(fc))}
        raw = {k: row.get(k) for k in ("total_mv","pe","pe_ttm","market_score_adj",
                                        "mf_divergence","mf_strength","mf_consecutive")}
        ts_code = str(row["ts_code"])
        ctx = {"mv_seg": bucket_mv(row.get("total_mv")),
               "pe_seg": bucket_pe(row.get("pe_ttm") or row.get("pe")),
               "industry": str(row.get("industry") or ""),
               "etf_held": ts_code in etf_holders,
               "_raw": raw}
        try:
            r = compute_sparse_layered_score(feats, ctx, matrix=matrix,
                                              use_eb=True, use_k_weight=False)
        except Exception:
            continue
        pos = r.get("position_suggestion") or {}
        mg = r.get("maxgain") or {}
        records.append({
            "ts_code": ts_code, "date": row["trade_date"],
            "mv_seg": ctx["mv_seg"], "etf_held": ctx["etf_held"],
            "sparse": r["layered_score"],
            "pred_gain": mg.get("pred_gain"),
            "pred_dd": mg.get("pred_dd"),
            "position_pct": pos.get("position_pct"),
            "tier": pos.get("tier"),
            "true_gain": row["max_gain_20"],
            "true_dd": row["max_dd_20"],
        })

    res = pd.DataFrame(records)
    res.to_parquet(OUT_DIR / "scored.parquet", index=False)
    print(f"\n打分完成: {len(res)} 行 ({time.time()-t0:.1f}s)")

    # === 仓位档 vs 真实表现 ===
    print("\n" + "=" * 78)
    print("【新四引擎】 仓位档 vs 真实 max_gain 表现")
    print("=" * 78)
    res["pos_bucket"] = pd.cut(res["position_pct"].fillna(0),
                                bins=[-1, 5, 15, 30, 50, 70, 100],
                                labels=["0-5%","5-15%","15-30%","30-50%","50-70%","70-100%"])
    summary = res.groupby("pos_bucket", observed=True).agg(
        n=("true_gain","count"),
        pred_gain_avg=("pred_gain","mean"),
        true_gain_avg=("true_gain","mean"),
        true_gain_med=("true_gain","median"),
        gain_15=("true_gain", lambda x: (x>=15).mean()*100),
        gain_20=("true_gain", lambda x: (x>=20).mean()*100),
        gain_30=("true_gain", lambda x: (x>=30).mean()*100),
        dd_avg=("true_dd","mean"),
    ).round(2)
    print(summary.to_string())
    summary.to_csv(OUT_DIR / "pos_bucket.csv", encoding="utf-8-sig")

    # === pred_gain 和 true_gain 的相关性 ===
    print("\n=== pred_gain vs true_gain 相关性 ===")
    from scipy import stats as ss
    cln = res.dropna(subset=["pred_gain","true_gain"])
    spearman = ss.spearmanr(cln["pred_gain"], cln["true_gain"])[0]
    pearson  = cln[["pred_gain","true_gain"]].corr().iloc[0,1]
    print(f"Spearman IC: {spearman:.4f}  Pearson: {pearson:.4f}  N={len(cln)}")

    # === 高仓位组按 mv×ETF 切片 ===
    print("\n=== 强买档 (>=70%) MV × ETF ===")
    strong = res[res["position_pct"] >= 70]
    if len(strong) > 0:
        cuts = strong.groupby(["mv_seg","etf_held"], observed=True).agg(
            n=("true_gain","count"),
            true_gain=("true_gain","mean"),
            gain_15=("true_gain", lambda x: (x>=15).mean()*100),
            gain_20=("true_gain", lambda x: (x>=20).mean()*100),
            gain_30=("true_gain", lambda x: (x>=30).mean()*100),
        ).round(2)
        print(cuts.to_string())
        cuts.to_csv(OUT_DIR / "strong_mv_etf.csv", encoding="utf-8-sig")
    else:
        print("(没有 >=70% 仓位的样本)")


if __name__ == "__main__":
    main()
