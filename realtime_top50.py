#!/usr/bin/env python3
"""实战测试 — 用最新一日数据生成 top 50 候选清单 + 风险等级.

模拟实战流程:
  1. 取最新一日全市场所有股票
  2. 对每只跑 sparse + LGBM 完整流程
  3. 按 entry_score 排序
  4. 输出 top 50 清单 (entry / risk / sparse / 关键预测值)
  5. 看其中 LLM 4 专家 真正会买的有几只 (entry≥75 + risk≤50)

输出: output/realtime/top50_candidates.csv
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS = ROOT / "output" / "labels" / "max_gain_labels.parquet"
ETF    = ROOT / "output" / "etf_analysis" / "stock_to_etfs.json"
OUT_DIR = ROOT / "output" / "realtime"
OUT_DIR.mkdir(exist_ok=True)


def main(target_date: str = None):
    t0 = time.time()
    # 取最新一个交易日
    if target_date is None:
        # 找最新有数据的日期
        last_part = sorted(PARQUET_DIR.glob("*.parquet"))[-1]
        df = pd.read_parquet(last_part, columns=["trade_date"])
        target_date = df["trade_date"].astype(str).max()
    print(f"目标日期: {target_date}")

    # 加载所有股票该日数据
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[df["trade_date"] == target_date]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    print(f"该日全市场: {len(full)} 只股票")

    # merge 所有附加特征
    for path, key, name in [
        (ROOT / "output" / "regimes" / "daily_regime.parquet", "trade_date", "regime"),
        (ROOT / "output" / "amount_features" / "amount_features.parquet",
         ["ts_code","trade_date"], "amount"),
        (ROOT / "output" / "regime_extra" / "regime_extra.parquet", "trade_date", "rextra"),
    ]:
        if path.exists():
            d = pd.read_parquet(path)
            if "trade_date" in d.columns:
                d["trade_date"] = d["trade_date"].astype(str)
            if name == "regime":
                d = d.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                        "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14",
                                        "vol_ratio":"mkt_vol_ratio"})
            full = full.merge(d, on=key, how="left")

    # 真实 max_gain (可选, 用于事后验证)
    if LABELS.exists():
        labels = pd.read_parquet(LABELS, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
        labels["trade_date"] = labels["trade_date"].astype(str)
        full = full.merge(labels, on=["ts_code","trade_date"], how="left")

    # 加载 4 个 LGBM 模型
    models = {}
    for name, dir_path in [
        ("uptrend", "lgbm_uptrend"),
        ("risk",    "lgbm_risk"),
        ("maxgain_g", "lgbm_maxgain"),
    ]:
        booster = lgb.Booster(model_file=f"output/{dir_path}/classifier.txt"
                                if name in ("uptrend","risk")
                                else f"output/{dir_path}/regressor_gain.txt")
        meta = json.loads(Path(f"output/{dir_path}/feature_meta.json").read_text(encoding="utf-8"))
        models[name] = (booster, meta)

    # 加载 max_dd
    booster_dd = lgb.Booster(model_file="output/lgbm_maxgain/regressor_dd.txt")

    etf_holders = set(json.loads(Path(ETF).read_text(encoding="utf-8")).keys())
    full["etf_held"] = full["ts_code"].isin(etf_holders)

    # 批量预测每个模型
    print("批量预测中...")
    for name, (booster, meta) in models.items():
        feat_cols = meta["feature_cols"]
        indmap = meta["industry_map"]
        full[f"{name}_iid"] = full["industry"].fillna("unknown").map(
            lambda x: indmap.get(str(x), -1))
        # 临时 frame
        # 创建/复用 industry_id
        if "industry_id" not in full.columns or full[f"{name}_iid"].iloc[0] != full.get("industry_id", pd.Series([None])).iloc[0]:
            full["industry_id"] = full[f"{name}_iid"]
        for c in feat_cols:
            if c not in full.columns: full[c] = np.nan
        X = full[feat_cols].astype(float)
        full[f"{name}_pred"] = booster.predict(X)

    # max_dd 用同 maxgain_g 的特征
    feat_cols_g = models["maxgain_g"][1]["feature_cols"]
    X = full[feat_cols_g].astype(float)
    full["maxdd_pred"] = booster_dd.predict(X)

    # MV 分桶
    full["mv_seg"] = full["total_mv"].apply(lambda v:
        ("20-50亿" if v/1e4 < 50 else "50-100亿" if v/1e4 < 100 else
         "100-300亿" if v/1e4 < 300 else "300-1000亿" if v/1e4 < 1000 else "1000亿+")
        if pd.notna(v) else None)

    # 第一步: 用 uptrend_pred 取 top 200 候选 (粗筛)
    full = full.sort_values("uptrend_pred", ascending=False)
    candidates = full.head(200).copy()
    print(f"\n第一步: uptrend_prob 粗筛 top 200, 起涨概率均值 {candidates['uptrend_pred'].mean()*100:.2f}%")

    # 第二步: 对 top 200 跑完整 sparse_layered_score (含黑名单/域限/regime)
    print("第二步: 综合评分排序 (含黑名单 + 域限 + sparse 因子)...")
    sys.path.insert(0, str(ROOT))
    from src.stockagent_analysis.sparse_layered_score import (
        compute_sparse_layered_score, bucket_mv, bucket_pe,
    )
    matrix_path = ROOT / "output" / "factor_lab_oos" / "validity_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    factor_excludes = {"ts_code","trade_date","industry","name",
                        "r5","r10","r20","r30","r40","dd5","dd10","dd20","dd30","dd40",
                        "mv_bucket","pe_bucket","total_mv","pe","pe_ttm",
                        "market_score_adj","mf_divergence","mf_strength","mf_consecutive"}
    factor_cols = [c for c in candidates.columns
                    if c not in factor_excludes
                    and pd.api.types.is_numeric_dtype(candidates[c])]
    entry_scores = []; risk_scores = []
    for _, row in candidates.iterrows():
        feats = {fc: float(row[fc]) for fc in factor_cols if pd.notna(row.get(fc))}
        raw = {k: row.get(k) for k in ("total_mv","pe","pe_ttm","market_score_adj",
                                        "mf_divergence","mf_strength","mf_consecutive")}
        ts = str(row["ts_code"])
        ctx = {"mv_seg": bucket_mv(row.get("total_mv")),
                "pe_seg": bucket_pe(row.get("pe_ttm") or row.get("pe")),
                "industry": str(row.get("industry") or ""),
                "etf_held": ts in etf_holders,
                "_raw": raw}
        try:
            r = compute_sparse_layered_score(feats, ctx, matrix=matrix,
                                              use_eb=True, use_k_weight=False)
            es = (r.get("entry_score") or {}).get("score", 50)
            rs = (r.get("risk_score") or {}).get("score", 50)
        except Exception:
            es, rs = 50, 50
        entry_scores.append(es); risk_scores.append(rs)
    candidates["entry_score"] = entry_scores
    candidates["risk_score"] = risk_scores

    # 第三步: 用 entry_score 重排取 top 50
    candidates = candidates.sort_values("entry_score", ascending=False)
    top50 = candidates.head(50).copy()
    print(f"第三步: entry_score 重排, top 50 entry 均值 {top50['entry_score'].mean():.1f} risk 均值 {top50['risk_score'].mean():.1f}")

    # 提取关键列
    out = top50[[
        "ts_code", "industry", "mv_seg", "etf_held",
        "entry_score", "risk_score",
        "uptrend_pred", "risk_pred",
        "maxgain_g_pred", "maxdd_pred",
    ]].rename(columns={
        "entry_score": "买入评分",
        "risk_score": "风险评分",
        "uptrend_pred": "起涨概率",
        "risk_pred": "回撤风险",
        "maxgain_g_pred": "预测max_gain%",
        "maxdd_pred": "预测max_dd%",
    })
    # 加真实结果如果有
    if "max_gain_20" in top50.columns:
        out["真实max_gain%"] = top50["max_gain_20"]
        out["真实max_dd%"] = top50["max_dd_20"]

    # 加 entry/risk 评分概念 (简化版: 基于 uptrend + risk)
    out["综合建议"] = out.apply(lambda r:
        "强买" if r["起涨概率"] >= 0.30 and r["回撤风险"] < 0.50 else
        "中买" if r["起涨概率"] >= 0.15 and r["回撤风险"] < 0.60 else
        "试探" if r["起涨概率"] >= 0.08 else
        "回避", axis=1)

    out["起涨概率"] = (out["起涨概率"] * 100).round(2)
    out["回撤风险"] = (out["回撤风险"] * 100).round(2)
    out["预测max_gain%"] = out["预测max_gain%"].round(2)
    out["预测max_dd%"] = out["预测max_dd%"].round(2)

    print(f"\n=== {target_date} Top 50 候选 ===")
    print(out.to_string(index=False, max_colwidth=15))

    # 统计
    print(f"\n=== 综合建议分布 ===")
    print(out["综合建议"].value_counts().to_string())

    if "真实max_gain%" in out.columns:
        avg_real_gain = out["真实max_gain%"].mean()
        gain_15_rate = (out["真实max_gain%"] >= 15).mean() * 100
        gain_20_rate = (out["真实max_gain%"] >= 20).mean() * 100
        avg_real_dd = out["真实max_dd%"].mean()
        dd_lt_8_rate = (out["真实max_dd%"] < -8).mean() * 100
        print(f"\n=== 真实结果验证 (50 只) ===")
        print(f"平均真实 max_gain: {avg_real_gain:.2f}%")
        print(f"涨过 +15%: {gain_15_rate:.1f}%")
        print(f"涨过 +20%: {gain_20_rate:.1f}%")
        print(f"平均真实 max_dd:  {avg_real_dd:.2f}%")
        print(f"破 -8% 比例:    {dd_lt_8_rate:.1f}%")

    # 强买子集
    strong = out[out["综合建议"] == "强买"]
    print(f"\n=== 强买子集 ({len(strong)} 只) ===")
    if len(strong) > 0 and "真实max_gain%" in strong.columns:
        print(f"平均真实 max_gain: {strong['真实max_gain%'].mean():.2f}%")
        print(f"涨过 +20%: {(strong['真实max_gain%']>=20).mean()*100:.1f}%")
        print(f"破 -8% 比例:   {(strong['真实max_dd%']<-8).mean()*100:.1f}%")

    out.to_csv(OUT_DIR / f"top50_{target_date}.csv", index=False, encoding="utf-8-sig")
    print(f"\n输出: {OUT_DIR / f'top50_{target_date}.csv'}")
    print(f"耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    main(target)
