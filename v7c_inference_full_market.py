#!/usr/bin/env python3
"""V7c 全市场推理 (用 2026-04-20 最新数据).

合并 factor_lab + extension + 各 features → V7c 4 模型推理 → 推荐池
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
TARGET_DATE = "20260420"

R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)


def _map_anchored(v, p5, p50, p95):
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, 50.0)
    out = np.where(v <= p5, 0, out)
    out = np.where(v >= p95, 100, out)
    mask_lo = (v > p5) & (v <= p50)
    out = np.where(mask_lo, (v - p5) / (p50 - p5) * 50, out)
    mask_hi = (v > p50) & (v < p95)
    out = np.where(mask_hi, 50 + (v - p50) / (p95 - p50) * 50, out)
    return out


def predict_model(df, name):
    d = PROD_DIR / name
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta.get("industry_map", {})
    df = df.copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), -1)
    )
    for fc in feat_cols:
        if fc not in df.columns:
            df[fc] = np.nan
    return booster.predict(df[feat_cols])


def main():
    t0 = time.time()
    print(f"=== V7c 全市场推理 (目标日期 {TARGET_DATE}) ===\n")

    # 1. 加载 factor_lab extension (5072 股 × 60 天 × 153 因子)
    EXT_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups_extension"
    fl_parts = []
    for p in sorted(EXT_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[df["trade_date"] == TARGET_DATE]  # 仅 2026-04-20
        if not df.empty: fl_parts.append(df)
    df_fl = pd.concat(fl_parts, ignore_index=True)
    print(f"factor_lab 全市场 4-20: {len(df_fl):,} 股", flush=True)

    # 2. merge 各 features (按 ts_code + trade_date)
    for path, name in [
        (ROOT / "output" / "amount_features" / "amount_features.parquet", "amount"),
        (ROOT / "output" / "regime_extra" / "regime_extra.parquet", "regime_extra"),
        (ROOT / "output" / "moneyflow" / "features.parquet", "moneyflow_v1"),
        (ROOT / "output" / "cogalpha_features" / "features.parquet", "cogalpha"),
        (ROOT / "output" / "mfk_features" / "features.parquet", "mfk"),
        (ROOT / "output" / "pyramid_v2" / "features.parquet", "pyramid"),
        (ROOT / "output" / "v7_extras" / "features.parquet", "v7_extras"),
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        d["trade_date"] = d["trade_date"].astype(str)
        if "ts_code" in d.columns:
            d_target = d[d["trade_date"] == TARGET_DATE]
            if d_target.empty:  # cogalpha 末日 03-19, 取最近
                d_target = d.sort_values(["ts_code","trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
                d_target = d_target.drop(columns=["trade_date"])  # 不带日期免冲突
                df_fl = df_fl.merge(d_target, on="ts_code", how="left", suffixes=("","_x"))
            else:
                df_fl = df_fl.merge(d_target.drop(columns=["trade_date"]), on="ts_code", how="left", suffixes=("","_x"))
            print(f"  merge {name:12s}: {len(d_target):,} 股, {len(df_fl)} 行", flush=True)
        else:
            d_target = d[d["trade_date"] == TARGET_DATE]
            if d_target.empty:
                d_target = d.sort_values("trade_date").tail(1).reset_index(drop=True)
            for col in d_target.columns:
                if col == "trade_date": continue
                if col not in df_fl.columns:
                    df_fl[col] = d_target[col].iloc[0]

    # regime_id 等市场级
    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path)
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg_t = rg[rg["trade_date"] == TARGET_DATE]
        if rg_t.empty:
            rg_t = rg.sort_values("trade_date").tail(1).reset_index(drop=True)
        rg_t = rg_t.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                       "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        for col in ["regime_id","mkt_ret_5d","mkt_ret_20d","mkt_ret_60d","mkt_rsi14","mkt_vol_ratio"]:
            if col not in df_fl.columns and col in rg_t.columns:
                df_fl[col] = rg_t[col].iloc[0]

    print(f"\n特征拼接完成: {len(df_fl):,} 股 × {len(df_fl.columns)} 列\n", flush=True)

    # 3. V7c 4 模型推理
    print("V7c 推理中...", flush=True)
    df_fl["r10_pred"] = predict_model(df_fl, "r10_v4_all")
    df_fl["r20_pred"] = predict_model(df_fl, "r20_v4_all")
    df_fl["sell_10_v6_prob"] = predict_model(df_fl, "sell_10_v6")
    df_fl["sell_20_v6_prob"] = predict_model(df_fl, "sell_20_v6")
    s10 = _map_anchored(df_fl["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df_fl["r20_pred"].values, *R20_ANCHOR)
    df_fl["buy_score"] = 0.5 * s10 + 0.5 * s20
    s10s = _map_anchored(df_fl["sell_10_v6_prob"].values, *SELL10_V6)
    s20s = _map_anchored(df_fl["sell_20_v6_prob"].values, *SELL20_V6)
    df_fl["sell_score"] = 0.5 * s10s + 0.5 * s20s

    # 4. V7c 推荐池筛选 (用 V7c 完整规则)
    if "pyr_velocity_20_60" in df_fl.columns:
        p35 = df_fl["pyr_velocity_20_60"].quantile(0.35)
    else:
        p35 = -0.1  # fallback

    df_fl["quadrant"] = "中性区"
    df_fl.loc[(df_fl["buy_score"] >= 70) & (df_fl["sell_score"] <= 30), "quadrant"] = "理想多"
    df_fl.loc[(df_fl["buy_score"] >= 70) & (df_fl["sell_score"] >= 70), "quadrant"] = "矛盾段"
    df_fl.loc[(df_fl["buy_score"] <= 30) & (df_fl["sell_score"] >= 70), "quadrant"] = "主流空"
    df_fl.loc[(df_fl["buy_score"] <= 30) & (df_fl["sell_score"] <= 30), "quadrant"] = "沉寂"

    # V7c 推荐池 (含 V7 + V6.5 完整过滤)
    if "f1_neg1" in df_fl.columns and "f2_pos1" in df_fl.columns:
        df_fl["v7c_recommend"] = ((df_fl["buy_score"] >= 70) & (df_fl["buy_score"] <= 85) &
                                     (df_fl["sell_score"] <= 30) &
                                     (df_fl["pyr_velocity_20_60"] < p35) &
                                     (df_fl["f1_neg1"].abs() < 0.005) &
                                     (df_fl["f2_pos1"].abs() < 0.005))
    else:
        df_fl["v7c_recommend"] = (df_fl["buy_score"] >= 70) & (df_fl["sell_score"] <= 30)

    # 5. 输出
    OUT_DIR = ROOT / "output" / "v7c_full_inference"
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"v7c_inference_{TARGET_DATE}.csv"
    keep = ["ts_code","trade_date","industry","total_mv","pe","pe_ttm",
             "buy_score","sell_score","quadrant","v7c_recommend",
             "r10_pred","r20_pred","sell_10_v6_prob","sell_20_v6_prob",
             "mfk_pyramid_top_heavy","pyr_velocity_20_60",
             "f1_neg1","f2_pos1","main_net_5d"]
    keep = [c for c in keep if c in df_fl.columns]
    df_fl[keep].to_csv(out_path, index=False, encoding="utf-8-sig")

    # 统计
    print(f"\n=== 4 象限分布 ===")
    print(df_fl["quadrant"].value_counts().to_string())

    print(f"\n=== V7c 推荐池 (今日 {TARGET_DATE}) ===")
    rec = df_fl[df_fl["v7c_recommend"]]
    print(f"  推荐数: {len(rec):,} 股 (占全市场 {len(rec)/len(df_fl)*100:.1f}%)")
    print(f"  buy_score 中位: {rec['buy_score'].median():.0f}, 平均: {rec['buy_score'].mean():.0f}")
    print(f"  r20 预测中位: {rec['r20_pred'].median():+.2f}%")

    if len(rec) > 0:
        print(f"\n=== Top 30 推荐 (按 r20 预测降序) ===")
        top30 = rec.nlargest(30, "r20_pred")[
            ["ts_code","industry","buy_score","sell_score","r20_pred","sell_20_v6_prob"]
        ]
        print(top30.to_string(index=False))

    print(f"\n输出: {out_path}")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
