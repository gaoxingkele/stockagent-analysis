#!/usr/bin/env python3
"""V7c v2 全市场推理 2026-05-07.

改动 vs v7c_inference_0507.py:
  - 模型改为 r10_v7c_v2 / r20_v7c_v2 / sell_10_v7c_v2 / sell_20_v7c_v2
  - 锚点改为 v2 模型训练 OOS 锚点 (从 meta.json 读)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
TARGET_DATE = "20260507"

# 从 meta.json 读取 v2 锚点
def _load_anchor(name):
    m = json.loads((PROD_DIR / name / "meta.json").read_text(encoding="utf-8"))
    return (m["anchor_p5"], m["anchor_p50"], m["anchor_p95"])


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


def load_target_slice(parquet_paths, ts_keyed=True):
    parts = []
    for p in parquet_paths:
        if not Path(p).exists(): continue
        d = pd.read_parquet(p)
        d["trade_date"] = d["trade_date"].astype(str)
        d_t = d[d["trade_date"] == TARGET_DATE]
        if not d_t.empty: parts.append(d_t)
    if not parts: return None
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["ts_code","trade_date"] if ts_keyed else ["trade_date"], keep="last"
    )


def main():
    t0 = time.time()
    print(f"=== V7c v2 全市场推理 (目标 {TARGET_DATE}) ===\n", flush=True)

    R10_ANCHOR = _load_anchor("r10_v7c_v2")
    R20_ANCHOR = _load_anchor("r20_v7c_v2")
    SELL10    = _load_anchor("sell_10_v7c_v2")
    SELL20    = _load_anchor("sell_20_v7c_v2")
    print(f"锚点 v2:")
    print(f"  r10: {R10_ANCHOR}")
    print(f"  r20: {R20_ANCHOR}")
    print(f"  sell_10: {SELL10}")
    print(f"  sell_20: {SELL20}\n")

    # 1. factor_lab: ext + ext2 拼起来取 05-07 截面
    EXT_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups_extension"
    fl_parts = []
    for p in sorted(EXT_DIR.glob("*.parquet")):
        d = pd.read_parquet(p)
        d["trade_date"] = d["trade_date"].astype(str)
        d_t = d[d["trade_date"] == TARGET_DATE]
        if not d_t.empty: fl_parts.append(d_t)
    df_fl = pd.concat(fl_parts, ignore_index=True).drop_duplicates(
        subset=["ts_code","trade_date"], keep="last"
    )
    print(f"factor_lab 05-07: {len(df_fl):,} 股", flush=True)

    # 2. merge 各 features (主 + ext_0507)
    FEATURES = [
        ("amount_features",  ["output/amount_features/amount_features.parquet",
                              "output/amount_features/amount_features_ext_0507.parquet"]),
        ("moneyflow_v1",     ["output/moneyflow/features.parquet",
                              "output/moneyflow/features_ext_0507.parquet"]),
        ("mfk",              ["output/mfk_features/features.parquet",
                              "output/mfk_features/features_ext_0507.parquet"]),
        ("pyramid",          ["output/pyramid_v2/features.parquet",
                              "output/pyramid_v2/features_ext_0507.parquet"]),
        ("v7_extras",        ["output/v7_extras/features.parquet",
                              "output/v7_extras/features_ext_0507.parquet"]),
        ("cogalpha",         ["output/cogalpha_features/features.parquet"]),
    ]
    for name, paths in FEATURES:
        d_t = load_target_slice(paths)
        if d_t is None or d_t.empty:
            for p in paths:
                if not Path(p).exists(): continue
                d = pd.read_parquet(p)
                d["trade_date"] = d["trade_date"].astype(str)
                d_t = d.sort_values(["ts_code","trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
                d_t = d_t.drop(columns=["trade_date"])
                df_fl = df_fl.merge(d_t, on="ts_code", how="left", suffixes=("","_x"))
                print(f"  merge {name:14s} (fallback): {len(d_t):,} 股", flush=True)
                break
        else:
            d_t = d_t.drop(columns=["trade_date"])
            df_fl = df_fl.merge(d_t, on="ts_code", how="left", suffixes=("","_x"))
            print(f"  merge {name:14s}: {len(d_t):,} 股, 总 {len(df_fl)} 行", flush=True)

    rg = pd.read_parquet(ROOT / "output" / "regimes" / "daily_regime.parquet")
    rg["trade_date"] = rg["trade_date"].astype(str)
    rg_t = rg[rg["trade_date"] == TARGET_DATE]
    if rg_t.empty:
        rg_t = rg.sort_values("trade_date").tail(1).reset_index(drop=True)
    rg_t = rg_t.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                 "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14",
                                 "vol_ratio":"mkt_vol_ratio"})
    for col in ["regime_id","mkt_ret_5d","mkt_ret_20d","mkt_ret_60d","mkt_rsi14","mkt_vol_ratio"]:
        if col not in df_fl.columns and col in rg_t.columns:
            df_fl[col] = rg_t[col].iloc[0]

    rgx = pd.read_parquet(ROOT / "output" / "regime_extra" / "regime_extra.parquet")
    rgx["trade_date"] = rgx["trade_date"].astype(str)
    rgx_t = rgx[rgx["trade_date"] == TARGET_DATE]
    if rgx_t.empty:
        rgx_t = rgx.sort_values("trade_date").tail(1).reset_index(drop=True)
    for col in rgx_t.columns:
        if col == "trade_date": continue
        if col not in df_fl.columns:
            df_fl[col] = rgx_t[col].iloc[0]

    print(f"\n特征拼接完成: {len(df_fl):,} 股 × {len(df_fl.columns)} 列\n", flush=True)

    # 3. v2 4 模型推理
    print("V7c v2 推理中...", flush=True)
    df_fl["r10_pred"] = predict_model(df_fl, "r10_v7c_v2")
    df_fl["r20_pred"] = predict_model(df_fl, "r20_v7c_v2")
    df_fl["sell_10_v6_prob"] = predict_model(df_fl, "sell_10_v7c_v2")
    df_fl["sell_20_v6_prob"] = predict_model(df_fl, "sell_20_v7c_v2")
    s10 = _map_anchored(df_fl["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df_fl["r20_pred"].values, *R20_ANCHOR)
    df_fl["buy_score"] = 0.5 * s10 + 0.5 * s20
    s10s = _map_anchored(df_fl["sell_10_v6_prob"].values, *SELL10)
    s20s = _map_anchored(df_fl["sell_20_v6_prob"].values, *SELL20)
    df_fl["sell_score"] = 0.5 * s10s + 0.5 * s20s

    if "pyr_velocity_20_60" in df_fl.columns:
        p35 = df_fl["pyr_velocity_20_60"].quantile(0.35)
    else:
        p35 = -0.1

    df_fl["quadrant"] = "中性区"
    df_fl.loc[(df_fl["buy_score"] >= 70) & (df_fl["sell_score"] <= 30), "quadrant"] = "理想多"
    df_fl.loc[(df_fl["buy_score"] >= 70) & (df_fl["sell_score"] >= 70), "quadrant"] = "矛盾段"
    df_fl.loc[(df_fl["buy_score"] <= 30) & (df_fl["sell_score"] >= 70), "quadrant"] = "主流空"
    df_fl.loc[(df_fl["buy_score"] <= 30) & (df_fl["sell_score"] <= 30), "quadrant"] = "沉寂"

    if "f1_neg1" in df_fl.columns and "f2_pos1" in df_fl.columns:
        df_fl["v7c_recommend"] = ((df_fl["buy_score"] >= 70) & (df_fl["buy_score"] <= 85) &
                                     (df_fl["sell_score"] <= 30) &
                                     (df_fl["pyr_velocity_20_60"] < p35) &
                                     (df_fl["f1_neg1"].abs() < 0.005) &
                                     (df_fl["f2_pos1"].abs() < 0.005))
    else:
        df_fl["v7c_recommend"] = (df_fl["buy_score"] >= 70) & (df_fl["sell_score"] <= 30)

    OUT_DIR = ROOT / "output" / "v7c_full_inference"
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"v7c_v2_inference_{TARGET_DATE}.csv"
    keep = ["ts_code","trade_date","industry","total_mv","pe","pe_ttm",
             "buy_score","sell_score","quadrant","v7c_recommend",
             "r10_pred","r20_pred","sell_10_v6_prob","sell_20_v6_prob",
             "mfk_pyramid_top_heavy","pyr_velocity_20_60",
             "f1_neg1","f2_pos1","main_net_5d"]
    keep = [c for c in keep if c in df_fl.columns]
    df_fl[keep].to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=== 4 象限分布 ===")
    print(df_fl["quadrant"].value_counts().to_string())

    print(f"\n=== V7c v2 推荐池 ({TARGET_DATE}) ===")
    rec = df_fl[df_fl["v7c_recommend"]]
    print(f"  推荐数: {len(rec):,} 股 (占全市场 {len(rec)/len(df_fl)*100:.1f}%)")
    if len(rec) > 0:
        print(f"  buy_score 中位: {rec['buy_score'].median():.0f}, 平均: {rec['buy_score'].mean():.0f}")
        print(f"  r20 预测中位: {rec['r20_pred'].median():+.2f}%")

        print(f"\n=== Top 30 推荐 (按 r20 预测降序) ===")
        top30 = rec.nlargest(30, "r20_pred")[
            ["ts_code","industry","buy_score","sell_score","r20_pred","sell_20_v6_prob"]
        ]
        print(top30.to_string(index=False))

    # ─── 对比 v1 vs v2 推荐池 ───
    v1_path = ROOT / "output" / "v7c_full_inference" / f"v7c_inference_{TARGET_DATE}.csv"
    if v1_path.exists():
        v1 = pd.read_csv(v1_path)
        v1_rec = set(v1[v1["v7c_recommend"]]["ts_code"].tolist())
        v2_rec = set(rec["ts_code"].tolist())
        common = v1_rec & v2_rec
        only_v1 = v1_rec - v2_rec
        only_v2 = v2_rec - v1_rec
        print(f"\n=== v1 vs v2 推荐池差异 ===")
        print(f"  v1 推荐: {len(v1_rec)}, v2 推荐: {len(v2_rec)}")
        print(f"  重叠: {len(common)} 股 — {sorted(common)}")
        print(f"  仅 v1: {len(only_v1)} 股 — {sorted(only_v1)}")
        print(f"  仅 v2: {len(only_v2)} 股 — {sorted(only_v2)}")

    print(f"\n输出: {out_path}")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
