#!/usr/bin/env python3
"""阶段 4: V7c 用最新数据 (2026-04-20) 给 5 股推理 + V11 LLM 联合评估.

数据: 全部更新到 2026-04-20 (cogalpha 到 2026-03-19, 用最近可用)
输出: 5 股 V7c 评分 + V11 LLM 视觉 + 综合推荐
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"

R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)

STOCKS = [
    ("300567.SZ", "精测电子"),
    ("688037.SH", "芯源微"),
    ("688577.SH", "浙海德曼"),
    ("688409.SH", "富创精密"),
    ("688720.SH", "艾森股份"),
]

# V11 LLM 视觉结果 (2026-04-20 时点)
V11_LLM = {
    "300567.SZ": {"trend": "横盘震荡", "ma": "neutral", "vol_match": False,
                   "bull": 0.35, "base": 0.45, "bear": 0.20,
                   "bull_target": 8.5, "base_target": 1.2, "bear_target": -6.8,
                   "key_pattern": "(类似日级形态)"},
    "688037.SH": {"trend": "sideways/weak", "ma": "neutral", "vol_match": False,
                   "bull": 0.35, "base": 0.45, "bear": 0.20,
                   "bull_target": 8.5, "base_target": 1.2, "bear_target": -6.8,
                   "key_pattern": "月线箱体, 周线 240 背驰后入中枢, 日线接近二类买点"},
    "688577.SH": {"trend": "downtrend/moderate", "ma": "bearish", "vol_match": False,
                   "bull": 0.25, "base": 0.50, "bear": 0.25,
                   "bull_target": 8.5, "base_target": 1.2, "bear_target": -6.8,
                   "key_pattern": "月线双顶 (140/95) 后深度回调, 周线空头排列, 接近三类买点"},
    "688409.SH": {"trend": "uptrend/moderate", "ma": "bullish", "vol_match": False,
                   "bull": 0.35, "base": 0.45, "bear": 0.20,
                   "bull_target": 8.0, "base_target": 0.0, "bear_target": -7.0,
                   "key_pattern": "月线 W 底后突破中枢, 周线三买回踩确认, 日线中枢上沿"},
    "688720.SH": {"trend": "uptrend/moderate", "ma": "bullish", "vol_match": False,
                   "bull": 0.35, "base": 0.45, "bear": 0.20,
                   "bull_target": 8.5, "base_target": 2.0, "bear_target": -6.0,
                   "key_pattern": "月线突破上升通道后回落上轨, 周线 W 底+中枢上沿, 接近三类买点"},
}


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
    # 缺失列填 NaN, LGBM 内置 missing handling
    for fc in feat_cols:
        if fc not in df.columns:
            df[fc] = np.nan
    return booster.predict(df[feat_cols])


def quadrant(buy, sell):
    if buy >= 70 and sell <= 30: return "理想多 ⭐"
    if buy >= 70 and sell >= 70: return "矛盾段 ⚠"
    if buy <= 30 and sell >= 70: return "主流空"
    if buy <= 30 and sell <= 30: return "沉寂"
    return "中性区"


def main():
    t0 = time.time()
    print("=== 阶段 4: V7c 用 2026-04-20 数据 5 股推理 + V11 LLM 联合 ===\n")

    # 加载 5 股全部特征 (从 factor_lab 5 股增量 + 各 features.parquet)
    target_codes = [s[0] for s in STOCKS]

    # 1. factor_lab 5 股增量数据
    fl_path = ROOT / "output/factor_lab_3y/factor_groups_5stocks/5stocks_2026q1q2.parquet"
    df_fl = pd.read_parquet(fl_path)
    df_fl["trade_date"] = df_fl["trade_date"].astype(str)
    # 取每只股最新一行 (2026-04-20)
    df_fl = df_fl.sort_values(["ts_code","trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
    print(f"factor_lab 5 股最新行: {df_fl[['ts_code','trade_date']].values.tolist()}\n")

    # 2. merge 各 features
    for path in [
        ROOT / "output" / "amount_features" / "amount_features.parquet",
        ROOT / "output" / "regime_extra" / "regime_extra.parquet",
        ROOT / "output" / "moneyflow" / "features.parquet",
        ROOT / "output" / "cogalpha_features" / "features.parquet",
        ROOT / "output" / "mfk_features" / "features.parquet",
        ROOT / "output" / "pyramid_v2" / "features.parquet",
        ROOT / "output" / "v7_extras" / "features.parquet",
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        d["trade_date"] = d["trade_date"].astype(str)
        # 仅保留 5 股 + 最近 trade_date
        if "ts_code" in d.columns:
            d = d[d["ts_code"].isin(target_codes)]
            # 对每股取最新或刚好 2026-04-20 的行
            d = d.sort_values(["ts_code","trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
            df_fl = df_fl.merge(d, on=["ts_code","trade_date"], how="left",
                                 suffixes=("","_x"))
            # 如果 merge 失败 (不同日期), 用 ts_code only merge
            if df_fl["ts_code"].count() < 5 or df_fl[d.columns[-1]].isna().all():
                df_fl = df_fl.drop(columns=[c for c in d.columns if c.endswith("_x") or (c not in df_fl.columns and c != "ts_code")], errors="ignore")
                df_fl = df_fl.merge(d.drop(columns=["trade_date"]), on="ts_code", how="left",
                                     suffixes=("","_x"))
        else:
            # regime_extra 是市场级
            d = d.sort_values("trade_date").tail(1).reset_index(drop=True)
            tmpl = d.iloc[0].drop("trade_date")
            for col, val in tmpl.items():
                if col not in df_fl.columns:
                    df_fl[col] = val

    # regime_id 从最新 daily_regime
    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path,
              columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.sort_values("trade_date").tail(1).reset_index(drop=True)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        for col in ["regime_id","mkt_ret_5d","mkt_ret_20d","mkt_ret_60d","mkt_rsi14","mkt_vol_ratio"]:
            if col not in df_fl.columns:
                df_fl[col] = rg[col].iloc[0]

    print(f"特征拼接完成: {len(df_fl.columns)} 列\n")

    # 3. V7c 4 模型推理
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

    # 4. 输出
    print("="*100)
    for ts_code, name in STOCKS:
        row = df_fl[df_fl["ts_code"] == ts_code]
        if row.empty: continue
        row = row.iloc[0]
        v11 = V11_LLM.get(ts_code, {})

        print(f"\n## {ts_code} {name}")
        print(f"   行业: {row.get('industry', '?')}, 数据日期: {row['trade_date']}")
        print(f"\n【V7c 量化 (2026-04-20)】")
        print(f"   buy_score = {row['buy_score']:.0f}/100")
        print(f"   sell_score = {row['sell_score']:.0f}/100")
        print(f"   4 象限 → {quadrant(row['buy_score'], row['sell_score'])}")
        print(f"   r10 预测: {row['r10_pred']:+.2f}%, r20 预测: {row['r20_pred']:+.2f}%")
        print(f"   sell_10/20 概率: {row['sell_10_v6_prob']*100:.0f}% / {row['sell_20_v6_prob']*100:.0f}%")
        if "pyr_velocity_20_60" in row.index and pd.notna(row["pyr_velocity_20_60"]):
            print(f"   pyr_velocity_20_60: {row['pyr_velocity_20_60']:+.3f}")
        if "mfk_pyramid_top_heavy" in row.index and pd.notna(row["mfk_pyramid_top_heavy"]):
            print(f"   mfk_pyramid_top_heavy: {row['mfk_pyramid_top_heavy']:.2f}")

        print(f"\n【V11 LLM 视觉 (2026-04-20)】")
        print(f"   趋势: {v11.get('trend')}, MA: {v11.get('ma')}, 量价配合: {v11.get('vol_match')}")
        print(f"   形态: {v11.get('key_pattern')}")
        print(f"   3 场景: Bull {v11.get('bull',0)*100:.0f}% (target {v11.get('bull_target',0):+.1f}%), "
              f"Base {v11.get('base',0)*100:.0f}% ({v11.get('base_target',0):+.1f}%), "
              f"Bear {v11.get('bear',0)*100:.0f}% ({v11.get('bear_target',0):+.1f}%)")

        # 综合推荐
        bs = row["buy_score"]; ss = row["sell_score"]; v11_bull = v11.get("bull", 0.35)
        print(f"\n【联合判断】")
        if bs >= 70 and ss <= 30:
            print(f"   ⭐ V7c 理想多, 直接入场 (V7c 实战 +4.82pp/月)")
        elif bs >= 70 and ss >= 70:
            if v11_bull >= 0.5:
                print(f"   🌟 V7c 矛盾段 + LLM bull≥0.5, V11 反挖 (+3.76pp), 小仓试探")
            else:
                print(f"   ❌ 矛盾段 + LLM 看空, 跳过")
        elif 50 <= bs < 70 and v11_bull >= 0.4:
            print(f"   ⚪ V7c 中性偏多 + LLM 一致看多, 可参与")
        elif bs >= 60:
            print(f"   ⚪ V7c 偏多 但未到推荐池, 观察")
        else:
            print(f"   ❌ V7c 不推荐, 跳过")
        print(f"\n{'-'*100}")

    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
