#!/usr/bin/env python3
"""V7c + V11 LLM 视觉 联合评估 (5 只指定股).

⚠️ 数据限制说明:
  V7c 训练特征数据只到 2026-01-26 (factor_lab/mfk/pyramid_v2/v7_extras)
  LLM 视觉用今天 (2026-05-06) K 线图
  时间错位 3+ 个月 — V7c 评分代表"最近一次有数据时点"的判断

输出每只股的 6 个维度:
  V7c r10/r20 预测 (取 2026-01-26 数据)
  V7c sell_10/sell_20 概率
  V7c buy_score / sell_score (4 象限位置)
  V11 LLM 视觉 (从 v11_test_5 复用)
  综合判断 + 实战建议
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from dotenv import load_dotenv

load_dotenv(".env.cloubic")
load_dotenv(".env")

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
    booster = lgb.Booster(model_file=str(d / "classifier.txt"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta.get("industry_map", {})
    df = df.copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), -1)
    )
    return booster.predict(df[feat_cols])


def load_5_stocks_features():
    """加载 5 只股最近可用日期 (2026-01-26) 的全部特征."""
    PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
    LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
    LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"

    target_codes = [s[0] for s in STOCKS]

    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[df["ts_code"].isin(target_codes)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    if full.empty:
        return None
    # 取每只股最新日期
    full = full.sort_values(["ts_code","trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
    print(f"  factor_lab 5 股最新日期: {full['trade_date'].tolist()}", flush=True)

    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    l20 = pd.read_parquet(LABELS_20, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")

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
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")
    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path,
              columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        full = full.merge(rg, on="trade_date", how="left")
    return full


# V11 LLM 视觉结果 (复用之前 test_v11_5_stocks 的输出)
V11_LLM = {
    "300567.SZ": {
        "trend": "未完整记录",
        "bull": 0.35, "base": 0.45, "bear": 0.20,
        "bull_target": 8.5, "base_target": 1.2, "bear_target": -6.8,
        "key_pattern": "(数据不全)",
    },
    "688037.SH": {
        "trend": "sideways/weak", "ma": "neutral", "vol_match": False,
        "bull": 0.35, "base": 0.45, "bear": 0.20,
        "bull_target": 8.5, "base_target": 1.2, "bear_target": -6.8,
        "support": [160, 140, 120, 90], "resistance": [200, 220, 240, 280],
        "key_pattern": "月线长期箱体, 周线 240 背驰后入中枢, 日线接近二类买点",
        "elliott": "周线 c 浪末端, 日线 5 浪下跌完成",
    },
    "688577.SH": {
        "trend": "downtrend/moderate", "ma": "bearish", "vol_match": False,
        "bull": 0.25, "base": 0.50, "bear": 0.25,
        "bull_target": 8.5, "base_target": 1.2, "bear_target": -6.8,
        "support": [62, 65, 70], "resistance": [75, 80, 85, 90],
        "key_pattern": "月线双顶 (140/95) 后深度回调, 周线空头排列, 接近三类买点",
        "elliott": "主跌 5 浪结束, b 浪反弹",
    },
    "688409.SH": {
        "trend": "uptrend/moderate", "ma": "bullish", "vol_match": False,
        "bull": 0.35, "base": 0.45, "bear": 0.20,
        "bull_target": 8.0, "base_target": 0.0, "bear_target": -7.0,
        "support": [95, 90, 85], "resistance": [105, 110, 115],
        "key_pattern": "月线 W 底后突破中枢, 周线三买回踩确认, 日线中枢上沿",
        "elliott": "周线主升 3 浪末端 / 4 浪修正",
    },
    "688720.SH": {
        "trend": "uptrend/moderate", "ma": "bullish", "vol_match": False,
        "bull": 0.35, "base": 0.45, "bear": 0.20,
        "bull_target": 8.5, "base_target": 2.0, "bear_target": -6.0,
        "support": [62, 65, 70], "resistance": [78, 80, 85],
        "key_pattern": "月线突破上升通道后回落上轨, 周线 W 底+中枢震荡上沿, 接近三类买点",
        "elliott": "c 浪末端, 启动新 1 浪, 0.618 回撤位",
    },
}


def quadrant_label(buy, sell):
    if buy >= 70 and sell <= 30: return "理想多 ⭐"
    if buy >= 70 and sell >= 70: return "矛盾段 ⚠"
    if buy <= 30 and sell >= 70: return "主流空"
    if buy <= 30 and sell <= 30: return "沉寂"
    return "中性区"


def main():
    t0 = time.time()
    print("加载 5 只股 V7c 特征...", flush=True)
    df = load_5_stocks_features()
    if df is None or df.empty:
        print("⚠ 5 只股都不在 factor_lab 数据里"); return
    print(f"  匹配股票: {len(df)}/5", flush=True)

    # V7c 4 模型预测
    df["r10_pred"] = predict_model(df, "r10_v4_all")
    df["r20_pred"] = predict_model(df, "r20_v4_all")
    df["sell_10_v6_prob"] = predict_model(df, "sell_10_v6")
    df["sell_20_v6_prob"] = predict_model(df, "sell_20_v6")
    s10 = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df["r20_pred"].values, *R20_ANCHOR)
    df["buy_score"] = 0.5 * s10 + 0.5 * s20
    s10s = _map_anchored(df["sell_10_v6_prob"].values, *SELL10_V6)
    s20s = _map_anchored(df["sell_20_v6_prob"].values, *SELL20_V6)
    df["sell_score"] = 0.5 * s10s + 0.5 * s20s

    # V7c 推荐池条件
    p35_v20_global = 0.0  # 简化, 用 0 作 pyr_velocity 阈值 (实际应取全市场分位)

    print(f"\n{'='*100}", flush=True)
    print(f"### V7c + V11 LLM 联合评估", flush=True)
    print(f"{'='*100}\n", flush=True)

    for _, row in df.iterrows():
        ts_code = row["ts_code"]
        name = next((n for c, n in STOCKS if c == ts_code), "?")
        v11 = V11_LLM.get(ts_code, {})

        print(f"## {ts_code} {name}", flush=True)
        print(f"V7c 数据日期: {row['trade_date']} (今天 2026-05-06, 时间错位 ~3 月)", flush=True)
        print(f"行业: {row.get('industry', '?')}", flush=True)

        # V7c
        bs = row["buy_score"]; ss = row["sell_score"]
        quad = quadrant_label(bs, ss)
        print(f"\n【V7c 量化 (2026-01-26)】", flush=True)
        print(f"  buy_score = {bs:.0f}/100, sell_score = {ss:.0f}/100", flush=True)
        print(f"  4 象限 → {quad}", flush=True)
        print(f"  r10 预测 {row['r10_pred']:+.2f}%, r20 预测 {row['r20_pred']:+.2f}%", flush=True)
        print(f"  sell_10/20 概率 {row['sell_10_v6_prob']*100:.0f}% / {row['sell_20_v6_prob']*100:.0f}%", flush=True)
        print(f"  pyr_velocity_20_60 = {row.get('pyr_velocity_20_60', np.nan):+.3f}", flush=True)

        # V11 LLM
        print(f"\n【V11 LLM 视觉 (2026-05-06)】", flush=True)
        print(f"  趋势: {v11.get('trend', '?')}", flush=True)
        print(f"  MA 排列: {v11.get('ma', '?')}, 量价配合: {v11.get('vol_match', '?')}", flush=True)
        print(f"  形态: {v11.get('key_pattern', '?')}", flush=True)
        print(f"  Elliott: {v11.get('elliott', '?')}", flush=True)
        print(f"  3 场景: Bull {v11.get('bull',0)*100:.0f}%/+{v11.get('bull_target',0):.1f}%, "
              f"Base {v11.get('base',0)*100:.0f}%/{v11.get('base_target',0):+.1f}%, "
              f"Bear {v11.get('bear',0)*100:.0f}%/{v11.get('bear_target',0):+.1f}%", flush=True)

        # 综合判断
        print(f"\n【综合判断】", flush=True)
        buy_consensus = bs >= 60 and v11.get("bull", 0) >= 0.4
        sell_warning = ss >= 60 or v11.get("bull", 1) < 0.25
        if quad == "理想多 ⭐":
            print(f"  ⭐ V7c 理想多段, 直接入场建议 (V7c 实战 r20=+7.28%)", flush=True)
        elif quad == "矛盾段 ⚠":
            v11_bull = v11.get("bull", 0)
            if v11_bull >= 0.5:
                print(f"  🌟 矛盾段 + LLM bull≥0.5 (V11 反挖 +3.76pp), 建议小仓试探 (2-3%)", flush=True)
            else:
                print(f"  ❌ 矛盾段 + LLM bull<0.5, 跳过", flush=True)
        elif buy_consensus and not sell_warning:
            print(f"  ✅ V7c+LLM 双向看多, 中性区可介入", flush=True)
        elif quad == "沉寂":
            print(f"  💤 沉寂段, 收益弱但风险也低 (r20≈+2.5%)", flush=True)
        else:
            print(f"  ⚪ 信号不强, 暂观察", flush=True)
        print(f"\n{'-'*100}", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
