#!/usr/bin/env python3
"""V6 综合分析: P0 sell+mfk + P1 过滤组合 + P3 交易成本.

使用 V6 sell 模型 (含 mfk) 重算 sell_score, 然后:
- P1: buy_70_85 段加 5 类过滤 (mfk_gold/cross_strength/pyramid/bull_regime/industry)
- P3: 扣交易成本 (-0.5% stock, -0.2% ETF, 净差 -0.3%)
- 配对 alpha vs 6 ETF (含 net alpha)

输出 output/v6/{filters_alpha.csv, net_alpha.csv, summary.txt}
"""
from __future__ import annotations
import os, struct, datetime as dt, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
OUT_DIR = ROOT / "output" / "v6"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")
TEST_START = "20250501"
TEST_END   = "20260126"

ETFS = [
    ("sh000001", "上证指数"),
    ("sh000300", "沪深300"),
    ("sh000905", "中证500"),
    ("sh000852", "中证1000"),
    ("sz399006", "创业板指"),
    ("sz399303", "北证50"),
]

# V4 锚点 (r10/r20 用 v4 不变)
R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
# V6 sell 锚点 — 训练完后从 meta 读取
SELL10_V6_ANCHOR = None  # 占位, 运行时读取
SELL20_V6_ANCHOR = None

# 交易成本 (双边)
STOCK_COST = 0.5  # %
ETF_COST = 0.2    # %


def read_tdx_close(market_code):
    market = market_code[:2]; code = market_code[2:]
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    data = p.read_bytes()
    n = len(data) // 32
    rows = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append((d.strftime("%Y%m%d"), f[4]/100.0))
    return rows


def compute_etf_future_r20(etf_rows):
    df = pd.DataFrame(etf_rows, columns=["trade_date", "close"])
    df["entry"] = df["close"].shift(-1)
    df["exit_20"] = df["close"].shift(-21)
    df["etf_r20"] = (df["exit_20"] / df["entry"] - 1) * 100
    return df[["trade_date", "etf_r20"]].dropna()


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
    if not (d / "classifier.txt").exists(): return None
    booster = lgb.Booster(model_file=str(d / "classifier.txt"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta.get("industry_map", {})
    df = df.copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), -1)
    )
    return booster.predict(df[feat_cols])


def load_oos():
    PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
    LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
    LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
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
        ROOT / "output" / "moneyflow_v2" / "features.parquet",
        ROOT / "output" / "cogalpha_features" / "features.parquet",
        ROOT / "output" / "mfk_features" / "features.parquet",
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


def get_v6_anchors():
    """读 V6 sell 模型 meta 取 OOS 锚点."""
    s10 = json.loads((PROD_DIR / "sell_10_v6" / "meta.json").read_text(encoding="utf-8"))
    s20 = json.loads((PROD_DIR / "sell_20_v6" / "meta.json").read_text(encoding="utf-8"))
    return ((s10["anchor_p5"], s10["anchor_p50"], s10["anchor_p95"]),
            (s20["anchor_p5"], s20["anchor_p50"], s20["anchor_p95"]))


def main():
    t0 = time.time()

    # 锚点
    s10a, s20a = get_v6_anchors()
    print(f"V6 sell 锚点:")
    print(f"  sell_10: {s10a}")
    print(f"  sell_20: {s20a}")

    print(f"\n[{int(time.time()-t0)}s] 加载 ETF baseline...")
    etf_dfs = []
    for code, name in ETFS:
        rows = read_tdx_close(code)
        if not rows: continue
        ed = compute_etf_future_r20(rows)
        ed = ed.rename(columns={"etf_r20": f"r20_{code}"})
        ed = ed[(ed["trade_date"] >= TEST_START) & (ed["trade_date"] <= TEST_END)]
        etf_dfs.append(ed)
    etf_all = etf_dfs[0]
    for d in etf_dfs[1:]:
        etf_all = etf_all.merge(d, on="trade_date", how="outer")

    print(f"[{int(time.time()-t0)}s] 加载 OOS + V4 r10/r20 + V6 sell 预测...")
    df = load_oos()
    df["r10_pred"] = predict_model(df, "r10_v4_all")
    df["r20_pred"] = predict_model(df, "r20_v4_all")
    df["sell_10_v6_prob"] = predict_model(df, "sell_10_v6")
    df["sell_20_v6_prob"] = predict_model(df, "sell_20_v6")
    # 同时也算 v4 sell 用于对比
    df["sell_10_v4_prob"] = predict_model(df, "sell_10_v4")
    df["sell_20_v4_prob"] = predict_model(df, "sell_20_v4")

    # buy_score (V4)
    s10s_buy = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
    s20s_buy = _map_anchored(df["r20_pred"].values, *R20_ANCHOR)
    df["buy_score"] = 0.5 * s10s_buy + 0.5 * s20s_buy

    # sell_score V6 (含 mfk)
    s10_sell_v6 = _map_anchored(df["sell_10_v6_prob"].values, *s10a)
    s20_sell_v6 = _map_anchored(df["sell_20_v6_prob"].values, *s20a)
    df["sell_score_v6"] = 0.5 * s10_sell_v6 + 0.5 * s20_sell_v6
    # sell_score V4 (无 mfk) 用于对比
    s10_sell_v4 = _map_anchored(df["sell_10_v4_prob"].values, 0.27, 0.48, 0.70)
    s20_sell_v4 = _map_anchored(df["sell_20_v4_prob"].values, 0.04, 0.43, 0.88)
    df["sell_score_v4"] = 0.5 * s10_sell_v4 + 0.5 * s20_sell_v4

    df = df.merge(etf_all, on="trade_date", how="left")
    df = df.dropna(subset=["r20"])

    print(f"[{int(time.time()-t0)}s] OOS samples: {len(df):,}")

    # ── P0: V6 vs V4 sell_score top decile lift ──
    print(f"\n[{int(time.time()-t0)}s] === P0: V6 vs V4 sell_score 避雷力对比 ===")
    rows = []
    for v_name, ss_col, sp10_col, sp20_col in [
        ("V4 (无 mfk)", "sell_score_v4", "sell_10_v4_prob", "sell_20_v4_prob"),
        ("V6 (含 mfk)", "sell_score_v6", "sell_10_v6_prob", "sell_20_v6_prob"),
    ]:
        # sell top 10 决策段
        for top_pct in [10, 5, 1]:
            thresh = df[ss_col].quantile(1 - top_pct/100)
            seg = df[df[ss_col] >= thresh]
            row = {
                "version": v_name, "top_pct": top_pct,
                "n": len(seg), "thresh": thresh,
                "dd20_lt_8pct": (seg["max_dd_20"] <= -8).mean() * 100,
                "dd20_lt_15pct": (seg["max_dd_20"] <= -15).mean() * 100,
                "r20_mean": seg["r20"].mean(),
            }
            rows.append(row)
    p0_df = pd.DataFrame(rows)
    print(p0_df.round(3).to_string())
    p0_df.to_csv(OUT_DIR / "p0_sell_v4_vs_v6.csv", index=False, encoding="utf-8-sig")

    # ── P1: buy_70_85+SELL≤30 加多种过滤 (使用 V6 sell) ──
    print(f"\n[{int(time.time()-t0)}s] === P1: buy_70_85 多过滤组合 (V6 sell) ===")
    base_mask = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score_v6"] <= 30))
    base = df[base_mask]
    print(f"  Baseline (buy_70_85 + SELL_v6≤30): n={len(base):,}, r20={base['r20'].mean():.2f}%")

    # 过滤定义
    filters = {
        "mfk_gold (cross=+1)":     df["mfk_main_cross_state"] == 1,
        "mfk_strong_cross":        df["mfk_main_cross_strength"] > 0.05,
        "low_pyramid (机构占比 < 0.45)":      df["mfk_pyramid_top_heavy"] < 0.45,
        "high_macd_hist":          df["mfk_main_macd_hist"] > 0,
        "bull_regime (mkt 60d>0)": df["mkt_ret_60d"] > 0,
        "main_inflow (main_net_5d>0)":         df["main_net_5d"] > 0,
        "no_extreme_event":        (df["f1_main_in_red"].abs() < 0.005) & (df["f2_main_out_green"].abs() < 0.005),
    }

    p1_rows = []
    # 基线
    for code, name in ETFS:
        col = f"r20_{code}"
        if col not in base.columns: continue
        sub = base[["r20", col]].dropna()
        alpha = sub["r20"] - sub[col]
        p1_rows.append({
            "filter": "baseline (buy_70_85+SELL_v6≤30)", "n": len(sub),
            "etf": code, "stock_r20": sub["r20"].mean(),
            "etf_r20": sub[col].mean(), "alpha": alpha.mean(),
            "win_pct": (alpha > 0).mean() * 100,
            "p_value": stats.ttest_1samp(alpha, 0)[1],
        })

    for fname, fmask in filters.items():
        m = base_mask & fmask
        seg = df[m]
        if len(seg) < 100: continue
        for code, name in ETFS:
            col = f"r20_{code}"
            if col not in seg.columns: continue
            sub = seg[["r20", col]].dropna()
            if len(sub) < 50: continue
            alpha = sub["r20"] - sub[col]
            p1_rows.append({
                "filter": fname, "n": len(sub),
                "etf": code, "stock_r20": sub["r20"].mean(),
                "etf_r20": sub[col].mean(), "alpha": alpha.mean(),
                "win_pct": (alpha > 0).mean() * 100,
                "p_value": stats.ttest_1samp(alpha, 0)[1],
            })
    # 多重过滤组合 (top-3)
    combo_filters = {
        "mfk_gold + bull_regime": filters["mfk_gold (cross=+1)"] & filters["bull_regime (mkt 60d>0)"],
        "mfk_gold + low_pyramid": filters["mfk_gold (cross=+1)"] & filters["low_pyramid (机构占比 < 0.45)"],
        "mfk_gold + main_inflow": filters["mfk_gold (cross=+1)"] & filters["main_inflow (main_net_5d>0)"],
        "low_pyramid + bull_regime": filters["low_pyramid (机构占比 < 0.45)"] & filters["bull_regime (mkt 60d>0)"],
        "全套 (gold + low_pyr + bull)": (filters["mfk_gold (cross=+1)"] &
                                         filters["low_pyramid (机构占比 < 0.45)"] &
                                         filters["bull_regime (mkt 60d>0)"]),
    }
    for fname, fmask in combo_filters.items():
        m = base_mask & fmask
        seg = df[m]
        if len(seg) < 100: continue
        for code, name in ETFS:
            col = f"r20_{code}"
            if col not in seg.columns: continue
            sub = seg[["r20", col]].dropna()
            if len(sub) < 50: continue
            alpha = sub["r20"] - sub[col]
            p1_rows.append({
                "filter": fname, "n": len(sub),
                "etf": code, "stock_r20": sub["r20"].mean(),
                "etf_r20": sub[col].mean(), "alpha": alpha.mean(),
                "win_pct": (alpha > 0).mean() * 100,
                "p_value": stats.ttest_1samp(alpha, 0)[1],
            })
    p1_df = pd.DataFrame(p1_rows)
    p1_df.to_csv(OUT_DIR / "p1_filters_alpha.csv", index=False, encoding="utf-8-sig")

    # 简洁透视: 每个 filter vs hs300/zz500/cyb
    pivot = p1_df[p1_df["etf"].isin(["sh000300", "sh000905", "sz399006"])].copy()
    pivot_alpha = pivot.pivot(index="filter", columns="etf", values="alpha").round(2)
    pivot_alpha["sample_n"] = pivot.groupby("filter")["n"].first()
    pivot_alpha = pivot_alpha.rename(columns={"sh000300":"vs_hs300_α","sh000905":"vs_zz500_α","sz399006":"vs_cyb_α"})
    print(f"\n  P1 alpha 透视 (按 ETF, 平均超额收益):")
    print(pivot_alpha.to_string())

    # ── P3: 加交易成本 ──
    print(f"\n[{int(time.time()-t0)}s] === P3: 加交易成本后 net alpha ===")
    print(f"  stock 双向成本 -{STOCK_COST}%, ETF 双向成本 -{ETF_COST}%, 净差 -{STOCK_COST-ETF_COST}%")
    p3_rows = []
    # 用 P1 同样过滤组合
    for fname, fmask in {**filters, **combo_filters, "baseline": pd.Series([True]*len(df), index=df.index)}.items():
        m = base_mask & fmask
        seg = df[m]
        if len(seg) < 100: continue
        for code, name in ETFS:
            col = f"r20_{code}"
            if col not in seg.columns: continue
            sub = seg[["r20", col]].dropna()
            if len(sub) < 50: continue
            stock_net = sub["r20"] - STOCK_COST
            etf_net = sub[col] - ETF_COST
            net_alpha = stock_net - etf_net
            p3_rows.append({
                "filter": fname, "n": len(sub),
                "etf": code,
                "gross_alpha": (sub["r20"] - sub[col]).mean(),
                "net_alpha": net_alpha.mean(),
                "stock_net_r20": stock_net.mean(),
                "etf_net_r20": etf_net.mean(),
                "net_win_pct": (net_alpha > 0).mean() * 100,
                "net_p": stats.ttest_1samp(net_alpha, 0)[1],
            })
    p3_df = pd.DataFrame(p3_rows)
    p3_df.to_csv(OUT_DIR / "p3_net_alpha.csv", index=False, encoding="utf-8-sig")

    # P3 透视
    pivot3 = p3_df[p3_df["etf"].isin(["sh000300", "sh000905", "sz399006"])].copy()
    pivot3_net = pivot3.pivot(index="filter", columns="etf", values="net_alpha").round(2)
    pivot3_net.columns = [f"net_α_vs_{c}" for c in pivot3_net.columns]
    pivot3_n = pivot3.groupby("filter")["n"].first()
    pivot3_combined = pivot3_net.copy()
    pivot3_combined["n"] = pivot3_n
    pivot3_combined = pivot3_combined[["n"] + [c for c in pivot3_combined.columns if c != "n"]]
    pivot3_combined = pivot3_combined.sort_values("net_α_vs_sh000300", ascending=False)
    print(f"\n  P3 net alpha 透视 (扣 stock -0.5%, ETF -0.2%, 按 vs hs300 排序):")
    print(pivot3_combined.to_string())
    pivot3_combined.to_csv(OUT_DIR / "p3_net_alpha_pivot.csv", encoding="utf-8-sig")

    # ── 输出汇总文本 ──
    summary = []
    summary.append("=" * 80)
    summary.append("V6 综合分析: P0 sell+mfk + P1 多过滤 + P3 交易成本")
    summary.append("=" * 80)
    summary.append(f"\nOOS: {len(df):,} 样本 (2025-05 → 2026-01)")
    summary.append(f"V6 sell 锚点: 10 {s10a}, 20 {s20a}")
    summary.append(f"\n=== P0: V6 vs V4 sell 避雷力 ===")
    summary.append(p0_df.round(3).to_string())
    summary.append(f"\n=== P1: 过滤组合 alpha 透视 ===")
    summary.append(pivot_alpha.to_string())
    summary.append(f"\n=== P3: 净 alpha 透视 (扣交易成本) ===")
    summary.append(pivot3_combined.to_string())
    Path(OUT_DIR / "summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"\n总耗时 {time.time()-t0:.0f}s, 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
