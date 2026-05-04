#!/usr/bin/env python3
"""V4 全维度评估: 25 因子单因子 + 相关矩阵 + 段级激活 + 分桶 IC + 高分共振 + SOTA.

输出:
  output/v4_compare/single_factor_ic.csv      (25 因子单因子 RankIC)
  output/v4_compare/factor_correlation.csv    (25 因子 + 5 regime 相关矩阵)
  output/v4_compare/segment_activation.csv    (8 段 × 25 因子均值 lift)
  output/v4_compare/bucket_ic.csv             (mv_bucket × pe_bucket × regime IC)
  output/v4_compare/high_score_resonance.csv  (高 buy/sell + V2 + mfk 共振)
  output/v4_compare/sota_comparison.csv       (vs CogAlpha)
  output/v4_compare/full_summary.txt          (汇总)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
OUT_DIR = ROOT / "output" / "v4_compare"
OUT_DIR.mkdir(exist_ok=True)

TEST_START = "20250501"
TEST_END   = "20260126"

V2_COLS = ["f1_main_in_red","f2_main_out_green","f3_main_follow_breakout",
            "f4_dispersion_price_align","f5_main_acc_velocity_60d",
            "f6_inst_buy_pressure","f7_silent_accumulation","f8_quiet_distribution"]
MFK_COLS = ["mfk_main_ma5","mfk_main_ma20","mfk_main_ma_diff",
             "mfk_main_cross_state","mfk_main_days_in_cross","mfk_main_cross_strength",
             "mfk_main_macd","mfk_main_macd_hist","mfk_main_macd_zero_dist",
             "mfk_smart_dumb_spread","mfk_pyramid_top_heavy","mfk_inst_acc_velocity",
             "mfk_main_price_sync_20d","mfk_main_lead_5d",
             "mfk_main_velocity_5d","mfk_main_consec_above_ma20","mfk_main_acc_ratio_60d"]
ALL_NEW = V2_COLS + MFK_COLS

REGIME_COLS = ["mkt_ret_5d","mkt_ret_20d","mkt_ret_60d","mkt_rsi14","mkt_vol_ratio"]


def load_oos():
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10","max_dd_10"])
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
        ROOT / "output" / "moneyflow_v2" / "features.parquet",
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


def main():
    t0 = time.time()
    print(f"[{int(time.time()-t0)}s] 加载 OOS...")
    df = load_oos()
    print(f"  shape: {df.shape}")

    # 计算 buy/sell score (用 r20_v3)
    print(f"[{int(time.time()-t0)}s] 预测 + 评分...")
    df["r10_pred"] = predict_model(df, "r10_all")
    df["r20_pred"] = predict_model(df, "r20_v3_all")
    df["sell_10_prob"] = predict_model(df, "sell_10")
    df["sell_20_prob"] = predict_model(df, "sell_20")

    p5, p50, p95 = (df["r20_pred"].quantile(0.05),
                    df["r20_pred"].quantile(0.50),
                    df["r20_pred"].quantile(0.95))
    s10 = _map_anchored(df["r10_pred"].values, 0.72, 0.94, 1.40)
    s20 = _map_anchored(df["r20_pred"].values, p5, p50, p95)
    df["buy_score"] = 0.5 * s10 + 0.5 * s20
    s10s = _map_anchored(df["sell_10_prob"].values, 0.05, 0.20, 0.64)
    s20s = _map_anchored(df["sell_20_prob"].values, 0.01, 0.07, 0.67)
    df["sell_score"] = 0.5 * s10s + 0.5 * s20s

    # ──── 1. 单因子 RankIC ────
    print(f"\n[{int(time.time()-t0)}s] === 1. 25 因子单因子 IC ===")
    ic_rows = []
    for fc in ALL_NEW:
        if fc not in df.columns: continue
        m = df["r20"].notna() & df[fc].notna() & np.isfinite(df[fc])
        if m.sum() < 1000: continue
        d_clipped = df.loc[m, fc].clip(df.loc[m, fc].quantile(0.001),
                                          df.loc[m, fc].quantile(0.999))
        ic_r10 = stats.spearmanr(d_clipped, df.loc[m, "r10"])[0] if df.loc[m, "r10"].notna().sum() > 1000 else np.nan
        ic_r20 = stats.spearmanr(d_clipped, df.loc[m, "r20"])[0]
        ic_rows.append({
            "factor": fc, "group": "v2" if fc in V2_COLS else "mfk",
            "n": int(m.sum()),
            "RankIC_r10": ic_r10, "RankIC_r20": ic_r20,
            "mean": d_clipped.mean(), "std": d_clipped.std(),
        })
    ic_df = pd.DataFrame(ic_rows).sort_values("RankIC_r20", key=abs, ascending=False)
    print(ic_df.round(4).to_string())
    ic_df.to_csv(OUT_DIR / "single_factor_ic.csv", index=False, encoding="utf-8-sig")

    # ──── 2. 因子相关矩阵 ────
    print(f"\n[{int(time.time()-t0)}s] === 2. 因子相关矩阵 ===")
    corr_cols = [c for c in ALL_NEW if c in df.columns] + REGIME_COLS
    corr_df = df[corr_cols].corr(method="spearman").round(3)
    corr_df.to_csv(OUT_DIR / "factor_correlation.csv", encoding="utf-8-sig")
    print(f"  保存 {len(corr_cols)}×{len(corr_cols)} 相关矩阵")
    # 高相关 (|r|>0.7) 警告
    high_corr = []
    for i, a in enumerate(corr_cols):
        for j in range(i+1, len(corr_cols)):
            b = corr_cols[j]
            r = corr_df.loc[a, b]
            if abs(r) >= 0.7:
                high_corr.append((a, b, r))
    if high_corr:
        print(f"  [HIGH CORR] |r|>=0.7 factor pairs:")
        for a, b, r in high_corr:
            print(f"    {a:30s} <-> {b:30s}: {r:+.3f}")

    # ──── 3. 段级激活 ────
    print(f"\n[{int(time.time()-t0)}s] === 3. 段级激活度 ===")
    bp80, bp20 = df["buy_score"].quantile(0.80), df["buy_score"].quantile(0.20)
    sp80, sp20 = df["sell_score"].quantile(0.80), df["sell_score"].quantile(0.20)
    segments = [
        ("ALL",        df.index >= 0),
        ("sell_top20", df["sell_score"] >= sp80),
        ("sell_bot20", df["sell_score"] <= sp20),
        ("buy_top20",  df["buy_score"] >= bp80),
        ("buy_bot20",  df["buy_score"] <= bp20),
        ("理想多",     (df["buy_score"] >= 70) & (df["sell_score"] <= 30)),
        ("矛盾",       (df["buy_score"] >= 70) & (df["sell_score"] >= 70)),
        ("沉寂",       (df["buy_score"] <= 30) & (df["sell_score"] <= 30)),
    ]
    seg_rows = []
    for label, mask in segments:
        g = df[mask]
        row = {"segment": label, "n": len(g)}
        for fc in ALL_NEW:
            if fc not in g.columns: continue
            v = g[fc].dropna()
            row[fc] = v.mean() if len(v) > 0 else np.nan
        if "r10" in g.columns: row["r10_mean"] = g["r10"].dropna().mean()
        if "r20" in g.columns: row["r20_mean"] = g["r20"].dropna().mean()
        if "max_dd_20" in g.columns:
            row["dd20_lt_8pct"] = (g["max_dd_20"].dropna() <= -8).mean() * 100
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(OUT_DIR / "segment_activation.csv", index=False, encoding="utf-8-sig")

    # ──── 4. 分桶 IC ────
    print(f"\n[{int(time.time()-t0)}s] === 4. 分桶 IC (mv × regime) ===")
    bucket_rows = []
    if "mv_bucket" in df.columns and "regime_id" in df.columns:
        for mv in sorted(df["mv_bucket"].dropna().unique()):
            for rg in sorted(df["regime_id"].dropna().unique()):
                m = (df["mv_bucket"] == mv) & (df["regime_id"] == rg) & df["r20"].notna()
                if m.sum() < 500: continue
                row = {"mv_bucket": mv, "regime_id": int(rg), "n": int(m.sum())}
                # 选 5 个最强单因子的桶 IC
                top5 = ic_df.nlargest(5, "RankIC_r20", keep="all").iloc[:5]
                for _, fr in top5.iterrows():
                    fc = fr["factor"]
                    if fc not in df.columns: continue
                    sub = df.loc[m, [fc, "r20"]].dropna()
                    if len(sub) < 100: continue
                    ic = stats.spearmanr(sub[fc], sub["r20"])[0]
                    row[f"{fc}_IC"] = ic
                bucket_rows.append(row)
    bucket_df = pd.DataFrame(bucket_rows)
    if not bucket_df.empty:
        bucket_df.to_csv(OUT_DIR / "bucket_ic.csv", index=False, encoding="utf-8-sig")
        print(bucket_df.head(10).round(4).to_string())

    # ──── 5. 高分共振分析 ────
    print(f"\n[{int(time.time()-t0)}s] === 5. 高分共振 (buy_high × sell_low × mfk_cross_state=+1 × f2 高) ===")
    res_rows = []
    # 单纯高 buy_score
    for label, m in [
        ("buy_top20",       df["buy_score"] >= bp80),
        ("buy_top20+mfk_gold", (df["buy_score"] >= bp80) & (df["mfk_main_cross_state"] == 1)),
        ("buy_top20+mfk_gold+macd_hist>0",
            (df["buy_score"] >= bp80) & (df["mfk_main_cross_state"] == 1) & (df["mfk_main_macd_hist"] > 0)),
        ("理想多",           (df["buy_score"] >= 70) & (df["sell_score"] <= 30)),
        ("理想多+mfk_gold",  (df["buy_score"] >= 70) & (df["sell_score"] <= 30) & (df["mfk_main_cross_state"] == 1)),
        ("理想多+f1>0",      (df["buy_score"] >= 70) & (df["sell_score"] <= 30) & (df["f1_main_in_red"] > 0)),
        ("理想多+mfk+f1",    (df["buy_score"] >= 70) & (df["sell_score"] <= 30) &
                               (df["mfk_main_cross_state"] == 1) & (df["f1_main_in_red"] > 0)),
        ("sell_top20",      df["sell_score"] >= sp80),
        ("sell_top20+mfk_dead", (df["sell_score"] >= sp80) & (df["mfk_main_cross_state"] == -1)),
        ("sell_top20+f2>0", (df["sell_score"] >= sp80) & (df["f2_main_out_green"] > 0)),
        ("sell_top20+mfk_dead+f2", (df["sell_score"] >= sp80) & (df["mfk_main_cross_state"] == -1) &
                                   (df["f2_main_out_green"] > 0)),
    ]:
        g = df[m]
        if len(g) < 100: continue
        res_rows.append({
            "filter": label, "n": len(g),
            "r10_mean": g["r10"].dropna().mean(),
            "r10_winrate": (g["r10"].dropna() > 0).mean() * 100,
            "r20_mean": g["r20"].dropna().mean(),
            "r20_winrate": (g["r20"].dropna() > 0).mean() * 100,
            "dd20_lt_8pct": (g["max_dd_20"].dropna() <= -8).mean() * 100,
            "dd20_lt_15pct": (g["max_dd_20"].dropna() <= -15).mean() * 100,
        })
    res_df = pd.DataFrame(res_rows)
    print(res_df.round(2).to_string())
    res_df.to_csv(OUT_DIR / "high_score_resonance.csv", index=False, encoding="utf-8-sig")

    # ──── 6. SOTA 对比 ────
    sota = {
        "CogAlpha_paper_r10_IC": 0.0591, "CogAlpha_paper_r10_RankICIR": 0.435,
    }
    # 读 train_v4_compare 的 results.csv (如果有)
    v4_path = OUT_DIR / "results.csv"
    sota_rows = []
    if v4_path.exists():
        v4 = pd.read_csv(v4_path)
        for _, r in v4.iterrows():
            sota_rows.append({
                "config": r["config"], "horizon": r["horizon"],
                "IC": r["IC"], "RankIC": r["RankIC"], "RankICIR": r["RankICIR"],
                "vs_paper_IC_lift": (r["IC"] - sota["CogAlpha_paper_r10_IC"]) /
                                       sota["CogAlpha_paper_r10_IC"] * 100 if r["horizon"] == "r10" else np.nan,
                "vs_paper_RankICIR_lift": (r["RankICIR"] - sota["CogAlpha_paper_r10_RankICIR"]) /
                                              sota["CogAlpha_paper_r10_RankICIR"] * 100,
            })
        sota_df = pd.DataFrame(sota_rows)
        sota_df.to_csv(OUT_DIR / "sota_comparison.csv", index=False, encoding="utf-8-sig")
        print(f"\n[{int(time.time()-t0)}s] === 6. vs CogAlpha SOTA ===")
        print(sota_df.round(3).to_string())

    print(f"\n总耗时 {time.time()-t0:.0f}s")
    print(f"输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
