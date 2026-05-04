#!/usr/bin/env python3
"""V2 组合因子评估: f1-f8 单因子 + 7 个组合, 在 5 段 + 4 象限的激活度.

7 组合:
  A. net_event_money         = f1 - f2          (大跌日吸筹 - 大涨日派发)
  B. silent_balance          = f7 - f8          (静默吸筹 - 静默派发)
  C. event_silent_combined   = (f1-f2) + 0.3*(f7-f8)  (综合)
  D. event_intensity         = |f1| + |f2|      (无方向异常事件强度)
  E. accumulation_ratio      = f1 / (|f1|+|f2|+0.001)  (吸筹占比)
  F. breakout_x_smart        = f3 * sign(f1)    (真假突破)
  G. inst_retail_misalign    = (f6-1) * sign(f1-f2)  (机构散户分歧 × 智能方向)

输出:
  output/production/v2_combos_analysis.csv
  output/production/v2_combos_role.txt
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
OUT_DIR = PROD_DIR
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"

TEST_START = "20250501"
TEST_END   = "20260126"

R10_ANCHOR = (0.72, 0.94, 1.40)
R20_ANCHOR = (-6.92, -1.36, 5.30)  # v3 锚点
SELL10_ANCHOR = (0.05, 0.20, 0.64)
SELL20_ANCHOR = (0.01, 0.07, 0.67)

V2_COLS = ["f1_main_in_red", "f2_main_out_green",
            "f3_main_follow_breakout", "f4_dispersion_price_align",
            "f5_main_acc_velocity_60d", "f6_inst_buy_pressure",
            "f7_silent_accumulation", "f8_quiet_distribution"]


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


def predict_model(df, model_dir_name):
    d = PROD_DIR / model_dir_name
    booster = lgb.Booster(model_file=str(d / "classifier.txt"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta.get("industry_map", {})
    df = df.copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), -1)
    )
    return booster.predict(df[feat_cols])


def add_combos(df):
    """添加 7 个组合因子."""
    f1 = df["f1_main_in_red"].fillna(0)
    f2 = df["f2_main_out_green"].fillna(0)
    f3 = df["f3_main_follow_breakout"].fillna(0)
    f4 = df["f4_dispersion_price_align"].fillna(0)
    f5 = df["f5_main_acc_velocity_60d"].fillna(0.5)
    f6 = df["f6_inst_buy_pressure"].fillna(1.0)
    f7 = df["f7_silent_accumulation"].fillna(0)
    f8 = df["f8_quiet_distribution"].fillna(0)

    df["A_net_event_money"]      = f1 - f2
    df["B_silent_balance"]       = f7 - f8
    df["C_event_silent_combined"]= (f1 - f2) + 0.3 * (f7 - f8)
    df["D_event_intensity"]      = f1.abs() + f2.abs()
    df["E_accumulation_ratio"]   = f1 / (f1.abs() + f2.abs() + 0.001)
    df["F_breakout_x_smart"]     = f3 * np.sign(f1)
    df["G_inst_retail_misalign"] = (f6 - 1) * np.sign(f1 - f2)
    return df


def main():
    t0 = time.time()
    print(f"[{int(time.time()-t0)}s] 加载 OOS...")
    df = load_oos()
    df = add_combos(df)
    print(f"  shape: {df.shape}")

    # 4 模型预测
    print(f"[{int(time.time()-t0)}s] 预测...")
    df["r10_pred"] = predict_model(df, "r10_all")
    df["r20_pred"] = predict_model(df, "r20_v3_all")
    df["sell_10_prob"] = predict_model(df, "sell_10")
    df["sell_20_prob"] = predict_model(df, "sell_20")

    p5, p50, p95 = (df["r20_pred"].quantile(0.05),
                    df["r20_pred"].quantile(0.50),
                    df["r20_pred"].quantile(0.95))
    s10 = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df["r20_pred"].values, p5, p50, p95)
    df["buy_score"] = 0.5 * s10 + 0.5 * s20
    s10s = _map_anchored(df["sell_10_prob"].values, *SELL10_ANCHOR)
    s20s = _map_anchored(df["sell_20_prob"].values, *SELL20_ANCHOR)
    df["sell_score"] = 0.5 * s10s + 0.5 * s20s

    # ── 单因子 RankIC 表 (8 单因子 + 7 组合 = 15 个) ──
    print(f"\n[{int(time.time()-t0)}s] === 15 个因子单因子 IC vs r20 ===")
    factors = V2_COLS + ["A_net_event_money","B_silent_balance",
                          "C_event_silent_combined","D_event_intensity",
                          "E_accumulation_ratio","F_breakout_x_smart",
                          "G_inst_retail_misalign"]
    ic_rows = []
    for fc in factors:
        if fc not in df.columns: continue
        m = df["r20"].notna() & df[fc].notna()
        if m.sum() < 1000: continue
        ic = stats.pearsonr(df.loc[m, fc], df.loc[m, "r20"])[0]
        rank_ic = stats.spearmanr(df.loc[m, fc], df.loc[m, "r20"])[0]
        ic_rows.append({"factor": fc, "n": int(m.sum()),
                          "IC": ic, "RankIC": rank_ic,
                          "mean": df.loc[m, fc].mean(),
                          "std": df.loc[m, fc].std(),
                          "p_active": (df.loc[m, fc].abs() > 1e-6).mean() * 100})
    ic_df = pd.DataFrame(ic_rows).sort_values("RankIC", key=abs, ascending=False)
    print(ic_df.round(4).to_string())
    ic_df.to_csv(OUT_DIR / "v2_combos_ic.csv", index=False, encoding="utf-8-sig")

    # ── 段级激活度 lift ──
    print(f"\n[{int(time.time()-t0)}s] === 段级因子激活 lift ===")

    bp80 = df["buy_score"].quantile(0.80)
    bp20 = df["buy_score"].quantile(0.20)
    sp80 = df["sell_score"].quantile(0.80)
    sp20 = df["sell_score"].quantile(0.20)

    segments = [
        ("ALL", df.index >= 0),
        ("sell_top20", df["sell_score"] >= sp80),
        ("sell_bot20", df["sell_score"] <= sp20),
        ("buy_top20",  df["buy_score"] >= bp80),
        ("buy_bot20",  df["buy_score"] <= bp20),
        ("理想多 BUY≥70+SELL≤30", (df["buy_score"] >= 70) & (df["sell_score"] <= 30)),
        ("矛盾 双高",              (df["buy_score"] >= 70) & (df["sell_score"] >= 70)),
        ("主流空 BUY≤30+SELL≥70", (df["buy_score"] <= 30) & (df["sell_score"] >= 70)),
        ("沉寂 双低",              (df["buy_score"] <= 30) & (df["sell_score"] <= 30)),
    ]

    seg_means = {}
    for label, mask in segments:
        g = df[mask]
        if len(g) == 0:
            seg_means[label] = {"n": 0}
            continue
        row = {"n": len(g)}
        for fc in factors:
            if fc not in g.columns: continue
            v = g[fc].dropna()
            row[fc] = v.mean() if len(v) > 0 else np.nan
        seg_means[label] = row

    # 用 lift = seg_mean / abs(ALL_mean) 来归一化 (ALL_mean 接近 0 时 mean 取代 lift)
    baseline = seg_means["ALL"]
    lift_rows = []
    for label, mask in segments[1:]:
        sm = seg_means[label]
        row = {"segment": label, "n": sm["n"]}
        for fc in factors:
            base_mean = baseline.get(fc, np.nan)
            seg_mean = sm.get(fc, np.nan)
            if pd.isna(seg_mean) or pd.isna(base_mean):
                row[fc] = np.nan
            elif abs(base_mean) > 1e-3:
                row[fc] = seg_mean / base_mean  # 真 lift
            else:
                # baseline 接近 0, 用差值
                row[fc] = seg_mean
        lift_rows.append(row)
    lift_df = pd.DataFrame(lift_rows)
    lift_df.to_csv(OUT_DIR / "v2_combos_segments.csv", index=False, encoding="utf-8-sig")

    # 区分能力 score: |sell_top_lift - sell_bot_lift|
    print(f"\n[{int(time.time()-t0)}s] === 因子对 sell_top / sell_bot 区分能力 ===")
    sell_top_row = lift_df[lift_df["segment"] == "sell_top20"].iloc[0]
    sell_bot_row = lift_df[lift_df["segment"] == "sell_bot20"].iloc[0]
    buy_top_row  = lift_df[lift_df["segment"] == "buy_top20"].iloc[0]
    buy_bot_row  = lift_df[lift_df["segment"] == "buy_bot20"].iloc[0]
    score_rows = []
    for fc in factors:
        if fc not in sell_top_row.index: continue
        diff_sell = abs(sell_top_row[fc] - sell_bot_row[fc])
        diff_buy  = abs(buy_top_row[fc] - buy_bot_row[fc])
        score_rows.append({
            "factor": fc,
            "sell_top": sell_top_row[fc], "sell_bot": sell_bot_row[fc],
            "sell_diff": diff_sell,
            "buy_top": buy_top_row[fc], "buy_bot": buy_bot_row[fc],
            "buy_diff": diff_buy,
        })
    score_df = pd.DataFrame(score_rows).sort_values("sell_diff", ascending=False)
    print(score_df.round(3).to_string())
    score_df.to_csv(OUT_DIR / "v2_combos_separation.csv", index=False, encoding="utf-8-sig")

    # 输出文本摘要
    summary = []
    summary.append("=" * 80)
    summary.append("V2 单因子 + 组合因子全量评估")
    summary.append(f"OOS: {len(df):,} 样本, r20_v3 锚点 p5={p5:.2f}/p50={p50:.2f}/p95={p95:.2f}")
    summary.append("=" * 80)
    summary.append("\n=== 1. 单因子 IC (按 |RankIC| 排序) ===")
    summary.append(ic_df.round(4).to_string())
    summary.append("\n=== 2. 段级激活度 (lift = seg_mean / ALL_mean, 接近0时为绝对均值) ===")
    summary.append(lift_df.round(3).to_string())
    summary.append("\n=== 3. 因子区分能力 (越大越好) ===")
    summary.append(score_df.round(3).to_string())
    Path(OUT_DIR / "v2_combos_role.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"\n总耗时 {time.time()-t0:.0f}s, 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
