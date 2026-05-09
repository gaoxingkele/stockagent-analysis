#!/usr/bin/env python3
"""分析 8 个 moneyflow_v2 因子在高 buy/sell score 中的作用.

回答: "价量耦合事件因子是真护城河, 在高分多/空中是否发挥作用?"

方法:
  1. 用 r20_v3 (ALL+v2) + 现有 r10/sell_10/sell_20 重新计算 buy_score / sell_score
  2. 把样本分成: top/mid/bot (各 20% buy_score), top/mid/bot sell_score
  3. 在每段计算 f1-f8 均值 + 触发率 (非零比例) + 实际 r10/r20/dd
  4. 输出富集表: top buy 段 f1 均值 vs 全样本均值, 看是否显著高
  5. 同时算每个因子的单因子 IC (vs r20)

输出:
  output/production/v2_factor_analysis.csv  (每因子 × 5 段统计)
  output/production/v2_factor_role.txt      (核心结论)
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
# r20_v3 OOS 锚点 (新分布) — 待数据出来后修正, 暂用 r20_all 的
R20_ANCHOR = (-2.34, 2.50, 6.36)
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
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta.get("industry_map", {})
    df = df.copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), -1)
    )
    return booster.predict(df[feat_cols])


def segment_stats(df, mask, label):
    g = df[mask]
    if len(g) == 0:
        return None
    row = {"segment": label, "n": len(g)}
    # 因子均值 + 触发率 (非零)
    for fc in V2_COLS:
        if fc not in g.columns: continue
        v = g[fc].dropna()
        if len(v) == 0:
            row[f"{fc}_mean"] = np.nan
            row[f"{fc}_p_active"] = np.nan
            continue
        row[f"{fc}_mean"] = v.mean()
        if fc in ("f1_main_in_red", "f2_main_out_green",
                  "f3_main_follow_breakout",
                  "f7_silent_accumulation", "f8_quiet_distribution"):
            # 事件因子: 触发率 = 非零比例
            row[f"{fc}_p_active"] = (v != 0).mean() * 100
        else:
            # 连续因子: p95
            row[f"{fc}_p95"] = v.quantile(0.95)
    # 实际收益
    if "r10" in g.columns:
        row["r10_mean"] = g["r10"].dropna().mean()
        row["r10_winrate"] = (g["r10"].dropna() > 0).mean() * 100
    if "r20" in g.columns:
        row["r20_mean"] = g["r20"].dropna().mean()
        row["r20_winrate"] = (g["r20"].dropna() > 0).mean() * 100
    if "max_dd_20" in g.columns:
        row["dd20_lt_8pct"] = (g["max_dd_20"].dropna() <= -8).mean() * 100
        row["dd20_lt_15pct"] = (g["max_dd_20"].dropna() <= -15).mean() * 100
    return row


def main():
    t0 = time.time()
    print(f"[{int(time.time()-t0)}s] 加载 OOS...")
    df = load_oos()
    print(f"  OOS shape: {df.shape}")

    # 4 模型预测 (r20 用 v3)
    print(f"[{int(time.time()-t0)}s] 预测 r10_all / r20_v3_all / sell_10 / sell_20...")
    df["r10_pred"] = predict_model(df, "r10_all")
    df["r20_pred"] = predict_model(df, "r20_v3_all")
    df["sell_10_prob"] = predict_model(df, "sell_10")
    df["sell_20_prob"] = predict_model(df, "sell_20")

    # 重新算 r20 锚点 (v3 分布可能不同)
    p5, p50, p95 = (df["r20_pred"].quantile(0.05),
                    df["r20_pred"].quantile(0.50),
                    df["r20_pred"].quantile(0.95))
    print(f"[{int(time.time()-t0)}s] r20_v3 OOS 锚点: p5={p5:.3f}, p50={p50:.3f}, p95={p95:.3f}")

    s10 = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df["r20_pred"].values, p5, p50, p95)  # 用 v3 实际锚点
    df["buy_score"] = 0.5 * s10 + 0.5 * s20

    s10s = _map_anchored(df["sell_10_prob"].values, *SELL10_ANCHOR)
    s20s = _map_anchored(df["sell_20_prob"].values, *SELL20_ANCHOR)
    df["sell_score"] = 0.5 * s10s + 0.5 * s20s

    print(f"  buy:  p5={df['buy_score'].quantile(0.05):.1f} p50={df['buy_score'].median():.1f} p95={df['buy_score'].quantile(0.95):.1f}")
    print(f"  sell: p5={df['sell_score'].quantile(0.05):.1f} p50={df['sell_score'].median():.1f} p95={df['sell_score'].quantile(0.95):.1f}")

    # 全样本 baseline
    print(f"\n[{int(time.time()-t0)}s] === 5 段对比 ===")
    rows = [segment_stats(df, df.index >= 0, "ALL")]

    # buy_score 段
    bp80 = df["buy_score"].quantile(0.80)
    bp20 = df["buy_score"].quantile(0.20)
    rows.append(segment_stats(df, df["buy_score"] >= bp80, f"buy_top20 (>={bp80:.0f})"))
    rows.append(segment_stats(df, df["buy_score"] <= bp20, f"buy_bot20 (<={bp20:.0f})"))

    # sell_score 段
    sp80 = df["sell_score"].quantile(0.80)
    sp20 = df["sell_score"].quantile(0.20)
    rows.append(segment_stats(df, df["sell_score"] >= sp80, f"sell_top20 (>={sp80:.0f})"))
    rows.append(segment_stats(df, df["sell_score"] <= sp20, f"sell_bot20 (<={sp20:.0f})"))

    # 4 象限
    rows.append(segment_stats(df, (df["buy_score"] >= 70) & (df["sell_score"] <= 30),
                                "BUY_high+SELL_low (理想多)"))
    rows.append(segment_stats(df, (df["buy_score"] >= 70) & (df["sell_score"] >= 70),
                                "BUY_high+SELL_high (矛盾)"))
    rows.append(segment_stats(df, (df["buy_score"] <= 30) & (df["sell_score"] >= 70),
                                "BUY_low+SELL_high (主流空)"))
    rows.append(segment_stats(df, (df["buy_score"] <= 30) & (df["sell_score"] <= 30),
                                "BUY_low+SELL_low (沉寂)"))

    rows = [r for r in rows if r is not None]
    res_df = pd.DataFrame(rows)

    # 富集表 (相对全样本均值的倍数)
    baseline = res_df.iloc[0]
    enrich_rows = []
    for _, r in res_df.iloc[1:].iterrows():
        er = {"segment": r["segment"], "n": r["n"]}
        for fc in V2_COLS:
            mc = f"{fc}_mean"
            if mc in r and not pd.isna(r[mc]) and not pd.isna(baseline[mc]) and baseline[mc] != 0:
                er[f"{fc}_lift"] = r[mc] / baseline[mc]
            elif mc in r:
                er[f"{fc}_diff"] = r[mc] - baseline[mc] if not pd.isna(r[mc]) else np.nan
        enrich_rows.append(er)
    enrich_df = pd.DataFrame(enrich_rows)

    res_df.to_csv(OUT_DIR / "v2_factor_analysis.csv", index=False, encoding="utf-8-sig")
    enrich_df.to_csv(OUT_DIR / "v2_factor_enrichment.csv", index=False, encoding="utf-8-sig")

    # 单因子 IC vs r20
    print(f"\n[{int(time.time()-t0)}s] === 8 因子单因子 IC (vs r20) ===")
    ic_rows = []
    valid_mask = df["r20"].notna()
    for fc in V2_COLS:
        if fc not in df.columns: continue
        m = valid_mask & df[fc].notna()
        if m.sum() < 1000: continue
        ic = stats.pearsonr(df.loc[m, fc], df.loc[m, "r20"])[0]
        rank_ic = stats.spearmanr(df.loc[m, fc], df.loc[m, "r20"])[0]
        ic_rows.append({"factor": fc, "n": int(m.sum()),
                          "IC": ic, "RankIC": rank_ic,
                          "mean": df.loc[m, fc].mean(),
                          "p_active": (df.loc[m, fc] != 0).mean() * 100})
    ic_df = pd.DataFrame(ic_rows)
    print(ic_df.round(4).to_string())
    ic_df.to_csv(OUT_DIR / "v2_factor_single_ic.csv", index=False, encoding="utf-8-sig")

    # 输出关键摘要
    print(f"\n[{int(time.time()-t0)}s] === 段级 V2 因子均值 (核心) ===")
    show_cols = ["segment", "n"] + [f"{fc}_mean" for fc in V2_COLS] + ["r20_mean","dd20_lt_8pct"]
    show_cols = [c for c in show_cols if c in res_df.columns]
    print(res_df[show_cols].round(4).to_string())

    print(f"\n[{int(time.time()-t0)}s] === 富集表 (vs ALL baseline 倍数) ===")
    print(enrich_df.round(2).to_string())

    # 写结论
    summary = []
    summary.append(f"OOS 验证 V3 (r20_v3_all + 现有 r10/sell), {len(df):,} 样本")
    summary.append(f"V3 r20 锚点: p5={p5:.3f}, p50={p50:.3f}, p95={p95:.3f}")
    summary.append("\n=== 8 因子单因子 IC ===")
    summary.append(ic_df.round(4).to_string())
    summary.append("\n=== 段级因子均值 ===")
    summary.append(res_df[show_cols].round(4).to_string())
    summary.append("\n=== 富集表 ===")
    summary.append(enrich_df.round(2).to_string())
    Path(OUT_DIR / "v2_factor_role.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"\n总耗时 {time.time()-t0:.0f}s, 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
