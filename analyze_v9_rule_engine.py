#!/usr/bin/env python3
"""V9 规则引擎: V7c 推荐池后置过滤 + 仓位调整.

⚠️ 重要说明:
  AKShare 数据是今天 (2026-05-05) 的快照, 历史不可回填.
  V7c OOS 推荐池基于 2025-05 → 2026-01-26 的数据.
  时间错位, 此评估**仅做规则方向性验证**, 不是严谨回测.
  严谨回测需更新 factor_lab 到今天 + 用 AKShare 历史时序覆盖近 30 天 (V9 后续 B 阶段).

V9 规则引擎 (基于 8 AKShare 指标):
  基础仓位 = 3% (V7c 推荐池默认)

  加分 (向 5%):
    + ef_score >= 70:        +1%
    + xq_follow_pct >= 0.7:  +1%
    + ef_rank_chg > 500:     +1%
    + ef_inst_pct >= 0.5:    +1%

  减分 (向 1%):
    - ef_score < 50:                  -1%
    - xq_follow_pct < 0.3 and xq_tweet_pct < 0.3: -1%
    - ef_main_cost_dev > 5:           -1% (追高)
    - ef_main_cost_dev < -5:          -1% (套牢)
    - ef_rank_chg < -500:             -1%

  极端 SKIP:
    - ef_score < 35:        SKIP
    - ef_main_cost_dev > 10: SKIP (严重追高)

输出: output/v9/rule_eval.csv
"""
from __future__ import annotations
import json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
V9_DIR = ROOT / "output" / "v9_data"
OUT_DIR = ROOT / "output" / "v9"
OUT_DIR.mkdir(exist_ok=True)
TEST_START = "20250501"
TEST_END   = "20260126"

R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)
STOCK_COST = 0.5
ETF_COST = 0.2


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


def v9_rule(row):
    """V9 规则引擎: 输入 V7c 推荐 + AKShare 8 指标, 输出仓位.
    返回: (decision, position_pct, reasons)
    """
    score = row.get("ef_score", 60)
    focus = row.get("ef_focus", 70)
    inst_pct = row.get("ef_inst_pct", 0.003)  # 注意: AKShare 实际 0-1 但取了原始
    main_cost_dev = row.get("ef_main_cost_dev", 0)
    rank_chg = row.get("ef_rank_chg", 0)
    xq_follow = row.get("xq_follow_pct", 0.5)
    xq_tweet = row.get("xq_tweet_pct", 0.5)

    if pd.isna(score) or pd.isna(focus):
        return "NO_DATA", 3, ["AKShare 数据缺失, 默认 3%"]

    reasons = []

    # 极端 SKIP
    if score < 35:
        return "SKIP", 0, [f"综合分极低 {score:.0f} < 35"]
    if main_cost_dev > 10:
        return "SKIP", 0, [f"严重追高: 偏离主力成本 +{main_cost_dev:.1f}%"]

    base = 3
    delta = 0

    # 加分项
    if score >= 70:
        delta += 1; reasons.append(f"高综合分 {score:.0f}")
    if xq_follow >= 0.7:
        delta += 1; reasons.append(f"雪球高关注 p{xq_follow*100:.0f}")
    if rank_chg > 500:
        delta += 1; reasons.append(f"排名快升 +{rank_chg:.0f}")
    if not pd.isna(inst_pct) and inst_pct >= 0.5:  # 0-1 (修正后)
        delta += 1; reasons.append(f"机构高参与 {inst_pct*100:.0f}%")

    # 减分项
    if score < 50:
        delta -= 1; reasons.append(f"低综合分 {score:.0f}")
    if xq_follow < 0.3 and xq_tweet < 0.3:
        delta -= 1; reasons.append(f"雪球冷门 follow={xq_follow*100:.0f}%/tweet={xq_tweet*100:.0f}%")
    if main_cost_dev > 5:
        delta -= 1; reasons.append(f"中度追高 +{main_cost_dev:.1f}%")
    if main_cost_dev < -5:
        delta -= 1; reasons.append(f"套牢盘 {main_cost_dev:.1f}%")
    if rank_chg < -500:
        delta -= 1; reasons.append(f"排名快降 {rank_chg:.0f}")

    pos = max(1, min(5, base + delta))
    return "BUY", pos, reasons


def main():
    t0 = time.time()
    print(f"[{int(time.time()-t0)}s] 加载 OOS + V7c 推荐...", flush=True)
    df = load_oos()
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

    p35_v20 = df["pyr_velocity_20_60"].quantile(0.35)
    pool_mask = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30) &
                  (df["pyr_velocity_20_60"] < p35_v20) &
                  (df["f1_neg1"].abs() < 0.005) &
                  (df["f2_pos1"].abs() < 0.005))
    pool = df[pool_mask].dropna(subset=["r20"]).reset_index(drop=True)
    print(f"  V7c 推荐池: n={len(pool):,}, r20 mean={pool['r20'].mean():.2f}%", flush=True)

    # 加载 AKShare 8 指标 (今天的快照)
    today = datetime.now().strftime("%Y%m%d")
    ak_path = V9_DIR / f"akshare_snapshot_{today}.parquet"
    if not ak_path.exists():
        # 找最新的
        files = sorted(V9_DIR.glob("akshare_snapshot_*.parquet"))
        if not files:
            print("⚠️ 无 AKShare 数据, 先跑 extract_v9_akshare.py")
            return
        ak_path = files[-1]
    print(f"[{int(time.time()-t0)}s] 加载 AKShare 快照: {ak_path.name}", flush=True)
    ak = pd.read_parquet(ak_path)
    # 修正 ef_inst_pct (实际 0-1, 之前误除 100)
    if ak["ef_inst_pct"].max() < 0.05:
        ak["ef_inst_pct"] = ak["ef_inst_pct"] * 100  # 还原
    print(f"  AKShare 5184 股, ef_score mean={ak['ef_score'].mean():.1f} "
          f"ef_inst_pct mean={ak['ef_inst_pct'].mean():.3f}", flush=True)

    # 合并 (按 ts_code, 时间错位但作架构验证)
    print(f"[{int(time.time()-t0)}s] V9 规则应用 (时间错位仅作架构验证)...", flush=True)
    pool = pool.merge(ak, on="ts_code", how="left")
    print(f"  推荐池中匹配到 AKShare 数据: {pool['ef_score'].notna().sum()}/{len(pool)}", flush=True)

    # 应用规则
    decisions = []
    positions = []
    for _, row in pool.iterrows():
        d, p, _ = v9_rule(row)
        decisions.append(d)
        positions.append(p)
    pool["v9_decision"] = decisions
    pool["v9_position"] = positions

    # ── 评估 ──
    print(f"\n[{int(time.time()-t0)}s] === V9 决策分布 ===", flush=True)
    print(pool["v9_decision"].value_counts().to_string(), flush=True)

    print(f"\n=== V9 仓位段 r20 对比 ===", flush=True)
    for pos in sorted(pool["v9_position"].unique()):
        sub = pool[pool["v9_position"] == pos]
        r20 = sub["r20"].dropna()
        if len(r20) >= 10:
            print(f"  仓位 {pos}%: n={len(sub):4d} r20 mean={r20.mean():+.2f}% win={(r20>0).mean()*100:.1f}%",
                  flush=True)

    print(f"\n=== V9 BUY vs SKIP 子集对比 ===", flush=True)
    buy = pool[pool["v9_decision"] == "BUY"]
    skip = pool[pool["v9_decision"] == "SKIP"]
    no_data = pool[pool["v9_decision"] == "NO_DATA"]
    print(f"  BUY (n={len(buy)}): r20 mean={buy['r20'].mean():+.2f}% win={(buy['r20']>0).mean()*100:.1f}%",
          flush=True)
    print(f"  SKIP (n={len(skip)}): r20 mean={skip['r20'].mean():+.2f}% win={(skip['r20']>0).mean()*100:.1f}%",
          flush=True)
    print(f"  NO_DATA (n={len(no_data)}): r20 mean={no_data['r20'].mean():+.2f}%",
          flush=True)
    if len(buy) > 10 and len(skip) > 10:
        t, p = stats.ttest_ind(buy["r20"].dropna(), skip["r20"].dropna())
        print(f"  t-test: diff={buy['r20'].mean()-skip['r20'].mean():+.2f}pp, t={t:.2f}, p={p:.4f}",
              flush=True)

    print(f"\n=== V9 4%/5% (强买) vs 1%/2% (弱买) 对比 ===", flush=True)
    strong = pool[pool["v9_position"] >= 4]
    weak = pool[(pool["v9_position"] >= 1) & (pool["v9_position"] <= 2)]
    if len(strong) > 10 and len(weak) > 10:
        print(f"  强买 (4-5% pos, n={len(strong)}): r20={strong['r20'].mean():+.2f}% win={(strong['r20']>0).mean()*100:.1f}%",
              flush=True)
        print(f"  弱买 (1-2% pos, n={len(weak)}): r20={weak['r20'].mean():+.2f}% win={(weak['r20']>0).mean()*100:.1f}%",
              flush=True)
        t, p = stats.ttest_ind(strong["r20"].dropna(), weak["r20"].dropna())
        print(f"  t-test: diff={strong['r20'].mean()-weak['r20'].mean():+.2f}pp, p={p:.4f}",
              flush=True)

    pool[["ts_code","trade_date","buy_score","sell_score","r20",
          "ef_score","ef_focus","ef_inst_pct","ef_main_cost_dev","ef_rank_chg",
          "xq_follow_pct","xq_tweet_pct","v9_decision","v9_position"]].to_csv(
        OUT_DIR / "rule_eval.csv", index=False, encoding="utf-8-sig")

    print(f"\n总耗时 {time.time()-t0:.0f}s, 输出: {OUT_DIR}/rule_eval.csv", flush=True)


if __name__ == "__main__":
    main()
