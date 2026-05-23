"""V12 双轨架构长 OOS 回测 (用 _nost 模型, 142 日严格 OOS).

目的:
  1. 验证 _nost 模型切换后, V12 双轨架构真实月化 alpha
  2. 找灾难月 (大幅负 alpha 的月)
  3. 对比 OLD 旧版含 ST 模型 vs NOST 排 ST 模型的双轨

策略复用 src.stockagent_analysis.v12_scoring + v12_dual_track 的核心逻辑, 但跳过
zombie/policy 等慢路径, 一次性批量推理.

持仓周期: 20 日 (= r20 label = 后续 20 个交易日 close / next_open - 1)
进/出: 每日 EOD 推荐 → 次日 09:30 入场 → 持有 20 个交易日 → 平仓

输出: output/backtest_v12_dual/report.md + monthly_pnl.csv + curves/
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

OUT = ROOT / "output" / "backtest_v12_dual"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "curves").mkdir(exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20251001"
OOS_END = "20260331"
HOLD_DAYS = 20
COST_BPS = 35.0 / 10000   # 0.35% 单边 (买+卖 = 0.7%)

# V7c 5 铁律 + R5 反向参数
BUY_SCORE_LO, BUY_SCORE_HI = 70, 85
PYR_VELOCITY_QUANTILE = 0.35
F1_F2_THRESHOLD = 0.005
R5_REVERSE_BOTTOM_A = 0.15   # 轨 A 严过滤
R5_REVERSE_BOTTOM_B = 0.35   # 轨 B 宽过滤
TRACK_A_PCT = 0.70           # 70% 资金
TRACK_B_PCT = 0.20           # 20% 资金
CASH_PCT = 0.10              # 10% 现金
MAX_A_STOCKS = 15
MAX_B_STOCKS = 25

# 锚点 (与 v12_scoring.py 一致, _nost 版本)
R5_ANCHOR = (-0.21, 0.82, 1.80)
R10_ANCHOR = (-3.55, 0.56, 6.91)
R20_ANCHOR = (-7.10, 2.45, 14.08)


def map_anchored(v: np.ndarray, p5: float, p50: float, p95: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, 50.0)
    out = np.where(v <= p5, 0, out)
    out = np.where(v >= p95, 100, out)
    mask_lo = (v > p5) & (v <= p50)
    out = np.where(mask_lo, (v - p5) / (p50 - p5 + 1e-9) * 50, out)
    mask_hi = (v > p50) & (v < p95)
    out = np.where(mask_hi, 50 + (v - p50) / (p95 - p50 + 1e-9) * 50, out)
    return out


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def main():
    t0 = time.time()
    print(f"\n=== V12 双轨长 OOS 回测 ({OOS_START}-{OOS_END}) ===\n", flush=True)

    # 1. 加载日线 + r20 label (load_window 已 ST 排除)
    print("[1] 加载日线...", flush=True)
    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    print(f"  loaded: {len(daily):,} 行, {daily['trade_date'].nunique()} 日", flush=True)

    # merge long_return (虽然 _nost regression 模型不用, 但 P1 R5 反向用 r5_v17_long_nost)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")

    # 2. 推理 r5/r10/r20 nost (3 个用于 v7c, 另 1 个 r5_long_nost 用于 R5 反向)
    print("[2] 推理 r5/r10/r20_v16_all_nost + r5_v17_long_nost ...", flush=True)
    for name in ["r5_v17_all_nost", "r10_v16_all_nost", "r20_v16_all_nost",
                   "r5_v17_long_nost"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily[f"pred_{name}"] = b.predict(X)
        print(f"  {name}: pred μ={daily[f'pred_{name}'].mean():+.3f}", flush=True)

    # 3. 锚定 0-100 评分 + buy_score
    print("\n[3] 锚定 0-100 评分 + V7c 5 铁律 ...", flush=True)
    daily["buy_r5_score"] = map_anchored(daily["pred_r5_v17_all_nost"].values, *R5_ANCHOR)
    daily["buy_r10_score"] = map_anchored(daily["pred_r10_v16_all_nost"].values, *R10_ANCHOR)
    daily["buy_r20_score"] = map_anchored(daily["pred_r20_v16_all_nost"].values, *R20_ANCHOR)
    daily["buy_score"] = 0.5 * daily["buy_r10_score"] + 0.5 * daily["buy_r20_score"]

    # 5 铁律 (简化版, 跳过 zombie 因为 long_oos 内已经 ST 排除, zombie 影响小)
    # 用 pyr_velocity_20_60 + f1_neg1 + f2_pos1 (factor 里已有)
    # 注: 这些列由 v7_extras feature 提供, load_window 已 merge
    has_pyr = "pyr_velocity_20_60" in daily.columns
    has_f1f2 = "f1_neg1" in daily.columns and "f2_pos1" in daily.columns
    print(f"  pyr_velocity_20_60: {has_pyr}, f1/f2: {has_f1f2}", flush=True)

    # 每日 5 铁律
    print("[4] 按日构建 V7c 主推荐 + R5 反向过滤 + 双轨 ...", flush=True)
    days = sorted(daily["trade_date"].unique())
    holdings_log = []  # 每日推荐持仓

    for d_ in days:
        g = daily[daily["trade_date"] == d_].copy()
        if len(g) < 1000: continue

        # 5 铁律
        m_buy = (g["buy_score"] >= BUY_SCORE_LO) & (g["buy_score"] <= BUY_SCORE_HI)
        if has_pyr:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_VELOCITY_QUANTILE)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        if has_f1f2:
            m_f12 = (g["f1_neg1"].abs() < F1_F2_THRESHOLD) & (g["f2_pos1"].abs() < F1_F2_THRESHOLD)
        else:
            m_f12 = pd.Series(True, index=g.index)

        g["v7c_recommend"] = m_buy & m_pyr & m_f12

        # R5 反向 rank_in_pool
        v7c = g[g["v7c_recommend"]].copy()
        if len(v7c) == 0: continue
        v7c["r5_long_rank"] = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
        v7c = v7c.sort_values("r5_long_rank")

        # 轨 A: Bot 15%
        a_pool = v7c[v7c["r5_long_rank"] < R5_REVERSE_BOTTOM_A].head(MAX_A_STOCKS)
        # 轨 B: Bot 15-35%
        b_pool = v7c[(v7c["r5_long_rank"] >= R5_REVERSE_BOTTOM_A) &
                       (v7c["r5_long_rank"] < R5_REVERSE_BOTTOM_B)].head(MAX_B_STOCKS)

        for track_tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock_pct = alloc / len(pool)
            for _, row in pool.iterrows():
                holdings_log.append({
                    "entry_date": d_,
                    "ts_code": row["ts_code"],
                    "track": track_tag,
                    "buy_score": float(row["buy_score"]),
                    "r5_long_rank": float(row["r5_long_rank"]),
                    "r20": float(row["r20"]) if pd.notna(row.get("r20")) else np.nan,
                    "alloc_pct": per_stock_pct,
                })

    hold_df = pd.DataFrame(holdings_log)
    hold_df.to_csv(OUT / "holdings_log.csv", index=False)
    print(f"  总持仓记录: {len(hold_df):,}, 日数 {hold_df['entry_date'].nunique()}", flush=True)
    print(f"  轨 A: {(hold_df['track']=='A').sum():,}", flush=True)
    print(f"  轨 B: {(hold_df['track']=='B').sum():,}", flush=True)

    # 4. 每日组合收益 = 加权 r20 (减成本)
    # r20 表示该股从 entry_date 起 20 个交易日的累计 return (close[t+20]/next_open[t+1] - 1)
    hold_df = hold_df.dropna(subset=["r20"]).copy()
    hold_df["r20"] = hold_df["r20"].clip(-30, 30)  # cap 极端值
    hold_df["weighted_r20"] = hold_df["alloc_pct"] * hold_df["r20"]

    # 按 entry_date 聚合 (该日双轨组合的 20 日总收益)
    daily_pnl = hold_df.groupby("entry_date").agg(
        port_r20_gross_pct=("weighted_r20", "sum"),     # gross 已加权
        n_stocks=("ts_code", "count"),
        track_a_alloc=("alloc_pct", lambda x: x[hold_df.loc[x.index, "track"] == "A"].sum()),
        track_b_alloc=("alloc_pct", lambda x: x[hold_df.loc[x.index, "track"] == "B"].sum()),
    ).reset_index()
    # 注意 alloc 总和: 0.7 (A) + 0.2 (B) = 0.9, 0.1 cash 不参与
    # weighted_r20 已经包含 alloc, sum 后是组合 20 日净 return (含 cash 拖累)
    # 减成本: 一买一卖 = 0.7% per stock, 但所有 alloc 都买卖, 总成本 = (0.9 * 0.7%) = 0.63%
    total_alloc = TRACK_A_PCT + TRACK_B_PCT
    daily_pnl["port_r20_net_pct"] = daily_pnl["port_r20_gross_pct"] - total_alloc * COST_BPS * 200

    # 市场基准: 当日全市场 r20 均值 (排 ST 已经在 load_window 做)
    mkt = daily.groupby("trade_date")["r20"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt_r20_pct"]
    daily_pnl = daily_pnl.merge(mkt, on="entry_date", how="left")
    daily_pnl["alpha_pct"] = daily_pnl["port_r20_net_pct"] - daily_pnl["mkt_r20_pct"] * total_alloc

    # 月份 + 月汇总
    daily_pnl["month"] = daily_pnl["entry_date"].str[:6]
    monthly = daily_pnl.groupby("month").agg(
        n_days=("entry_date", "count"),
        avg_n_stocks=("n_stocks", "mean"),
        port_r20_net_avg=("port_r20_net_pct", "mean"),
        mkt_r20_avg=("mkt_r20_pct", "mean"),
        alpha_avg=("alpha_pct", "mean"),
        alpha_std=("alpha_pct", "std"),
    ).reset_index()
    monthly["sharpe"] = monthly["alpha_avg"] / (monthly["alpha_std"] + 1e-9) * np.sqrt(20)
    monthly.to_csv(OUT / "monthly_pnl.csv", index=False)

    # 整体指标
    avg_net = daily_pnl["port_r20_net_pct"].mean()
    avg_mkt_scaled = daily_pnl["mkt_r20_pct"].mean() * total_alloc
    monthly_net = avg_net   # 20 日 ≈ 1 月
    monthly_alpha = (avg_net - avg_mkt_scaled)
    sharpe = daily_pnl["alpha_pct"].mean() / (daily_pnl["alpha_pct"].std() + 1e-9) * np.sqrt(12)

    # 报告
    md = [f"# V12 双轨架构长 OOS 回测 (_nost 模型)\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START}-{OOS_END}\n",
            f"持仓周期: {HOLD_DAYS} 日 (r20 label)\n",
            f"双轨分配: A {TRACK_A_PCT*100:.0f}% (Bot {R5_REVERSE_BOTTOM_A*100:.0f}%) / "
            f"B {TRACK_B_PCT*100:.0f}% (Bot {R5_REVERSE_BOTTOM_A*100:.0f}-{R5_REVERSE_BOTTOM_B*100:.0f}%) / "
            f"现金 {CASH_PCT*100:.0f}%\n",
            f"成本: 单边 {COST_BPS*100:.2f}%\n\n",
            f"## 整体指标 (日均)\n\n",
            f"- 组合 20 日净收益均值: **{avg_net:+.3f}%/期** ({monthly_net:+.3f}%/月化)\n",
            f"- 市场 20 日 (按 alloc 缩放): {avg_mkt_scaled:+.3f}%/期\n",
            f"- **月化 α: {monthly_alpha:+.3f}pp**\n",
            f"- Sharpe (月化): {sharpe:.2f}\n\n",
            f"## 月度明细 (找灾难月)\n\n",
            "| 月份 | 日数 | 平均股数 | 组合净 % | 市场 % | α % | Sharpe |\n",
            "|---|---|---|---|---|---|---|\n"]
    for _, r in monthly.iterrows():
        md.append(f"| {r['month']} | {int(r['n_days'])} | {r['avg_n_stocks']:.1f} | "
                   f"{r['port_r20_net_avg']:+.3f} | {r['mkt_r20_avg']:+.3f} | "
                   f"{r['alpha_avg']:+.3f} | {r['sharpe']:+.2f} |\n")

    # 灾难月 (α < -1pp)
    bad = monthly[monthly["alpha_avg"] < -1.0]
    md.append(f"\n## 灾难月 (α < -1pp)\n\n")
    if bad.empty:
        md.append("无灾难月 (所有月 α > -1pp)\n")
    else:
        for _, r in bad.iterrows():
            md.append(f"- **{r['month']}**: α {r['alpha_avg']:+.3f}pp, "
                       f"组合净 {r['port_r20_net_avg']:+.3f}%, 市场 {r['mkt_r20_avg']:+.3f}%\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"\n--- 整体 ---")
    print(f"  组合 20 日净均: {avg_net:+.3f}%/期 ({monthly_net:+.3f}%/月)")
    print(f"  市场 (按 alloc 缩放): {avg_mkt_scaled:+.3f}%/期")
    print(f"  月化 α: {monthly_alpha:+.3f}pp")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"\n--- 月度 ---")
    for _, r in monthly.iterrows():
        flag = " [BAD]" if r["alpha_avg"] < -1.0 else ""
        print(f"  {r['month']}: n={int(r['n_days']):2d} 股={r['avg_n_stocks']:5.1f} "
               f"组合={r['port_r20_net_avg']:+.2f}% 市场={r['mkt_r20_avg']:+.2f}% "
               f"α={r['alpha_avg']:+.2f}pp{flag}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
