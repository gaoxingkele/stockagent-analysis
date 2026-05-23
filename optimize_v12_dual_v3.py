"""V12 双轨 v3 优化: A 网格扫描 P_BUY_TOP + E 灾难月 202603 持仓诊断.

网格: P_BUY_TOP ∈ [0.05, 0.10, 0.15, 0.20, 0.25]
对每个 P, 跑 v2 回测, 输出月度 α + Sharpe

输出:
  output/backtest_v12_dual_v3/grid_results.csv  各 P 的整体指标
  output/backtest_v12_dual_v3/monthly_by_P.csv  各 P × 各月 α 表
  output/backtest_v12_dual_v3/diag_202603.csv   灾难月持仓详情 (P=0.10)
  output/backtest_v12_dual_v3/report.md
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

OUT = ROOT / "output" / "backtest_v12_dual_v3"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"

OOS_START = "20251001"
OOS_END = "20260331"
COST_BPS = 35.0 / 10000
P_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
PYR_VELOCITY_QUANTILE = 0.35
R5_REVERSE_BOTTOM_A = 0.15
R5_REVERSE_BOTTOM_B = 0.35
TRACK_A_PCT = 0.70
TRACK_B_PCT = 0.20
MAX_A_STOCKS = 15
MAX_B_STOCKS = 25


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def build_dual(daily: pd.DataFrame, p_buy: float) -> pd.DataFrame:
    """跑一次 v2 双轨, 返回 holdings_log df."""
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - p_buy)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_VELOCITY_QUANTILE)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue
        v7c["r5_long_rank"] = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
        a_pool = v7c[v7c["r5_long_rank"] < R5_REVERSE_BOTTOM_A].nsmallest(MAX_A_STOCKS, "r5_long_rank")
        b_pool = v7c[(v7c["r5_long_rank"] >= R5_REVERSE_BOTTOM_A) &
                       (v7c["r5_long_rank"] < R5_REVERSE_BOTTOM_B)].nsmallest(MAX_B_STOCKS, "r5_long_rank")
        for track_tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({
                    "entry_date": d_, "ts_code": row["ts_code"], "track": track_tag,
                    "industry": row.get("industry", ""),
                    "r5_long_rank": float(row["r5_long_rank"]),
                    "r20": float(row["r20"]) if pd.notna(row.get("r20")) else np.nan,
                    "alloc_pct": per_stock,
                })
    return pd.DataFrame(rows)


def compute_metrics(hold_df: pd.DataFrame, daily: pd.DataFrame, p_buy: float) -> dict:
    """汇总月度 + 整体指标."""
    hold_df = hold_df.dropna(subset=["r20"]).copy()
    hold_df["r20"] = hold_df["r20"].clip(-30, 30)
    hold_df["weighted_r20"] = hold_df["alloc_pct"] * hold_df["r20"]
    total_alloc = TRACK_A_PCT + TRACK_B_PCT

    daily_pnl = hold_df.groupby("entry_date").agg(
        port_r20_gross=("weighted_r20", "sum"),
        n_stocks=("ts_code", "count"),
    ).reset_index()
    daily_pnl["port_net"] = daily_pnl["port_r20_gross"] - total_alloc * COST_BPS * 200

    mkt = daily.groupby("trade_date")["r20"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt_r20"]
    daily_pnl = daily_pnl.merge(mkt, on="entry_date", how="left")
    daily_pnl["alpha"] = daily_pnl["port_net"] - daily_pnl["mkt_r20"] * total_alloc
    daily_pnl["month"] = daily_pnl["entry_date"].str[:6]

    monthly = daily_pnl.groupby("month").agg(
        n_days=("entry_date", "count"),
        port_avg=("port_net", "mean"),
        mkt_avg=("mkt_r20", "mean"),
        alpha_avg=("alpha", "mean"),
        alpha_std=("alpha", "std"),
    ).reset_index()
    monthly["sharpe"] = monthly["alpha_avg"] / (monthly["alpha_std"] + 1e-9) * np.sqrt(20)

    return {
        "p_buy": p_buy,
        "monthly_alpha_avg": daily_pnl["alpha"].mean(),
        "monthly_port_net": daily_pnl["port_net"].mean(),
        "sharpe_overall": daily_pnl["alpha"].mean() / (daily_pnl["alpha"].std() + 1e-9) * np.sqrt(12),
        "n_hold_days": daily_pnl["entry_date"].nunique(),
        "n_total_days": daily["trade_date"].nunique(),
        "monthly": monthly,
        "daily_pnl": daily_pnl,
        "holdings": hold_df,
    }


def main():
    t0 = time.time()
    print(f"\n=== V12 双轨 v3: P_BUY_TOP 网格 + 灾难月诊断 ===\n", flush=True)

    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")

    for name in ["r20_v16_long_nost", "r5_v17_long_nost"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily[f"pred_{name}"] = b.predict(X)
    print(f"  推理完成 {time.time()-t0:.0f}s", flush=True)

    # 网格扫描
    print(f"\n## A. P_BUY_TOP 网格扫描\n")
    print(f"  {'P_BUY':6s} {'cov %':6s} {'月化α':8s} {'Sharpe':8s} {'灾难月数':10s}", flush=True)
    all_metrics = []
    all_monthly = []
    for p in P_GRID:
        hold = build_dual(daily, p)
        m = compute_metrics(hold, daily, p)
        cov = m["n_hold_days"] / m["n_total_days"] * 100
        bad_months = (m["monthly"]["alpha_avg"] < -1.0).sum()
        print(f"  {p:.2f}   {cov:5.0f}  {m['monthly_alpha_avg']:+.3f}pp  "
               f"{m['sharpe_overall']:+.2f}    {bad_months}", flush=True)
        all_metrics.append({
            "p_buy": p, "cov_pct": cov,
            "monthly_alpha_pp": m["monthly_alpha_avg"],
            "sharpe": m["sharpe_overall"],
            "bad_months": bad_months,
        })
        m_monthly = m["monthly"].copy()
        m_monthly["p_buy"] = p
        all_monthly.append(m_monthly)

    grid_df = pd.DataFrame(all_metrics)
    grid_df.to_csv(OUT / "grid_results.csv", index=False)
    all_monthly_df = pd.concat(all_monthly, ignore_index=True)
    all_monthly_df.to_csv(OUT / "monthly_by_P.csv", index=False)

    # 选最佳 P
    best_p = grid_df.sort_values(["sharpe", "monthly_alpha_pp"], ascending=False).iloc[0]
    print(f"\n  → 最佳 P_BUY_TOP = {best_p['p_buy']:.2f} "
           f"(α {best_p['monthly_alpha_pp']:+.3f}pp, Sharpe {best_p['sharpe']:+.2f})", flush=True)

    # E. 灾难月 202603 诊断 (用 P=0.10)
    print(f"\n## E. 202603 灾难月持仓诊断 (P_BUY=0.10)\n")
    hold = build_dual(daily, 0.10)
    bad_hold = hold[hold["entry_date"].str.startswith("202603")].copy()
    bad_hold = bad_hold.dropna(subset=["r20"])
    print(f"  202603 持仓: {len(bad_hold)} 条记录", flush=True)

    # merge name
    if BASIC_P.exists():
        basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
        bad_hold = bad_hold.merge(basic, on="ts_code", how="left")

    # 行业分布
    ind_perf = bad_hold.groupby("industry").agg(
        n_holdings=("ts_code", "count"),
        r20_mean=("r20", "mean"),
        alloc_total=("alloc_pct", "sum"),
    ).reset_index().sort_values("r20_mean")
    print(f"\n  行业 vs r20 平均 (排序):", flush=True)
    for _, r in ind_perf.head(10).iterrows():
        print(f"    {r['industry']:20s} n={r['n_holdings']:3d} r20={r['r20_mean']:+6.2f}% "
               f"alloc={r['alloc_total']:.3f}", flush=True)
    print(f"  ... (worst 行业, 拖累灾难月)", flush=True)
    print(f"\n  前 5 最好行业:", flush=True)
    for _, r in ind_perf.tail(5).iterrows():
        print(f"    {r['industry']:20s} n={r['n_holdings']:3d} r20={r['r20_mean']:+6.2f}% "
               f"alloc={r['alloc_total']:.3f}", flush=True)

    bad_hold.to_csv(OUT / "diag_202603_holdings.csv", index=False)
    ind_perf.to_csv(OUT / "diag_202603_by_industry.csv", index=False)

    # 报告
    md = [f"# V12 双轨 v3 优化: 网格扫描 + 灾难月诊断\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START}-{OOS_END} (~142 日)\n\n",
            f"## A. P_BUY_TOP 网格扫描\n\n",
            "| P_BUY | 覆盖率 % | 月化 α | Sharpe | 灾难月数 |\n",
            "|---|---|---|---|---|\n"]
    for _, r in grid_df.iterrows():
        md.append(f"| {r['p_buy']:.2f} | {r['cov_pct']:.0f} | {r['monthly_alpha_pp']:+.3f}pp | "
                   f"{r['sharpe']:+.2f} | {r['bad_months']} |\n")
    md.append(f"\n**最佳**: P_BUY={best_p['p_buy']:.2f} → α {best_p['monthly_alpha_pp']:+.3f}pp / "
               f"Sharpe {best_p['sharpe']:+.2f}\n\n")
    md.append(f"## E. 202603 灾难月持仓 (P=0.10)\n\n")
    md.append(f"持仓: {len(bad_hold)} 条记录\n\n")
    md.append("### 各行业平均 r20 (worst 5)\n\n")
    md.append("| 行业 | 持仓数 | 平均 r20 % | 总仓位 |\n|---|---|---|---|\n")
    for _, r in ind_perf.head(5).iterrows():
        md.append(f"| {r['industry']} | {r['n_holdings']} | {r['r20_mean']:+.2f} | "
                   f"{r['alloc_total']:.3f} |\n")
    md.append("\n### 各行业平均 r20 (best 5)\n\n")
    md.append("| 行业 | 持仓数 | 平均 r20 % | 总仓位 |\n|---|---|---|---|\n")
    for _, r in ind_perf.tail(5).iterrows():
        md.append(f"| {r['industry']} | {r['n_holdings']} | {r['r20_mean']:+.2f} | "
                   f"{r['alloc_total']:.3f} |\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
