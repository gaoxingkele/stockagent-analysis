"""V12 双轨 v7: 持仓周期网格 (5/10/20 日).

关键问题: V12 默认 r20 (20 日持仓), 但 5/8-5/15 实战 r5 (5 日) α +0.87pp.
更短持仓 = 高换手 + 高成本; 更长持仓 = 低换手 + 低成本.
年化 α = per_period_α × (252/N) - 成本 × (252/N)

网格:
  持仓周期 N ∈ [5, 10, 20]
  对每个 N: 算 per-period α, 年化 α, 扣成本年化, Sharpe

要 r5 label, r10 label, r20 label.
r5/r20 已有, r10 需要从 daily cache 算 (close[t+10] / open[t+1] - 1).

输出: output/backtest_v12_dual_v7/grid_results.csv + report.md
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

OUT = ROOT / "output" / "backtest_v12_dual_v7"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20251001"
OOS_END = "20260331"
COST_BPS = 35.0 / 10000   # 0.35% 单边
P_BUY_STATIC = 0.05
PYR_VELOCITY_QUANTILE = 0.35
M_EXCL = 0.10
INDUSTRY_CAP_IN = 0.20
INDUSTRY_CAP_CROSS = 0.20
TRACK_A_PCT = 0.70
TRACK_B_PCT = 0.20
MAX_A_STOCKS = 8
MAX_B_STOCKS = 15

HOLD_PERIODS = [5, 10, 20]   # N 日持仓


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def compute_ind_mom(daily):
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    files = sorted(daily_dir.glob("*.parquet"))
    dailies = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"]) for f in files]
    big = pd.concat(dailies, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["mom_60d"] = (big.groupby("ts_code")["close"].shift(1) /
                        big.groupby("ts_code")["close"].shift(61) - 1)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "industry"]].drop_duplicates("ts_code")
    big = big.merge(basic, on="ts_code", how="left")
    ind_mom = big.dropna(subset=["mom_60d"]).groupby(["trade_date", "industry"]).agg(
        industry_mom_60d=("mom_60d", "mean")
    ).reset_index()
    ind_mom["industry_mom_60d_rank"] = ind_mom.groupby("trade_date")["industry_mom_60d"].rank(
        pct=True, method="first")
    return ind_mom


def compute_forward_labels(periods):
    """从 daily cache 算多个 forward labels: r5, r10, r20 (close[t+N] / open[t+1] - 1)."""
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    files = sorted(daily_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"])
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    out = big[["ts_code", "trade_date"]].copy()
    for N in periods:
        big[f"close_{N}d"] = big.groupby("ts_code")["close"].shift(-N)
        out[f"r{N}_label"] = (big[f"close_{N}d"] / big["next_open"] - 1) * 100
    return out


def apply_cap(pool, alloc, cap, prior_alloc=None):
    if pool.empty or cap >= 1.0: return pool
    per_stock = alloc / len(pool)
    pool = pool.sort_values("r5_long_rank")
    industry_alloc = dict(prior_alloc or {})
    keep = []
    for idx, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        new_alloc = industry_alloc.get(ind, 0) + per_stock
        if new_alloc > cap + 1e-9: continue
        industry_alloc[ind] = new_alloc
        keep.append(idx)
    return pool.loc[keep]


def industry_alloc_dict(pool, alloc):
    if pool.empty: return {}
    per_stock = alloc / len(pool)
    counts = {}
    for _, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        counts[ind] = counts.get(ind, 0) + 1
    return {ind: cnt * per_stock for ind, cnt in counts.items()}


def build_dual(daily, ind_mom):
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 100: continue
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY_STATIC)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_VELOCITY_QUANTILE)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue
        v7c["r5_long_rank"] = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
        v7c = v7c.sort_values("r5_long_rank")
        a_pool = v7c.head(MAX_A_STOCKS).copy()
        a_pool = apply_cap(a_pool, TRACK_A_PCT, INDUSTRY_CAP_IN)
        a_ind = industry_alloc_dict(a_pool, TRACK_A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B_STOCKS).copy()
        b_pool = apply_cap(b_pool, TRACK_B_PCT, INDUSTRY_CAP_IN)
        b_pool = apply_cap(b_pool, TRACK_B_PCT, INDUSTRY_CAP_CROSS, prior_alloc=a_ind)
        for track_tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({
                    "entry_date": d_, "ts_code": row["ts_code"],
                    "industry": row.get("industry", ""),
                    "alloc_pct": per_stock,
                })
    return pd.DataFrame(rows)


def metrics_per_period(hold_df, label_df, N: int):
    """每 N 日持仓回测."""
    label_col = f"r{N}_label"
    h = hold_df.merge(label_df[["ts_code", "trade_date", label_col]],
                        left_on=["ts_code", "entry_date"],
                        right_on=["ts_code", "trade_date"], how="left")
    h = h.dropna(subset=[label_col])
    h[label_col] = h[label_col].clip(-30, 30)
    h["weighted"] = h["alloc_pct"] * h[label_col]
    total_alloc = TRACK_A_PCT + TRACK_B_PCT
    daily_pnl = h.groupby("entry_date").agg(
        gross=("weighted", "sum"),
        n=("ts_code", "count"),
    ).reset_index()
    # 成本: 单边 0.35%, 一进一出 = 0.7%. per period:
    daily_pnl["net"] = daily_pnl["gross"] - total_alloc * COST_BPS * 200
    return daily_pnl


def main():
    t0 = time.time()
    print(f"\n=== V12 双轨 v7: 持仓周期网格 (5/10/20 日) ===\n", flush=True)

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

    print("[2] 行业 momentum + 持仓清单...", flush=True)
    ind_mom = compute_ind_mom(daily)
    hold = build_dual(daily, ind_mom)
    print(f"  持仓清单: {len(hold):,} 条, {hold['entry_date'].nunique()} 日", flush=True)

    print("[3] 计算 r5/r10/r20 forward labels (从 daily cache)...", flush=True)
    label_df = compute_forward_labels(HOLD_PERIODS)
    label_df["trade_date"] = label_df["trade_date"].astype(str)

    # 各持仓周期算指标
    print(f"\n[4] 各持仓周期回测\n", flush=True)
    # 市场基准 (每个 N 日的市场平均 r_N)
    daily_with_label = daily.merge(label_df, on=["ts_code", "trade_date"], how="left")

    print(f"  {'N':3s} {'per-期 α':10s} {'年化 α':9s} {'Sharpe(月)':10s} {'灾难月':6s} {'最差月':8s}",
           flush=True)
    grid_rows = []
    for N in HOLD_PERIODS:
        label_col = f"r{N}_label"
        daily_pnl = metrics_per_period(hold, label_df, N)
        # 市场基准
        mkt = daily_with_label.groupby("trade_date")[label_col].apply(
            lambda x: x.clip(-30, 30).mean()).reset_index()
        mkt.columns = ["entry_date", "mkt_r"]
        daily_pnl = daily_pnl.merge(mkt, on="entry_date", how="left")
        total_alloc = TRACK_A_PCT + TRACK_B_PCT
        daily_pnl["alpha"] = daily_pnl["net"] - daily_pnl["mkt_r"] * total_alloc

        # 年化 α (扣成本)
        n_periods_per_year = 252 / N
        per_period_alpha = daily_pnl["alpha"].mean()
        annual_alpha = per_period_alpha * n_periods_per_year   # 算术近似

        # Sharpe (月化)
        sharpe_monthly = daily_pnl["alpha"].mean() / (daily_pnl["alpha"].std() + 1e-9) * np.sqrt(12)

        # 月度
        daily_pnl["month"] = daily_pnl["entry_date"].str[:6]
        monthly = daily_pnl.groupby("month")["alpha"].mean()
        bad_months = (monthly < -1.0).sum()
        worst_month = monthly.min()

        print(f"  {N:2d}日  {per_period_alpha:+.3f}pp  {annual_alpha:+.2f}pp  "
               f"{sharpe_monthly:+.2f}     {bad_months}     {worst_month:+.2f}", flush=True)

        grid_rows.append({
            "N_days": N,
            "per_period_alpha_pp": per_period_alpha,
            "annual_alpha_pp": annual_alpha,
            "sharpe_monthly": sharpe_monthly,
            "bad_months": bad_months,
            "worst_month_alpha": worst_month,
            "n_hold_days": daily_pnl["entry_date"].nunique(),
        })

    grid_df = pd.DataFrame(grid_rows).sort_values(["annual_alpha_pp", "sharpe_monthly"],
                                                     ascending=False)
    grid_df.to_csv(OUT / "grid_results.csv", index=False)

    best = grid_df.iloc[0]
    print(f"\n--- 最佳持仓周期 ---")
    print(f"  N = {int(best['N_days'])} 日")
    print(f"  per-期 α: {best['per_period_alpha_pp']:+.3f}pp")
    print(f"  年化 α: {best['annual_alpha_pp']:+.2f}pp")
    print(f"  月 Sharpe: {best['sharpe_monthly']:+.2f}")
    print(f"  灾难月 {int(best['bad_months'])} / 最差月 {best['worst_month_alpha']:+.2f}pp")

    md = [f"# V12 双轨 v7: 持仓周期网格\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            f"## 网格 (按年化 α 降序)\n\n",
            "| N 日 | per-期 α | 年化 α | 月 Sharpe | 灾难月 | 最差月 α |\n",
            "|---|---|---|---|---|---|\n"]
    for _, r in grid_df.iterrows():
        md.append(f"| {int(r['N_days'])} | {r['per_period_alpha_pp']:+.3f}pp | "
                   f"{r['annual_alpha_pp']:+.2f}pp | {r['sharpe_monthly']:+.2f} | "
                   f"{int(r['bad_months'])} | {r['worst_month_alpha']:+.2f}pp |\n")
    md.append(f"\n## 注\n\n")
    md.append(f"- 年化 α = per-期 α × (252/N), 算术近似\n")
    md.append(f"- 成本已扣 (单边 0.35%, 每周期 0.7%)\n")
    md.append(f"- Sharpe 用月化 (每月 ≈ 20 个交易日)\n")
    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
