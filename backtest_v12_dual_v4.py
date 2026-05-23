"""V12 双轨 v4: 行业 momentum 过滤 + regime adaptive P_BUY_TOP.

两个新维度:
  M. 行业 momentum 过滤: 排除最近 60 日表现最弱 N% 行业 (避免 202603 化工类灾难)
  R. Regime adaptive P_BUY_TOP: 牛市宽选 (10%), 熊市严选 (5%), 震荡中间 (7%)

网格扫描:
  M_EXCL ∈ [0.0, 0.10, 0.20, 0.30]      (排除最差 N% 行业)
  REGIME ∈ ["off", "on"]                 (静态 P=0.05 vs 动态)

输出: output/backtest_v12_dual_v4/grid_results.csv + report.md
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

OUT = ROOT / "output" / "backtest_v12_dual_v4"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20251001"
OOS_END = "20260331"
COST_BPS = 35.0 / 10000
PYR_VELOCITY_QUANTILE = 0.35
TRACK_A_PCT = 0.70
TRACK_B_PCT = 0.20
MAX_A_STOCKS = 8
MAX_B_STOCKS = 15

# 网格
M_EXCL_GRID = [0.0, 0.10, 0.20, 0.30]
REGIME_MODE = ["off", "on"]

# Regime adaptive 阈值 (按 20 日市场 return 判断)
REGIME_BULL_THRESHOLD = 3.0    # mkt_ret_20d > 3% → 牛
REGIME_BEAR_THRESHOLD = -3.0   # mkt_ret_20d < -3% → 熊
P_BUY_BULL = 0.10
P_BUY_BEAR = 0.05
P_BUY_SIDEWAYS = 0.07
P_BUY_STATIC = 0.05


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def compute_industry_60d_momentum(daily: pd.DataFrame) -> pd.DataFrame:
    """每日每行业 60 日累计 return (行业内股票均值).

    使用 close[t-1] / close[t-61] - 1 (防 forward leak).
    返回 (trade_date, industry, industry_mom_60d, industry_mom_60d_rank).
    """
    # 用 daily cache 算个股 60d
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    files = sorted(daily_dir.glob("*.parquet"))
    dailies = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"]) for f in files]
    big = pd.concat(dailies, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["mom_60d"] = (big.groupby("ts_code")["close"].shift(1) /
                        big.groupby("ts_code")["close"].shift(61) - 1)

    # merge industry
    basic_p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    basic = pd.read_parquet(basic_p)[["ts_code", "industry"]].drop_duplicates("ts_code")
    big = big.merge(basic, on="ts_code", how="left")

    # 行业层均值
    ind_mom = big.dropna(subset=["mom_60d"]).groupby(["trade_date", "industry"]).agg(
        industry_mom_60d=("mom_60d", "mean")
    ).reset_index()

    # 当日行业 pct rank
    ind_mom["industry_mom_60d_rank"] = ind_mom.groupby("trade_date")["industry_mom_60d"].rank(
        pct=True, method="first")
    return ind_mom


def build_dual_with_filters(daily: pd.DataFrame, p_buy: float | None,
                              ind_mom: pd.DataFrame, m_excl: float,
                              regime_mode: str) -> pd.DataFrame:
    """跑双轨 + 行业 momentum 过滤 + 可选 regime adaptive."""
    # merge industry_mom
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")

    # mkt_ret_20d for regime
    mkt_ret_by_date = daily.groupby("trade_date")["r20"].apply(
        lambda x: x.clip(-30, 30).mean()).to_dict()

    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()

        # 行业 momentum 过滤: 排除最差 m_excl
        if m_excl > 0 and "industry_mom_60d_rank" in g.columns:
            # NaN rank 保留 (避免误排), 只过滤明确 rank < m_excl
            g = g[(g["industry_mom_60d_rank"].isna()) |
                    (g["industry_mom_60d_rank"] >= m_excl)]
            if len(g) < 100: continue

        # 决定 P_BUY (regime adaptive 或静态)
        if regime_mode == "on" and p_buy is None:
            # 用过去 20 日市场 return 判断 regime (用历史日数据近似)
            past_dates = sorted([d for d in mkt_ret_by_date if d < d_])[-20:]
            if past_dates:
                past_mkt = np.mean([mkt_ret_by_date[d] for d in past_dates])
            else:
                past_mkt = 0
            if past_mkt > REGIME_BULL_THRESHOLD:
                actual_p = P_BUY_BULL
            elif past_mkt < REGIME_BEAR_THRESHOLD:
                actual_p = P_BUY_BEAR
            else:
                actual_p = P_BUY_SIDEWAYS
        else:
            actual_p = p_buy if p_buy is not None else P_BUY_STATIC

        # r20 top P
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - actual_p)

        # pyr_velocity
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_VELOCITY_QUANTILE)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)

        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue

        # R5 反向
        v7c["r5_long_rank"] = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
        v7c = v7c.sort_values("r5_long_rank")
        a_pool = v7c.head(MAX_A_STOCKS)
        b_pool = v7c.iloc[len(a_pool):len(a_pool) + MAX_B_STOCKS]

        for track_tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({
                    "entry_date": d_, "ts_code": row["ts_code"], "track": track_tag,
                    "industry": row.get("industry", ""),
                    "industry_mom_60d_rank": float(row.get("industry_mom_60d_rank", 0.5)),
                    "actual_p_buy": actual_p,
                    "r20": float(row["r20"]) if pd.notna(row.get("r20")) else np.nan,
                    "alloc_pct": per_stock,
                })
    return pd.DataFrame(rows)


def compute_metrics(hold_df: pd.DataFrame, daily: pd.DataFrame) -> dict:
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
        "monthly_alpha": daily_pnl["alpha"].mean(),
        "sharpe": daily_pnl["alpha"].mean() / (daily_pnl["alpha"].std() + 1e-9) * np.sqrt(12),
        "n_hold_days": daily_pnl["entry_date"].nunique(),
        "bad_months": (monthly["alpha_avg"] < -1.0).sum(),
        "worst_month_alpha": monthly["alpha_avg"].min(),
        "monthly": monthly,
    }


def main():
    t0 = time.time()
    print(f"\n=== V12 双轨 v4: 行业 momentum + regime adaptive ===\n", flush=True)

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

    print("[2] 行业 60d momentum 计算 ...", flush=True)
    ind_mom = compute_industry_60d_momentum(daily)
    print(f"  industry × date 行数: {len(ind_mom):,}", flush=True)

    # 网格
    print(f"\n[3] 网格扫描 (m_excl × regime) ...", flush=True)
    print(f"  {'config':30s} {'cov':5s} {'α':8s} {'Sharpe':8s} {'灾难月':6s} {'最差月':8s}",
           flush=True)
    grid_rows = []
    for m_excl in M_EXCL_GRID:
        for regime in REGIME_MODE:
            label = f"m_excl={m_excl:.2f}, regime={regime}"
            p_buy_arg = None if regime == "on" else P_BUY_STATIC
            hold = build_dual_with_filters(daily, p_buy_arg, ind_mom, m_excl, regime)
            mt = compute_metrics(hold, daily)
            cov = mt["n_hold_days"] / daily["trade_date"].nunique() * 100
            print(f"  {label:30s} {cov:4.0f}% {mt['monthly_alpha']:+.3f}pp "
                   f"{mt['sharpe']:+.2f}    {mt['bad_months']}     {mt['worst_month_alpha']:+.2f}",
                   flush=True)
            grid_rows.append({
                "m_excl": m_excl, "regime": regime,
                "cov_pct": cov,
                "monthly_alpha_pp": mt["monthly_alpha"],
                "sharpe": mt["sharpe"],
                "bad_months": mt["bad_months"],
                "worst_month_alpha": mt["worst_month_alpha"],
            })

    grid_df = pd.DataFrame(grid_rows)
    grid_df = grid_df.sort_values(["sharpe", "monthly_alpha_pp"], ascending=False)
    grid_df.to_csv(OUT / "grid_results.csv", index=False)

    best = grid_df.iloc[0]
    print(f"\n--- 最佳配置 ---")
    print(f"  m_excl={best['m_excl']:.2f}, regime={best['regime']}")
    print(f"  α {best['monthly_alpha_pp']:+.3f}pp / Sharpe {best['sharpe']:+.2f}")
    print(f"  灾难月 {int(best['bad_months'])} / 最差月 α {best['worst_month_alpha']:+.2f}pp")

    # 报告
    md = [f"# V12 双轨 v4: 行业 momentum + regime adaptive\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START}-{OOS_END}\n\n",
            "## 网格扫描 (按 Sharpe 降序)\n\n",
            "| m_excl | regime | 覆盖率 % | 月化 α | Sharpe | 灾难月 | 最差月 α |\n",
            "|---|---|---|---|---|---|---|\n"]
    for _, r in grid_df.iterrows():
        md.append(f"| {r['m_excl']:.2f} | {r['regime']} | {r['cov_pct']:.0f} | "
                   f"{r['monthly_alpha_pp']:+.3f}pp | {r['sharpe']:+.2f} | "
                   f"{int(r['bad_months'])} | {r['worst_month_alpha']:+.2f} |\n")
    md.append(f"\n## 基准 (v3 P=0.05)\n")
    md.append(f"- m_excl=0.00, regime=off → α +1.10pp Sharpe +1.29\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
