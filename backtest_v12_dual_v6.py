"""V12 双轨 v6: 跨轨累计行业 cap 长 OOS 验证.

vs v5 (单轨内 cap=0.20):
  - 新增 cross_track_industry_cap=0.30 (A+B 合计单行业 ≤ 30%)
  - 实施: B 轨 cap 后, 再用 A 的行业 alloc 作 prior, 跨轨 cap

输出: output/backtest_v12_dual_v6/report.md
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

OUT = ROOT / "output" / "backtest_v12_dual_v6"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20251001"
OOS_END = "20260331"
COST_BPS = 35.0 / 10000
P_BUY_STATIC = 0.05
PYR_VELOCITY_QUANTILE = 0.35
M_EXCL = 0.10
TRACK_A_PCT = 0.70
TRACK_B_PCT = 0.20
MAX_A_STOCKS = 8
MAX_B_STOCKS = 15


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def compute_industry_60d_mom(daily: pd.DataFrame) -> pd.DataFrame:
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


def apply_cap(pool: pd.DataFrame, alloc: float, cap: float,
                prior_alloc: dict = None) -> pd.DataFrame:
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


def industry_alloc_dict(pool: pd.DataFrame, alloc: float) -> dict:
    if pool.empty: return {}
    per_stock = alloc / len(pool)
    counts: dict = {}
    for _, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        counts[ind] = counts.get(ind, 0) + 1
    return {ind: cnt * per_stock for ind, cnt in counts.items()}


def build_dual(daily: pd.DataFrame, ind_mom: pd.DataFrame,
                 cap_in_track: float, cap_cross_track: float) -> pd.DataFrame:
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()
        # m_excl
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 100: continue
        # r20 top 5%
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
        a_pool = apply_cap(a_pool, TRACK_A_PCT, cap_in_track)
        a_ind_alloc = industry_alloc_dict(a_pool, TRACK_A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B_STOCKS).copy()
        b_pool = apply_cap(b_pool, TRACK_B_PCT, cap_in_track)
        if cap_cross_track is not None:
            b_pool = apply_cap(b_pool, TRACK_B_PCT, cap_cross_track,
                                 prior_alloc=a_ind_alloc)
        for track_tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({
                    "entry_date": d_, "ts_code": row["ts_code"], "track": track_tag,
                    "industry": row.get("industry", ""),
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

    # max industry alloc 跨日
    by_di = hold_df.groupby(["entry_date", "industry"])["alloc_pct"].sum().reset_index()
    max_by_d = by_di.groupby("entry_date")["alloc_pct"].max()

    return {
        "monthly_alpha": daily_pnl["alpha"].mean(),
        "sharpe": daily_pnl["alpha"].mean() / (daily_pnl["alpha"].std() + 1e-9) * np.sqrt(12),
        "bad_months": (monthly["alpha_avg"] < -1.0).sum(),
        "worst_month_alpha": monthly["alpha_avg"].min(),
        "max_ind_alloc_avg": float(max_by_d.mean()),
        "max_ind_alloc_max": float(max_by_d.max()),
        "max_ind_alloc_p95": float(max_by_d.quantile(0.95)),
        "monthly": monthly,
    }


def main():
    t0 = time.time()
    print(f"\n=== V12 双轨 v6: 跨轨累计行业 cap ===\n", flush=True)

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

    ind_mom = compute_industry_60d_mom(daily)

    # 网格 (cross_track ∈ [None=off, 0.30, 0.25, 0.20])
    print(f"\n  {'cap_in':6s} {'cap_cross':9s} {'α':9s} {'Sharpe':8s} {'灾':3s} {'最差':7s} "
           f"{'max_ind avg/p95/max':22s}", flush=True)
    grid_rows = []
    for cap_cross in [None, 0.30, 0.25, 0.20]:
        hold = build_dual(daily, ind_mom, 0.20, cap_cross)
        mt = compute_metrics(hold, daily)
        cross_str = "None" if cap_cross is None else f"{cap_cross:.2f}"
        print(f"  {0.20:.2f}   {cross_str:9s}  {mt['monthly_alpha']:+.3f}pp "
               f"{mt['sharpe']:+.2f}    {mt['bad_months']}    {mt['worst_month_alpha']:+.2f}   "
               f"{mt['max_ind_alloc_avg']:.2f}/{mt['max_ind_alloc_p95']:.2f}/{mt['max_ind_alloc_max']:.2f}",
               flush=True)
        grid_rows.append({
            "cap_in_track": 0.20, "cap_cross_track": cap_cross,
            "monthly_alpha_pp": mt["monthly_alpha"],
            "sharpe": mt["sharpe"],
            "bad_months": mt["bad_months"],
            "worst_month_alpha": mt["worst_month_alpha"],
            "max_ind_alloc_avg": mt["max_ind_alloc_avg"],
            "max_ind_alloc_p95": mt["max_ind_alloc_p95"],
            "max_ind_alloc_max": mt["max_ind_alloc_max"],
        })

    grid_df = pd.DataFrame(grid_rows).sort_values(["sharpe", "monthly_alpha_pp"], ascending=False)
    grid_df.to_csv(OUT / "grid_results.csv", index=False)
    best = grid_df.iloc[0]
    print(f"\n--- 最佳 ---")
    print(f"  cap_in={best['cap_in_track']}, cap_cross={best['cap_cross_track']}")
    print(f"  α {best['monthly_alpha_pp']:+.3f}pp / Sharpe {best['sharpe']:+.2f}")
    print(f"  灾难月 {int(best['bad_months'])} / 最差 {best['worst_month_alpha']:+.2f}pp")
    print(f"  max行业 avg={best['max_ind_alloc_avg']:.2f}, p95={best['max_ind_alloc_p95']:.2f}, "
           f"max={best['max_ind_alloc_max']:.2f}")

    # report
    md = [f"# V12 双轨 v6: 跨轨累计行业 cap 长 OOS\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            f"基准 (cap_cross=None): v5 配置, α +1.08pp Sharpe 1.26 (max行业 max 86%)\n\n",
            "## 网格\n\n",
            "| cap_in | cap_cross | α | Sharpe | 灾难月 | 最差月 | max行业 avg/p95/max |\n",
            "|---|---|---|---|---|---|---|\n"]
    for _, r in grid_df.iterrows():
        cross = "None" if pd.isna(r["cap_cross_track"]) else f"{r['cap_cross_track']:.2f}"
        md.append(f"| {r['cap_in_track']} | {cross} | {r['monthly_alpha_pp']:+.3f}pp | "
                   f"{r['sharpe']:+.2f} | {int(r['bad_months'])} | "
                   f"{r['worst_month_alpha']:+.2f} | "
                   f"{r['max_ind_alloc_avg']:.2f}/{r['max_ind_alloc_p95']:.2f}/{r['max_ind_alloc_max']:.2f} |\n")
    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
