"""V12 双轨 v8: m_excl × cap_in × cap_cross 联合网格.

v4 单独优化 m_excl=0.10, v5 单独优化 cap_in=0.20, v6 单独优化 cap_cross=0.20.
但三者可能有交互效应 (例如 m_excl 高时 cap 不需太严).

网格: m_excl ∈ [0.05, 0.10, 0.15, 0.20] × cap_in ∈ [0.20, 0.25] × cap_cross ∈ [0.20, 0.25, 0.30]
= 4 × 2 × 3 = 24 配置
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

OUT = ROOT / "output" / "backtest_v12_dual_v8"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20251001"
OOS_END = "20260331"
COST_BPS = 35.0 / 10000
P_BUY_STATIC = 0.05
PYR_VELOCITY_QUANTILE = 0.35
TRACK_A_PCT = 0.70
TRACK_B_PCT = 0.20
MAX_A_STOCKS = 8
MAX_B_STOCKS = 15

GRID_M_EXCL = [0.05, 0.10, 0.15, 0.20]
GRID_CAP_IN = [0.20, 0.25]
GRID_CAP_CROSS = [0.20, 0.25, 0.30]


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


def apply_cap(pool, alloc, cap, prior=None):
    if pool.empty or cap >= 1.0: return pool
    per_stock = alloc / len(pool)
    pool = pool.sort_values("r5_long_rank")
    industry_alloc = dict(prior or {})
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


def build_dual(daily, ind_mom, m_excl, cap_in, cap_cross):
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= m_excl)
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
        a_pool = apply_cap(a_pool, TRACK_A_PCT, cap_in)
        a_ind = industry_alloc_dict(a_pool, TRACK_A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B_STOCKS).copy()
        b_pool = apply_cap(b_pool, TRACK_B_PCT, cap_in)
        b_pool = apply_cap(b_pool, TRACK_B_PCT, cap_cross, prior=a_ind)
        for track_tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                                "industry": row.get("industry", ""),
                                "r20": float(row["r20"]) if pd.notna(row.get("r20")) else np.nan,
                                "alloc_pct": per_stock})
    return pd.DataFrame(rows)


def compute_metrics(hold, daily):
    hold = hold.dropna(subset=["r20"]).copy()
    hold["r20"] = hold["r20"].clip(-30, 30)
    hold["weighted"] = hold["alloc_pct"] * hold["r20"]
    total = TRACK_A_PCT + TRACK_B_PCT
    pnl = hold.groupby("entry_date").agg(
        gross=("weighted", "sum"), n=("ts_code", "count")).reset_index()
    pnl["net"] = pnl["gross"] - total * COST_BPS * 200
    mkt = daily.groupby("trade_date")["r20"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt"]
    pnl = pnl.merge(mkt, on="entry_date", how="left")
    pnl["alpha"] = pnl["net"] - pnl["mkt"] * total
    pnl["month"] = pnl["entry_date"].str[:6]
    monthly = pnl.groupby("month")["alpha"].mean()
    by_di = hold.groupby(["entry_date", "industry"])["alloc_pct"].sum().reset_index()
    max_by_d = by_di.groupby("entry_date")["alloc_pct"].max()
    return {
        "alpha": pnl["alpha"].mean(),
        "sharpe": pnl["alpha"].mean() / (pnl["alpha"].std() + 1e-9) * np.sqrt(12),
        "bad_months": (monthly < -1.0).sum(),
        "worst_month": monthly.min(),
        "max_ind_max": float(max_by_d.max()),
        "max_ind_p95": float(max_by_d.quantile(0.95)),
    }


def main():
    t0 = time.time()
    print(f"\n=== V12 v8: m_excl × cap_in × cap_cross 联合网格 (24 配置) ===\n", flush=True)
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

    ind_mom = compute_ind_mom(daily)

    rows = []
    print(f"\n  {'m_excl':6s} {'cap_in':6s} {'cap_X':6s} {'α':9s} {'Sharpe':8s} {'灾':3s} {'最差':7s} {'p95行业':8s}",
           flush=True)
    for m_excl in GRID_M_EXCL:
        for cap_in in GRID_CAP_IN:
            for cap_cross in GRID_CAP_CROSS:
                if cap_cross < cap_in:   # 跨轨 cap 应 ≥ 单轨 cap (否则没意义)
                    continue
                hold = build_dual(daily, ind_mom, m_excl, cap_in, cap_cross)
                mt = compute_metrics(hold, daily)
                print(f"  {m_excl:.2f}   {cap_in:.2f}   {cap_cross:.2f}   "
                       f"{mt['alpha']:+.3f}pp  {mt['sharpe']:+.2f}    {mt['bad_months']}    "
                       f"{mt['worst_month']:+.2f}   {mt['max_ind_p95']:.2f}", flush=True)
                rows.append({
                    "m_excl": m_excl, "cap_in": cap_in, "cap_cross": cap_cross,
                    "alpha": mt["alpha"], "sharpe": mt["sharpe"],
                    "bad_months": mt["bad_months"], "worst_month": mt["worst_month"],
                    "max_ind_p95": mt["max_ind_p95"], "max_ind_max": mt["max_ind_max"],
                })

    df = pd.DataFrame(rows).sort_values(["sharpe", "alpha"], ascending=False)
    df.to_csv(OUT / "grid_results.csv", index=False)
    best = df.iloc[0]
    print(f"\n--- 最佳 ---")
    print(f"  m_excl={best['m_excl']}, cap_in={best['cap_in']}, cap_cross={best['cap_cross']}")
    print(f"  α {best['alpha']:+.3f}pp, Sharpe {best['sharpe']:+.2f}, "
           f"灾难月 {int(best['bad_months'])}, 最差 {best['worst_month']:+.2f}pp")
    print(f"  p95 行业 alloc: {best['max_ind_p95']:.2f}")

    md = [f"# V12 v8 联合网格 (24 配置)\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            f"v6 基准: m_excl=0.10, cap_in=0.20, cap_cross=0.20 → α +1.14pp Sharpe 1.33\n\n",
            "## Top 10\n\n| m_excl | cap_in | cap_cross | α | Sharpe | 灾难月 | 最差月 | p95行业 |\n|---|---|---|---|---|---|---|---|\n"]
    for _, r in df.head(10).iterrows():
        md.append(f"| {r['m_excl']:.2f} | {r['cap_in']:.2f} | {r['cap_cross']:.2f} | "
                   f"{r['alpha']:+.3f}pp | {r['sharpe']:+.2f} | "
                   f"{int(r['bad_months'])} | {r['worst_month']:+.2f}pp | "
                   f"{r['max_ind_p95']:.2f} |\n")
    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
