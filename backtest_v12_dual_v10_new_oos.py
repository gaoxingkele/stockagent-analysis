"""V12 v10 在新 OOS (20260201-20260515) 上验证 v6 vs v10a vs v10b.

关键测试: pump_score 是真 alpha 还是 v3-v8 类反向过拟合?

如果 v10a 在新 OOS 期仍优于 v6 → 真 alpha
如果 v10a 在新 OOS 期反而劣 v6 → 又是 OOS 过拟合
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

OUT = ROOT / "output" / "backtest_v12_dual_v10_newoos"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20260201"
OOS_END = "20260515"
DATA_START = "20251001"   # 早 load 用于 ind_mom

COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def compute_ind_mom(daily):
    ddir = ROOT / "output/tushare_cache/daily"
    files = sorted(ddir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["mom_60d"] = (big.groupby("ts_code")["close"].shift(1) /
                        big.groupby("ts_code")["close"].shift(61) - 1)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "industry"]].drop_duplicates("ts_code")
    big = big.merge(basic, on="ts_code", how="left")
    ind = big.dropna(subset=["mom_60d"]).groupby(["trade_date", "industry"]).agg(
        industry_mom_60d=("mom_60d", "mean")).reset_index()
    ind["industry_mom_60d_rank"] = ind.groupby("trade_date")["industry_mom_60d"].rank(
        pct=True, method="first")
    return ind


def compute_r20_label(daily):
    ddir = ROOT / "output/tushare_cache/daily"
    files = sorted(ddir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"])
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    big["close_20d"] = big.groupby("ts_code")["close"].shift(-20)
    big["r20_fresh"] = (big["close_20d"] / big["next_open"] - 1) * 100
    return big[["ts_code", "trade_date", "r20_fresh"]]


def apply_cap(pool, alloc, cap, prior=None, sort_col="composite"):
    if pool.empty or cap >= 1.0: return pool
    per = alloc / len(pool)
    pool = pool.sort_values(sort_col, ascending=False)
    ia = dict(prior or {})
    keep = []
    for idx, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        new = ia.get(ind, 0) + per
        if new > cap + 1e-9: continue
        ia[ind] = new
        keep.append(idx)
    return pool.loc[keep]


def industry_alloc(pool, alloc):
    if pool.empty: return {}
    per = alloc / len(pool)
    counts = {}
    for _, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        counts[ind] = counts.get(ind, 0) + 1
    return {k: v * per for k, v in counts.items()}


def build_dual(daily, ind_mom, mode):
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if d_ < OOS_START or d_ > OOS_END: continue
        if len(g) < 500: continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 100: continue
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_Q)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue

        v7c["r5_long_rank_in"] = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
        v7c["pump_score_norm"] = v7c["pred_pump"]
        v7c["r20_rank_in"] = v7c["pred_r20_v16_long_nost"].rank(pct=True, method="first")

        if mode == "v6":
            v7c["composite_A"] = 1 - v7c["r5_long_rank_in"]
            v7c["composite_B"] = 1 - v7c["r5_long_rank_in"]
        elif mode == "v10a":
            v7c["composite_A"] = v7c["pump_score_norm"]
            v7c["composite_B"] = v7c["pump_score_norm"]
        elif mode == "v10b":
            v7c["composite_A"] = 0.7 * v7c["pump_score_norm"] + 0.3 * (1 - v7c["r5_long_rank_in"])
            v7c["composite_B"] = 0.5 * v7c["pump_score_norm"] + 0.5 * v7c["r20_rank_in"]

        v7c_a = v7c.sort_values("composite_A", ascending=False)
        a_pool = v7c_a.head(MAX_A).copy()
        a_pool["composite"] = a_pool["composite_A"]
        a_pool = apply_cap(a_pool, A_PCT, CAP_IN)
        a_ind = industry_alloc(a_pool, A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])].copy()
        b_cand["composite"] = b_cand["composite_B"]
        b_pool = b_cand.sort_values("composite_B", ascending=False).head(MAX_B).copy()
        b_pool = apply_cap(b_pool, B_PCT, CAP_IN)
        b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind)

        for tag, pool, alloc in [("A", a_pool, A_PCT), ("B", b_pool, B_PCT)]:
            if len(pool) == 0: continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                                "track": tag,
                                "industry": row.get("industry", ""),
                                "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                                "alloc_pct": per})
    return pd.DataFrame(rows)


def compute_metrics(hold, daily):
    h = hold.dropna(subset=["r20_fresh"]).copy()
    h["r20_fresh"] = h["r20_fresh"].clip(-30, 30)
    h["w"] = h["alloc_pct"] * h["r20_fresh"]
    total = A_PCT + B_PCT
    pnl = h.groupby("entry_date").agg(gross=("w", "sum"), n=("ts_code", "count")).reset_index()
    pnl["net"] = pnl["gross"] - total * COST_BPS * 200
    mkt = daily[(daily["trade_date"] >= OOS_START) & (daily["trade_date"] <= OOS_END)].groupby(
        "trade_date")["r20_fresh"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt"]
    pnl = pnl.merge(mkt, on="entry_date", how="left")
    pnl["alpha"] = pnl["net"] - pnl["mkt"] * total
    pnl["month"] = pnl["entry_date"].str[:6]
    monthly = pnl.groupby("month")["alpha"].mean()
    return {
        "alpha": pnl["alpha"].mean(),
        "sharpe": pnl["alpha"].mean() / (pnl["alpha"].std() + 1e-9) * np.sqrt(12),
        "bad_months": (monthly < -1.0).sum(),
        "worst_month": monthly.min(),
        "n_hold_days": pnl["entry_date"].nunique(),
        "monthly": monthly,
    }


def main():
    t0 = time.time()
    print(f"\n=== V12 v10 新 OOS 验证 ({OOS_START}-{OOS_END}) ===\n", flush=True)
    print(f"目的: pump 是真 alpha 还是 v3-v8 类 OOS 过拟合?\n", flush=True)

    daily = load_window(DATA_START, "20260522", with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")

    for name in ["r20_v16_long_nost", "r5_v17_long_nost", "r5_pump_lgbm_v1"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        if name == "r5_pump_lgbm_v1":
            daily["pred_pump"] = 1 / (1 + np.exp(-b.predict(X)))
        else:
            daily[f"pred_{name}"] = b.predict(X)
    print(f"  推理完成 {time.time()-t0:.0f}s", flush=True)

    ind_mom = compute_ind_mom(daily)
    r20_lab = compute_r20_label(daily)
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    daily = daily.merge(r20_lab, on=["ts_code", "trade_date"], how="left")

    print(f"\n  {'mode':10s} {'α':9s} {'Sharpe':8s} {'灾难月':6s} {'最差月':8s} {'持仓天数':6s}",
           flush=True)
    results = []
    monthly_dict = {}
    for mode in ["v6", "v10a", "v10b"]:
        hold = build_dual(daily, ind_mom, mode)
        mt = compute_metrics(hold, daily)
        print(f"  {mode:10s} {mt['alpha']:+.3f}pp  {mt['sharpe']:+.2f}    "
               f"{mt['bad_months']}    {mt['worst_month']:+.2f}pp     {mt['n_hold_days']}",
               flush=True)
        results.append({"mode": mode, **{k: mt[k] for k in
                          ['alpha','sharpe','bad_months','worst_month','n_hold_days']}})
        monthly_dict[mode] = mt["monthly"]

    print(f"\n--- 月度对比 ---", flush=True)
    months = sorted(set().union(*[set(m.index) for m in monthly_dict.values()]))
    for m_ in months:
        vals = []
        for mode in ["v6", "v10a", "v10b"]:
            v = monthly_dict[mode].get(m_, np.nan)
            vals.append(f"{v:+.2f}pp" if pd.notna(v) else "    -    ")
        print(f"  {m_}: v6={vals[0]}  v10a={vals[1]}  v10b={vals[2]}", flush=True)

    pd.DataFrame(results).to_csv(OUT / "compare.csv", index=False)
    print(f"\n输出: {OUT / 'compare.csv'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
