"""V12 排序信号对比: composite vs ratio vs ratio+过滤.

3 模式 (统一用 v3c 模型, 只换 V7c 池排序信号):
  composite      pump_up - 0.5*pump_down (现状)
  ratio          pump_up / (pump_down + 0.01)
  ratio_filter   ratio 排序 + 硬过滤 (ratio < 2 排除)

OOS: 20251001 - 20260331
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

OUT = ROOT / "output" / "backtest_v12_ratio"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15

PUMP_DOWN_EXCL_RANK = 0.80   # 池内 pump_down Top 20% 排除
RATIO_FILTER_MIN = 2.0       # ratio < 2 视为震荡, 模式 ratio_filter 时硬过滤
EPS = 0.01

OOS_START = "20251001"
OOS_END = "20260331"


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def compute_aux(daily):
    ddir = ROOT / "output/tushare_cache/daily"
    files = sorted(ddir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"])
                for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    big["close_20d"] = big.groupby("ts_code")["close"].shift(-20)
    big["r20_fresh"] = (big["close_20d"] / big["next_open"] - 1) * 100
    big["mom_60d"] = (big.groupby("ts_code")["close"].shift(1) /
                        big.groupby("ts_code")["close"].shift(61) - 1)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "industry"]].drop_duplicates("ts_code")
    big = big.merge(basic, on="ts_code", how="left")
    ind = big.dropna(subset=["mom_60d"]).groupby(["trade_date", "industry"]).agg(
        industry_mom_60d=("mom_60d", "mean")).reset_index()
    ind["industry_mom_60d_rank"] = ind.groupby("trade_date")["industry_mom_60d"].rank(
        pct=True, method="first")
    return big[["ts_code", "trade_date", "r20_fresh"]], ind


def apply_cap(pool, alloc, cap, prior=None, sort_col="sort_score"):
    if pool.empty or cap >= 1.0: return pool
    per_stock = alloc / len(pool)
    pool = pool.sort_values(sort_col, ascending=False)
    ia = dict(prior or {})
    keep = []
    for idx, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        new = ia.get(ind, 0) + per_stock
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


def build_dual(daily, mode):
    rows = []
    n_ratio_filtered = 0
    for d_, g in daily.groupby("trade_date"):
        if d_ < OOS_START or d_ > OOS_END: continue
        if len(g) < 500: continue
        g = g.copy()

        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 100: continue

        g["pump_up"] = g["pump_up_v3c"]
        g["pump_down"] = g["pump_down_v3c"]
        g["ratio"] = g["pump_up"] / (g["pump_down"] + EPS)

        # 选排序信号
        if mode == "composite":
            g["sort_score"] = g["pump_up"] - 0.5 * g["pump_down"]
        elif mode == "ratio":
            g["sort_score"] = g["ratio"]
        elif mode == "ratio_filter":
            g["sort_score"] = g["ratio"]

        # V7c 铁律
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_Q)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue

        # pump_down 池内 Top 20% 排除 (3 模式都用)
        v7c["pd_rank"] = v7c["pump_down"].rank(pct=True, method="first")
        v7c = v7c[v7c["pd_rank"] < PUMP_DOWN_EXCL_RANK]
        if len(v7c) == 0: continue

        # ratio_filter 模式: 再加 ratio >= 2.0 硬过滤
        if mode == "ratio_filter":
            before = len(v7c)
            v7c = v7c[v7c["ratio"] >= RATIO_FILTER_MIN]
            n_ratio_filtered += (before - len(v7c))
            if len(v7c) == 0: continue

        # 排序
        v7c = v7c.sort_values("sort_score", ascending=False)
        a_pool = v7c.head(MAX_A).copy()
        a_pool = apply_cap(a_pool, A_PCT, CAP_IN, sort_col="sort_score")
        a_ind = industry_alloc(a_pool, A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B).copy()
        b_pool = apply_cap(b_pool, B_PCT, CAP_IN, sort_col="sort_score")
        b_pool = apply_cap(b_pool, B_PCT, CAP_CROSS, prior=a_ind, sort_col="sort_score")

        for tag, pool, alloc in [("A", a_pool, A_PCT), ("B", b_pool, B_PCT)]:
            if len(pool) == 0: continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                                "track": tag,
                                "pump_up": float(row["pump_up"]),
                                "pump_down": float(row["pump_down"]),
                                "ratio": float(row["ratio"]),
                                "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                                "alloc_pct": per})
    if mode == "ratio_filter":
        print(f"    [ratio_filter] ratio<{RATIO_FILTER_MIN} 排除总样本: {n_ratio_filtered:,}",
               flush=True)
    return pd.DataFrame(rows)


def compute_metrics(hold, daily):
    if hold.empty:
        return {"alpha": 0, "sharpe": 0, "bad_months": 0, "worst_month": 0,
                 "n_hold_days": 0, "n_picks": 0, "monthly": pd.Series(dtype=float),
                 "avg_ratio": 0}
    h = hold.dropna(subset=["r20_fresh"]).copy()
    h["r20_fresh"] = h["r20_fresh"].clip(-30, 30)
    h["w"] = h["alloc_pct"] * h["r20_fresh"]
    pnl = h.groupby("entry_date").agg(gross=("w", "sum"),
                                        n=("ts_code", "count"),
                                        total_alloc=("alloc_pct", "sum")).reset_index()
    pnl["net"] = pnl["gross"] - pnl["total_alloc"] * COST_BPS * 200
    mkt = daily[(daily["trade_date"] >= OOS_START) & (daily["trade_date"] <= OOS_END)].groupby(
        "trade_date")["r20_fresh"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt"]
    pnl = pnl.merge(mkt, on="entry_date", how="left")
    pnl["alpha"] = pnl["net"] - pnl["mkt"] * pnl["total_alloc"]
    pnl["month"] = pnl["entry_date"].str[:6]
    monthly = pnl.groupby("month")["alpha"].mean()
    return {
        "alpha": pnl["alpha"].mean(),
        "sharpe": pnl["alpha"].mean() / (pnl["alpha"].std() + 1e-9) * np.sqrt(12),
        "bad_months": (monthly < -1.0).sum(),
        "worst_month": monthly.min(),
        "n_hold_days": pnl["entry_date"].nunique(),
        "n_picks": len(hold),
        "avg_ratio": hold["ratio"].mean(),
        "monthly": monthly,
    }


def main():
    t0 = time.time()
    print(f"\n=== V12 排序信号对比 (composite / ratio / ratio+过滤) "
           f"OOS {OOS_START}-{OOS_END} ===\n", flush=True)

    daily = load_window("20251001", "20260331", with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")
    print(f"  daily: {len(daily):,} 行", flush=True)

    print("\n  [1] 推理 r20_v16_long_nost ...", flush=True)
    b, fc, ind_map = load_model("r20_v16_long_nost")
    if ind_map:
        daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    for c in fc:
        if c not in daily.columns: daily[c] = 0.0
    X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    daily["pred_r20_v16_long_nost"] = b.predict(X)

    print("  [2] 推理 v3c 3way ...", flush=True)
    b, fc, ind_map = load_model("r5_pump_3way_lgbm_v3c")
    if ind_map:
        daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    for c in fc:
        if c not in daily.columns: daily[c] = 0.0
    X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    proba = b.predict(X)
    daily["pump_down_v3c"] = proba[:, 1]
    daily["pump_up_v3c"] = proba[:, 2]

    print("  [3] 算 r20_fresh + industry_mom ...", flush=True)
    aux, ind = compute_aux(daily)
    aux["trade_date"] = aux["trade_date"].astype(str)
    ind["trade_date"] = ind["trade_date"].astype(str)
    daily = daily.merge(aux, on=["ts_code", "trade_date"], how="left")
    daily = daily.merge(ind, on=["trade_date", "industry"], how="left")

    print(f"\n=== 回测 ===\n", flush=True)
    print(f"  {'mode':16s} {'α':9s} {'Sharpe':8s} {'灾':3s} {'最差':7s} {'天数':5s} "
           f"{'picks':6s} {'平均ratio':>9s}", flush=True)
    monthly_dict = {}
    for mode in ["composite", "ratio", "ratio_filter"]:
        hold = build_dual(daily, mode)
        mt = compute_metrics(hold, daily)
        print(f"  {mode:16s} {mt['alpha']:+.3f}pp  {mt['sharpe']:+.2f}    "
               f"{mt['bad_months']}    {mt['worst_month']:+.2f}pp   "
               f"{mt['n_hold_days']}    {mt['n_picks']}    "
               f"{mt['avg_ratio']:>7.2f}x", flush=True)
        monthly_dict[mode] = mt["monthly"]

    print(f"\n  月度对比:", flush=True)
    months = sorted(set().union(*[set(m.index) for m in monthly_dict.values()]))
    for m_ in months:
        vals = []
        for mode in ["composite", "ratio", "ratio_filter"]:
            v = monthly_dict[mode].get(m_, np.nan)
            vals.append(f"{v:+.2f}pp" if pd.notna(v) else "    -    ")
        print(f"    {m_}: composite={vals[0]}  ratio={vals[1]}  ratio_filter={vals[2]}",
               flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
