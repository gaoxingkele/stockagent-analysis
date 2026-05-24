"""V12 双轨 v10: pump_score 集成 (3 路对比).

v6 (基准): A/B 都按 r5_long_rank 升序 (现有)
v10a 纯 pump: A/B 都按 pump_score 降序
v10b 偏置混合:
  A 轨: 0.7 × pump_score + 0.3 × (1 - r5_rank)  (高确定性 + 超跌)
  B 轨: 0.5 × pump_score + 0.5 × r20_rank       (启动+中长期强)

OOS: 20251001-20260331 (跟 v6 一致)
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

OUT = ROOT / "output" / "backtest_v12_dual_v10"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20251001"
OOS_END = "20260331"
COST_BPS = 35.0 / 10000
P_BUY = 0.05
PYR_Q = 0.35
M_EXCL = 0.10
CAP_IN = 0.20
CAP_CROSS = 0.20
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


def apply_cap(pool, alloc, cap, prior=None, sort_col="composite"):
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
    per_stock = alloc / len(pool)
    counts = {}
    for _, row in pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        counts[ind] = counts.get(ind, 0) + 1
    return {k: v * per_stock for k, v in counts.items()}


def build_dual(daily, ind_mom, mode: str):
    """mode: 'v6' / 'v10a' / 'v10b'."""
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()
        # 行业 momentum 过滤
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 100: continue

        # V7c 5 铁律
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_Q)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue

        # 各信号 pct rank (V7c 池内)
        v7c["r5_long_rank_in"] = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
        v7c["pump_score_norm"] = v7c["pred_pump"]  # 已经是 0-1 概率
        v7c["r20_rank_in"] = v7c["pred_r20_v16_long_nost"].rank(pct=True, method="first")

        # 按 mode 计算 composite
        if mode == "v6":
            # A/B 都按 r5 升序 (越低越好) → 取负为 composite (高 = 好)
            v7c["composite_A"] = 1 - v7c["r5_long_rank_in"]
            v7c["composite_B"] = 1 - v7c["r5_long_rank_in"]
        elif mode == "v10a":
            # A/B 都按 pump 降序
            v7c["composite_A"] = v7c["pump_score_norm"]
            v7c["composite_B"] = v7c["pump_score_norm"]
        elif mode == "v10b":
            # A: 0.7 pump + 0.3 (1 - r5)
            # B: 0.5 pump + 0.5 r20
            v7c["composite_A"] = 0.7 * v7c["pump_score_norm"] + 0.3 * (1 - v7c["r5_long_rank_in"])
            v7c["composite_B"] = 0.5 * v7c["pump_score_norm"] + 0.5 * v7c["r20_rank_in"]
        else:
            raise ValueError(mode)

        # A 轨: composite_A top
        v7c_a = v7c.sort_values("composite_A", ascending=False)
        a_pool = v7c_a.head(MAX_A).copy()
        a_pool["composite"] = a_pool["composite_A"]
        a_pool = apply_cap(a_pool, A_PCT, CAP_IN)
        a_ind = industry_alloc(a_pool, A_PCT)
        # B 轨: 剩余股按 composite_B top
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
                                "pump_score": float(row.get("pump_score_norm", 0)),
                                "r20": float(row["r20"]) if pd.notna(row.get("r20")) else np.nan,
                                "alloc_pct": per})
    return pd.DataFrame(rows)


def compute_metrics(hold, daily):
    h = hold.dropna(subset=["r20"]).copy()
    h["r20"] = h["r20"].clip(-30, 30)
    h["w"] = h["alloc_pct"] * h["r20"]
    total = A_PCT + B_PCT
    pnl = h.groupby("entry_date").agg(gross=("w", "sum"), n=("ts_code", "count")).reset_index()
    pnl["net"] = pnl["gross"] - total * COST_BPS * 200
    mkt = daily.groupby("trade_date")["r20"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
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
    print(f"\n=== V12 v10 pump_score 集成对比 ===\n", flush=True)

    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")

    # 推理 3 个模型: r20_long, r5_long, pump
    for name in ["r20_v16_long_nost", "r5_v17_long_nost", "r5_pump_lgbm_v1"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        if name == "r5_pump_lgbm_v1":
            # binary classifier 输出 prob
            daily["pred_pump"] = 1 / (1 + np.exp(-b.predict(X)))  # sigmoid
        else:
            daily[f"pred_{name}"] = b.predict(X)
    print(f"  推理完成 {time.time()-t0:.0f}s", flush=True)
    print(f"  pump_score 均值: {daily['pred_pump'].mean():.3f}, "
           f"std: {daily['pred_pump'].std():.3f}", flush=True)

    ind_mom = compute_ind_mom(daily)

    # 跑 3 路
    print(f"\n  {'mode':10s} {'α':9s} {'Sharpe':8s} {'灾难月':6s} {'最差月':8s}", flush=True)
    results = []
    for mode in ["v6", "v10a", "v10b"]:
        hold = build_dual(daily, ind_mom, mode)
        mt = compute_metrics(hold, daily)
        print(f"  {mode:10s} {mt['alpha']:+.3f}pp  {mt['sharpe']:+.2f}    "
               f"{mt['bad_months']}    {mt['worst_month']:+.2f}pp", flush=True)
        results.append({"mode": mode, **{k: mt[k] for k in
                          ['alpha','sharpe','bad_months','worst_month','n_hold_days']}})

    pd.DataFrame(results).to_csv(OUT / "compare.csv", index=False)
    print(f"\n输出: {OUT / 'compare.csv'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
