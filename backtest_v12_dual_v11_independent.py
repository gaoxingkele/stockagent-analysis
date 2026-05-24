"""V12 双轨 v11: pump_up + pump_down 双向独立 + regime gate.

设计原则: pump_up 和 pump_down 独立 (不融合 composite)
  - pump_up_score: 做多评分 (排序信号, 高=好)
  - pump_down_score: 做空避免评分 (硬过滤, 高=该股危险)

3 路对比:
  v10 (基准): pump_up 降序, 不用 pump_down
  v11a (独立 + 排除): pump_up 降序, pump_down > 阈值 的股排除
  v11b (+ regime gate): v11a + 全市场过去 20 日跌幅 < -2% 时减仓到 45%

双 OOS 验证 (旧 142 日 + 新 70 日).
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

OUT = ROOT / "output" / "backtest_v12_dual_v11"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

COST_BPS = 35.0 / 10000
P_BUY, PYR_Q, M_EXCL, CAP_IN, CAP_CROSS = 0.05, 0.35, 0.10, 0.20, 0.20
A_PCT, B_PCT = 0.70, 0.20
MAX_A, MAX_B = 8, 15

# v11 新参数
PUMP_DOWN_EXCL_THRESHOLD = 0.50    # pump_down > 0.5 排除 (硬过滤)
REGIME_GATE_THRESHOLD = -2.0       # 过去 20 日 mkt_ret < -2% 视为 regime 失效
REGIME_GATE_A_PCT_REDUCED = 0.35   # 失效时 A 仓位降到 35%
REGIME_GATE_B_PCT_REDUCED = 0.10   # 失效时 B 仓位降到 10% (现金 55%)


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


def compute_r20_label_and_market(daily):
    """从 daily cache 算 r20 forward label + 过去 20 日 mkt_ret (regime gate 用)."""
    ddir = ROOT / "output/tushare_cache/daily"
    files = sorted(ddir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"])
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    big["close_20d"] = big.groupby("ts_code")["close"].shift(-20)
    big["r20_fresh"] = (big["close_20d"] / big["next_open"] - 1) * 100
    # 过去 20 日个股 ret
    big["close_20d_ago"] = big.groupby("ts_code")["close"].shift(20)
    big["ret_past_20d"] = (big["close"] / big["close_20d_ago"] - 1) * 100
    # 全市场每日过去 20 日均值 (regime 指标)
    mkt = big.groupby("trade_date")["ret_past_20d"].apply(
        lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["trade_date", "mkt_ret_past_20d"]
    return big[["ts_code", "trade_date", "r20_fresh"]], mkt


def apply_cap(pool, alloc, cap, prior=None, sort_col="pump_up_score"):
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


def build_dual(daily, ind_mom, mkt_regime, mode, oos_start, oos_end):
    """mode: 'v10' / 'v11a' / 'v11b'."""
    daily = daily.copy()  # 避免重复调用时污染
    if "industry_mom_60d_rank" not in daily.columns:
        daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    if "mkt_ret_past_20d" not in daily.columns:
        daily = daily.merge(mkt_regime, on="trade_date", how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if d_ < oos_start or d_ > oos_end: continue
        if len(g) < 500: continue
        g = g.copy()

        # regime gate (仅 v11b 启用)
        mkt_r = g["mkt_ret_past_20d"].iloc[0] if not g["mkt_ret_past_20d"].isna().all() else 0
        if mode == "v11b" and mkt_r < REGIME_GATE_THRESHOLD:
            a_pct = REGIME_GATE_A_PCT_REDUCED
            b_pct = REGIME_GATE_B_PCT_REDUCED
        else:
            a_pct = A_PCT
            b_pct = B_PCT

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

        # v11 (a/b): pump_down 硬过滤
        if mode in ("v11a", "v11b"):
            before = len(v7c)
            v7c = v7c[v7c["pump_down_score"] < PUMP_DOWN_EXCL_THRESHOLD]
            after = len(v7c)
            if d_ <= "20251005":  # 头几天打印 debug
                print(f"    [{mode} {d_}] V7c {before} → pump_down<{PUMP_DOWN_EXCL_THRESHOLD} → "
                       f"{after} 股", flush=True)
            if len(v7c) == 0: continue

        # 排序: pump_up 降序
        v7c["pump_up_rank"] = v7c["pump_up_score"].rank(pct=True, method="first")
        v7c = v7c.sort_values("pump_up_score", ascending=False)

        a_pool = v7c.head(MAX_A).copy()
        a_pool = apply_cap(a_pool, a_pct, CAP_IN, sort_col="pump_up_score")
        a_ind = industry_alloc(a_pool, a_pct)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B).copy()
        b_pool = apply_cap(b_pool, b_pct, CAP_IN, sort_col="pump_up_score")
        b_pool = apply_cap(b_pool, b_pct, CAP_CROSS, prior=a_ind, sort_col="pump_up_score")

        for tag, pool, alloc in [("A", a_pool, a_pct), ("B", b_pool, b_pct)]:
            if len(pool) == 0: continue
            per = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                                "track": tag, "industry": row.get("industry", ""),
                                "pump_up": float(row.get("pump_up_score", 0)),
                                "pump_down": float(row.get("pump_down_score", 0)),
                                "regime_gated": (mode == "v11b" and mkt_r < REGIME_GATE_THRESHOLD),
                                "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                                "alloc_pct": per})
    return pd.DataFrame(rows)


def compute_metrics(hold, daily, oos_start, oos_end):
    if hold.empty or "r20_fresh" not in hold.columns:
        return {"alpha": 0, "sharpe": 0, "bad_months": 0, "worst_month": 0,
                 "n_hold_days": 0, "monthly": pd.Series(dtype=float)}
    h = hold.dropna(subset=["r20_fresh"]).copy()
    h["r20_fresh"] = h["r20_fresh"].clip(-30, 30)
    h["w"] = h["alloc_pct"] * h["r20_fresh"]
    pnl = h.groupby("entry_date").agg(gross=("w", "sum"),
                                        n=("ts_code", "count"),
                                        total_alloc=("alloc_pct", "sum")).reset_index()
    # net 用各日实际 total_alloc (v11b regime 减仓时不同)
    pnl["net"] = pnl["gross"] - pnl["total_alloc"] * COST_BPS * 200

    mkt = daily[(daily["trade_date"] >= oos_start) & (daily["trade_date"] <= oos_end)].groupby(
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
        "monthly": monthly,
    }


def run_oos(label, oos_start, oos_end, daily_orig):
    print(f"\n{'='*50}\n=== {label}: {oos_start}-{oos_end} ===\n{'='*50}\n", flush=True)
    daily = daily_orig.copy()

    print("  推理 4 模型 ...", flush=True)
    for name in ["r20_v16_long_nost", "r5_v17_long_nost",
                   "r5_pump_lgbm_v1", "r5_pump_down_lgbm_v1"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        if name == "r5_pump_lgbm_v1":
            # LGBM binary booster.predict 直接输出 prob, 不需 sigmoid
            daily["pump_up_score"] = b.predict(X)
        elif name == "r5_pump_down_lgbm_v1":
            daily["pump_down_score"] = b.predict(X)
        else:
            daily[f"pred_{name}"] = b.predict(X)

    ind_mom = compute_ind_mom(daily)
    r20_lab, mkt_regime = compute_r20_label_and_market(daily)
    r20_lab["trade_date"] = r20_lab["trade_date"].astype(str)
    mkt_regime["trade_date"] = mkt_regime["trade_date"].astype(str)
    # 一次性 merge 全部, 避免 build_dual 内重复
    daily = daily.merge(r20_lab, on=["ts_code", "trade_date"], how="left")
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    daily = daily.merge(mkt_regime, on="trade_date", how="left")
    # debug
    n_pump_down_high = (daily["pump_down_score"] >= PUMP_DOWN_EXCL_THRESHOLD).sum()
    print(f"  pump_down >= {PUMP_DOWN_EXCL_THRESHOLD} 的 bar 数: {n_pump_down_high:,} "
           f"({n_pump_down_high/len(daily)*100:.1f}%)", flush=True)

    print(f"  {'mode':10s} {'α':9s} {'Sharpe':8s} {'灾':3s} {'最差':7s} {'天数':5s}", flush=True)
    results = []
    monthly_dict = {}
    for mode in ["v10", "v11a", "v11b"]:
        hold = build_dual(daily, ind_mom, mkt_regime, mode, oos_start, oos_end)
        mt = compute_metrics(hold, daily, oos_start, oos_end)
        print(f"  {mode:10s} {mt['alpha']:+.3f}pp  {mt['sharpe']:+.2f}    "
               f"{mt['bad_months']}    {mt['worst_month']:+.2f}pp   {mt['n_hold_days']}",
               flush=True)
        results.append({"oos": label, "mode": mode,
                          **{k: mt[k] for k in
                              ['alpha','sharpe','bad_months','worst_month','n_hold_days']}})
        monthly_dict[mode] = mt["monthly"]

    print(f"\n  月度对比:", flush=True)
    months = sorted(set().union(*[set(m.index) for m in monthly_dict.values()]))
    for m_ in months:
        vals = []
        for mode in ["v10", "v11a", "v11b"]:
            v = monthly_dict[mode].get(m_, np.nan)
            vals.append(f"{v:+.2f}pp" if pd.notna(v) else "    -    ")
        print(f"    {m_}: v10={vals[0]}  v11a={vals[1]}  v11b={vals[2]}", flush=True)
    return results


def main():
    t0 = time.time()
    print(f"\n=== V12 v11: pump_up + pump_down 双向独立 + regime gate ===\n", flush=True)

    # 加载 daily 一次, 跑两个 OOS
    daily_orig = load_window("20251001", "20260522", with_mfk=True)
    daily_orig["trade_date"] = daily_orig["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily_orig = daily_orig.merge(lf, on=["ts_code", "trade_date"], how="left")
    print(f"  daily 加载完成: {len(daily_orig):,} 行", flush=True)

    all_results = []
    # 旧 OOS (跟 v6/v10a 一致)
    r1 = run_oos("旧 OOS", "20251001", "20260331", daily_orig)
    all_results.extend(r1)
    # 新 OOS (v2 验证用)
    r2 = run_oos("新 OOS", "20260201", "20260515", daily_orig)
    all_results.extend(r2)

    df = pd.DataFrame(all_results)
    df.to_csv(OUT / "v11_compare.csv", index=False)
    print(f"\n输出: {OUT / 'v11_compare.csv'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
