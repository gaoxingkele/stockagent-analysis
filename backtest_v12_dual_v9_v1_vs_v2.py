"""V12 双轨 v9: OOS 扩窗对比.

共同 OOS 期 20260201-20260515 (~70 日), 用 v1 (cut 20250930) vs v2 (cut 20260131) 模型,
跑 V12 双轨, 看扩窗后 α/Sharpe 是否提升.

v1 模型: r5_v17_long_nost, r20_v16_long_nost (旧)
v2 模型: r5_v17_long_nost_v2, r20_v16_long_nost_v2 (扩窗后)

注: v2 的训练集包含部分 20251001-20260131, 这段对 v2 是 in-sample. 但 20260201+
对两者都是 OOS, 是公平的对比.
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

OUT = ROOT / "output" / "backtest_v12_dual_v9"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

# 共同 OOS 期 (对 v1 和 v2 都是真 OOS)
OOS_START = "20260201"
OOS_END = "20260515"
DATA_START = "20251001"  # 加载更早, 算 m_excl + label

COST_BPS = 35.0 / 10000
P_BUY_STATIC = 0.05
PYR_VELOCITY_QUANTILE = 0.35
M_EXCL = 0.10
CAP_IN = 0.20
CAP_CROSS = 0.20
TRACK_A_PCT = 0.70
TRACK_B_PCT = 0.20
MAX_A_STOCKS = 8
MAX_B_STOCKS = 15


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


def compute_r20_label(daily):
    """从 daily cache 算 r20 = close[t+20] / open[t+1] - 1."""
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    files = sorted(daily_dir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"])
    big["next_open"] = big.groupby("ts_code")["open"].shift(-1)
    big["close_20d"] = big.groupby("ts_code")["close"].shift(-20)
    big["r20_fresh"] = (big["close_20d"] / big["next_open"] - 1) * 100
    return big[["ts_code", "trade_date", "r20_fresh"]]


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


def build_dual(daily, ind_mom, r20_col, r5_col):
    """用指定 r20/r5 模型预测列构建双轨持仓."""
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if d_ < OOS_START or d_ > OOS_END: continue
        if len(g) < 500: continue
        g = g.copy()
        ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
        g = g[ind_ok]
        if len(g) < 100: continue
        g["r20_rank"] = g[r20_col].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY_STATIC)
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_VELOCITY_QUANTILE)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        v7c = g[m_buy & m_pyr].copy()
        if len(v7c) == 0: continue
        v7c["r5_long_rank"] = v7c[r5_col].rank(pct=True, method="first")
        v7c = v7c.sort_values("r5_long_rank")
        a_pool = v7c.head(MAX_A_STOCKS).copy()
        a_pool = apply_cap(a_pool, TRACK_A_PCT, CAP_IN)
        a_ind = industry_alloc_dict(a_pool, TRACK_A_PCT)
        b_cand = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_cand.head(MAX_B_STOCKS).copy()
        b_pool = apply_cap(b_pool, TRACK_B_PCT, CAP_IN)
        b_pool = apply_cap(b_pool, TRACK_B_PCT, CAP_CROSS, prior=a_ind)
        for tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({"entry_date": d_, "ts_code": row["ts_code"],
                                "industry": row.get("industry", ""),
                                "r20_fresh": float(row["r20_fresh"]) if pd.notna(row.get("r20_fresh")) else np.nan,
                                "alloc_pct": per_stock})
    return pd.DataFrame(rows)


def compute_metrics(hold, daily):
    hold = hold.dropna(subset=["r20_fresh"]).copy()
    hold["r20_fresh"] = hold["r20_fresh"].clip(-30, 30)
    hold["weighted"] = hold["alloc_pct"] * hold["r20_fresh"]
    total = TRACK_A_PCT + TRACK_B_PCT
    pnl = hold.groupby("entry_date").agg(gross=("weighted", "sum"),
                                          n=("ts_code", "count")).reset_index()
    pnl["net"] = pnl["gross"] - total * COST_BPS * 200
    mkt = daily[(daily["trade_date"] >= OOS_START) & (daily["trade_date"] <= OOS_END)].groupby(
        "trade_date")["r20_fresh"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt_r20"]
    pnl = pnl.merge(mkt, on="entry_date", how="left")
    pnl["alpha"] = pnl["net"] - pnl["mkt_r20"] * total
    pnl["month"] = pnl["entry_date"].str[:6]
    monthly = pnl.groupby("month").agg(n=("entry_date", "count"),
                                         alpha=("alpha", "mean")).reset_index()
    return {
        "alpha": pnl["alpha"].mean(),
        "sharpe": pnl["alpha"].mean() / (pnl["alpha"].std() + 1e-9) * np.sqrt(12),
        "n_hold_days": pnl["entry_date"].nunique(),
        "bad_months": (monthly["alpha"] < -1.0).sum(),
        "worst_month": monthly["alpha"].min(),
        "monthly": monthly,
    }


def main():
    t0 = time.time()
    print(f"\n=== V12 v9: v1 (cut 20250930) vs v2 (cut 20260131) ===\n", flush=True)
    print(f"共同 OOS 期: {OOS_START} - {OOS_END}\n", flush=True)

    daily = load_window(DATA_START, "20260522", with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")

    # 推理 v1 + v2 模型 (r5/r20 各两版本)
    print("[1] 推理 4 模型 (v1+v2) ...", flush=True)
    for name in ["r20_v16_long_nost", "r5_v17_long_nost",
                   "r20_v16_long_nost_v2", "r5_v17_long_nost_v2"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily[f"pred_{name}"] = b.predict(X)
        print(f"  {name}: μ={daily[f'pred_{name}'].mean():+.3f}", flush=True)

    print("\n[2] 行业 60d momentum ...", flush=True)
    ind_mom = compute_ind_mom(daily)

    print("\n[3] r20 forward label (最新 daily 算) ...", flush=True)
    r20_label = compute_r20_label(daily)
    r20_label["trade_date"] = r20_label["trade_date"].astype(str)
    daily = daily.merge(r20_label, on=["ts_code", "trade_date"], how="left")

    # v1 双轨
    print("\n[4] v1 模型双轨 ...", flush=True)
    hold_v1 = build_dual(daily, ind_mom, "pred_r20_v16_long_nost", "pred_r5_v17_long_nost")
    m_v1 = compute_metrics(hold_v1, daily)

    # v2 双轨
    print("[5] v2 模型双轨 ...", flush=True)
    hold_v2 = build_dual(daily, ind_mom, "pred_r20_v16_long_nost_v2", "pred_r5_v17_long_nost_v2")
    m_v2 = compute_metrics(hold_v2, daily)

    # 对比报告
    print(f"\n{'='*50}\n=== 共同 OOS {OOS_START}-{OOS_END} 对比 ===\n{'='*50}\n")
    print(f"{'指标':16s} {'v1 (cut 20250930)':22s} {'v2 (cut 20260131)':22s}")
    print(f"{'-'*16:16s} {'-'*22:22s} {'-'*22:22s}")
    print(f"{'持仓天数':16s} {m_v1['n_hold_days']:22d} {m_v2['n_hold_days']:22d}")
    print(f"{'月化 α':16s} {m_v1['alpha']:+.3f}pp {' ':12s} {m_v2['alpha']:+.3f}pp {' ':12s}")
    print(f"{'Sharpe (月)':16s} {m_v1['sharpe']:+.2f} {' ':17s} {m_v2['sharpe']:+.2f} {' ':17s}")
    print(f"{'灾难月':16s} {int(m_v1['bad_months']):22d} {int(m_v2['bad_months']):22d}")
    print(f"{'最差月 α':16s} {m_v1['worst_month']:+.3f}pp {' ':12s} {m_v2['worst_month']:+.3f}pp {' ':12s}")

    print(f"\n--- v1 月度 ---")
    for _, r in m_v1["monthly"].iterrows():
        print(f"  {r['month']}: n={int(r['n']):2d} α={r['alpha']:+.3f}pp")
    print(f"\n--- v2 月度 ---")
    for _, r in m_v2["monthly"].iterrows():
        print(f"  {r['month']}: n={int(r['n']):2d} α={r['alpha']:+.3f}pp")

    delta_alpha = m_v2["alpha"] - m_v1["alpha"]
    delta_sharpe = m_v2["sharpe"] - m_v1["sharpe"]
    print(f"\n--- 提升 (v2 - v1) ---")
    print(f"  α: {delta_alpha:+.3f}pp")
    print(f"  Sharpe: {delta_sharpe:+.2f}")

    # 保存
    pd.DataFrame([
        {"version": "v1_cut_20250930", **{k: m_v1[k] for k in ['alpha','sharpe','n_hold_days','bad_months','worst_month']}},
        {"version": "v2_cut_20260131", **{k: m_v2[k] for k in ['alpha','sharpe','n_hold_days','bad_months','worst_month']}},
    ]).to_csv(OUT / "compare.csv", index=False)

    md = [f"# V12 v9: OOS 扩窗 v1 vs v2 对比\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"共同 OOS: {OOS_START}-{OOS_END}\n\n",
            "## 对比\n\n| 指标 | v1 (cut 20250930) | v2 (cut 20260131) | Δ |\n|---|---|---|---|\n",
            f"| 持仓天数 | {m_v1['n_hold_days']} | {m_v2['n_hold_days']} | - |\n",
            f"| 月化 α | {m_v1['alpha']:+.3f}pp | {m_v2['alpha']:+.3f}pp | {delta_alpha:+.3f}pp |\n",
            f"| Sharpe 月 | {m_v1['sharpe']:+.2f} | {m_v2['sharpe']:+.2f} | {delta_sharpe:+.2f} |\n",
            f"| 灾难月 | {int(m_v1['bad_months'])} | {int(m_v2['bad_months'])} | - |\n",
            f"| 最差月 α | {m_v1['worst_month']:+.3f}pp | {m_v2['worst_month']:+.3f}pp | - |\n",
            ]
    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
