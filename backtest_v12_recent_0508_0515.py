"""V12 双轨 v5 最新一周实战回测 (20260508-20260515).

5 个交易日实测: 每日 EOD 推荐双轨, 隔夜 r1 收益累计净值
- 0508 推荐 → 0509 开盘买 → 0511 开盘卖 (r1_next_open 隔夜)
- 实际是日内换仓, 每日翻牌

也跟 r5 (5 日持仓) 评估: 0508 推荐 → 0509 开盘买 → 0515 收盘卖 (5 个交易日)

输出: output/backtest_v12_recent_0508/report.md
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

OUT = ROOT / "output" / "backtest_v12_recent_0508"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"

# 范围: 拉宽一点 (好算 long_return + 行业 mom), 但分析窗口只关注 5/8-5/15
DATA_START = "20260301"
DATA_END = "20260515"   # factors_v3 末日 (推理日)
TEST_DAYS = ["20260508", "20260511", "20260512", "20260513", "20260514", "20260515"]
# 现有 daily cache 已扩到 5/22, 所有 5/8-5/15 都能算 r5 label (5/15+5=5/22)
# r1 label 5/15 仍缺 (需要 5/18 open, 但 r1 用的是 next_open, 5/15 next 是 5/18 → daily 有)

COST_BPS = 35.0 / 10000
P_BUY_STATIC = 0.05
PYR_VELOCITY_QUANTILE = 0.35
M_EXCL = 0.10
INDUSTRY_CAP = 0.20            # v5 单轨内 cap
CROSS_TRACK_CAP = 0.30         # v6 跨轨累计 cap
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


def apply_industry_cap(pool: pd.DataFrame, alloc: float, cap: float,
                          prior_alloc: dict = None) -> pd.DataFrame:
    """支持跨轨累计 cap (prior_alloc = A 已用行业 alloc)."""
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


def build_dual_one_day(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """对单日 g 跑完整 V12 v5 双轨, 返回 (a_pool, b_pool, info)."""
    # m_excl
    ind_ok = g["industry_mom_60d_rank"].isna() | (g["industry_mom_60d_rank"] >= M_EXCL)
    g = g[ind_ok].copy()

    # r20 top 5%
    g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
    m_buy = g["r20_rank"] >= (1 - P_BUY_STATIC)
    if "pyr_velocity_20_60" in g.columns:
        p35 = g["pyr_velocity_20_60"].quantile(PYR_VELOCITY_QUANTILE)
        m_pyr = g["pyr_velocity_20_60"] < p35
    else:
        m_pyr = pd.Series(True, index=g.index)

    v7c = g[m_buy & m_pyr].copy()
    if len(v7c) == 0:
        return pd.DataFrame(), pd.DataFrame(), {"n_v7c": 0}

    v7c["r5_long_rank"] = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
    v7c = v7c.sort_values("r5_long_rank")
    a_pool = v7c.head(MAX_A_STOCKS).copy()
    a_pool = apply_industry_cap(a_pool, TRACK_A_PCT, INDUSTRY_CAP)
    # v6: B 轨先轨内 cap, 然后跨轨 cap (用 A 的行业 alloc)
    a_ind_alloc = industry_alloc_dict(a_pool, TRACK_A_PCT)
    b_candidates = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
    b_pool = b_candidates.head(MAX_B_STOCKS).copy()
    b_pool = apply_industry_cap(b_pool, TRACK_B_PCT, INDUSTRY_CAP)
    b_pool = apply_industry_cap(b_pool, TRACK_B_PCT, CROSS_TRACK_CAP,
                                  prior_alloc=a_ind_alloc)
    return a_pool, b_pool, {"n_v7c": len(v7c)}


def main():
    t0 = time.time()
    print(f"\n=== V12 双轨 v5 最新一周 (5/8-5/15) 实战 ===\n", flush=True)

    # 1. 日线数据 + 推理
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
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

    print(f"\n[2] 行业 60d momentum ...", flush=True)
    ind_mom = compute_industry_60d_mom(daily)
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")

    # 用 daily cache (含 5/18-5/22 最新数据) 自己算 r1 + r5, 避免 factors_v3 末日 r1 NaN
    print(f"\n[3] 加载 daily cache + 算 r1/r5 (含 5/22 最新数据) ...", flush=True)
    files = sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"])
    # r1_recent: t 日 EOD 推荐 → t+1 日 next_open 买入 → t+1 日 next_open 卖出 (但 r1 是隔夜, 用 (open[t+1]/close[t] - 1))
    # 但 V12 设计 → 持有从 t+1 开盘到 next day 开盘 ≈ 24 小时
    # 实际更简单: r1 = (open[t+1] - close[t]) / close[t] = "隔夜跳空收益"
    # 但用户语境里 r1_next_open 是 next_open/today_open - 1 = 一日持仓 (建仓 + 平仓都用 next_open)
    # 用 factors_v3 一致定义: r1 = next_next_open / next_open - 1 (= 持有 1 个交易日的收益)
    big["close_today"] = big["close"]
    big["open_tomorrow"] = big.groupby("ts_code")["open"].shift(-1)
    big["open_day_after"] = big.groupby("ts_code")["open"].shift(-2)
    big["close_5d"] = big.groupby("ts_code")["close"].shift(-5)
    # r1_next_open: 持仓从 t+1 open 到 t+2 open (T+1 一日)
    big["r1_recent"] = (big["open_day_after"] / big["open_tomorrow"] - 1) * 100
    # r5: 持仓从 t+1 open 到 t+5 close (5 个交易日)
    big["r5_recent"] = (big["close_5d"] / big["open_tomorrow"] - 1) * 100
    r_df = big[["ts_code", "trade_date", "r1_recent", "r5_recent"]]
    daily = daily.merge(r_df, on=["ts_code", "trade_date"], how="left")
    daily["r1_next_open"] = daily["r1_recent"]   # 兼容下游列名

    # 3. 每日跑双轨
    print(f"\n[4] 逐日跑双轨 (5/8-5/14) ...", flush=True)
    all_holdings = []
    md_per_day = []
    for d_ in TEST_DAYS:
        g = daily[daily["trade_date"] == d_].copy()
        if len(g) < 100:
            print(f"  {d_}: 无数据", flush=True); continue

        a_pool, b_pool, info = build_dual_one_day(g)
        n_a, n_b = len(a_pool), len(b_pool)
        per_a = TRACK_A_PCT / n_a if n_a > 0 else 0
        per_b = TRACK_B_PCT / n_b if n_b > 0 else 0
        print(f"\n  {d_}: V7c={info['n_v7c']}  →  A {n_a} 股 (单仓 {per_a*100:.2f}%) / "
               f"B {n_b} 股 (单仓 {per_b*100:.2f}%)", flush=True)

        # A 轨详情
        a_pool["alloc"] = per_a
        b_pool["alloc"] = per_b
        for tag, pool in [("A", a_pool), ("B", b_pool)]:
            for _, row in pool.iterrows():
                all_holdings.append({
                    "entry_date": d_, "track": tag, "ts_code": row["ts_code"],
                    "industry": row.get("industry", ""),
                    "r5_long_rank": float(row.get("r5_long_rank", 0)),
                    "r1_next_open": float(row["r1_next_open"]) if pd.notna(row.get("r1_next_open")) else np.nan,
                    "r5_recent": float(row["r5_recent"]) if pd.notna(row.get("r5_recent")) else np.nan,
                    "alloc": float(row["alloc"]),
                })

        # 日报
        md_per_day.append(f"### {d_}\n\n")
        md_per_day.append(f"V7c: {info['n_v7c']}; A {n_a} 股; B {n_b} 股\n\n")
        if n_a > 0:
            md_per_day.append(f"**A 轨 (单仓 {per_a*100:.2f}%)**:\n")
            md_per_day.append("| # | 代码 | 行业 | r5_rank | r1 % | r5(5d) % |\n|---|---|---|---|---|---|\n")
            for i, (_, r) in enumerate(a_pool.iterrows()):
                r1 = f"{r['r1_next_open']:+.2f}" if pd.notna(r.get("r1_next_open")) else "-"
                r5 = f"{r['r5_recent']:+.2f}" if pd.notna(r.get("r5_recent")) else "-"
                md_per_day.append(f"| {i+1} | {r['ts_code']} | {r.get('industry','')} | "
                                  f"{r['r5_long_rank']:.3f} | {r1} | {r5} |\n")
        md_per_day.append("\n")

    if not all_holdings:
        print("无持仓数据"); return
    hold = pd.DataFrame(all_holdings)
    hold.to_csv(OUT / "holdings.csv", index=False)

    # 4. 每日组合 r1 收益 + 累计
    print(f"\n\n=== 每日组合 r1 收益 (隔夜) ===\n", flush=True)
    hold["weighted_r1"] = hold["alloc"] * hold["r1_next_open"]
    hold["weighted_r5"] = hold["alloc"] * hold["r5_recent"]
    by_date = hold.groupby("entry_date").agg(
        n=("ts_code", "count"),
        port_r1_gross=("weighted_r1", "sum"),
        port_r5_gross=("weighted_r5", "sum"),
    ).reset_index()
    # 减成本
    total_alloc = TRACK_A_PCT + TRACK_B_PCT
    by_date["port_r1_net"] = by_date["port_r1_gross"] - total_alloc * COST_BPS * 200
    by_date["port_r5_net"] = by_date["port_r5_gross"] - total_alloc * COST_BPS * 200

    # 市场
    mkt = daily.groupby("trade_date").agg(
        mkt_r1=("r1_next_open", lambda x: x.dropna().mean()),
        mkt_r5=("r5_recent", lambda x: x.dropna().mean()),
    ).reset_index().rename(columns={"trade_date": "entry_date"})
    by_date = by_date.merge(mkt, on="entry_date", how="left")
    by_date["alpha_r1"] = by_date["port_r1_net"] - by_date["mkt_r1"] * total_alloc
    by_date["alpha_r5"] = by_date["port_r5_net"] - by_date["mkt_r5"] * total_alloc

    # 累计净值
    by_date["nav_r1"] = (1 + by_date["port_r1_net"] / 100).cumprod()
    by_date["nav_mkt_r1"] = (1 + by_date["mkt_r1"] * total_alloc / 100).cumprod()

    print(f"  {'date':10s} {'n':4s} {'r1_net':9s} {'mkt_r1':9s} {'α_r1':9s} {'r5_net':9s} {'mkt_r5':9s} {'α_r5':9s}",
           flush=True)
    for _, r in by_date.iterrows():
        print(f"  {r['entry_date']:10s} {int(r['n']):3d}  "
               f"{r['port_r1_net']:+6.3f}%  {r['mkt_r1']:+6.3f}%  {r['alpha_r1']:+6.3f}pp  "
               f"{r['port_r5_net']:+6.2f}%  {r['mkt_r5']:+6.2f}%  {r['alpha_r5']:+6.2f}pp",
               flush=True)

    print(f"\n--- 5 日合计 ---")
    total_r1_net = (by_date["nav_r1"].iloc[-1] - 1) * 100
    total_mkt_r1 = (by_date["nav_mkt_r1"].iloc[-1] - 1) * 100
    print(f"  组合 5 日累计 (r1 路径): {total_r1_net:+.2f}%")
    print(f"  市场 5 日累计 (r1 路径): {total_mkt_r1:+.2f}%")
    print(f"  累计 α: {total_r1_net - total_mkt_r1:+.2f}pp")
    print(f"  r1 均日 α: {by_date['alpha_r1'].mean():+.3f}pp")
    print(f"  r5 均日 α: {by_date['alpha_r5'].mean():+.3f}pp")

    by_date.to_csv(OUT / "daily_pnl.csv", index=False)

    # 报告
    md = [f"# V12 双轨 v5 最新一周实战 (5/8-5/15)\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            f"## 每日组合表现\n\n",
            "| 日期 | 持仓 | 组合 r1 % | 市场 r1 % | α_r1 | 组合 r5 % | 市场 r5 % | α_r5 | NAV |\n",
            "|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in by_date.iterrows():
        md.append(f"| {r['entry_date']} | {int(r['n'])} | "
                   f"{r['port_r1_net']:+.3f} | {r['mkt_r1']:+.3f} | "
                   f"{r['alpha_r1']:+.3f}pp | "
                   f"{r['port_r5_net']:+.2f} | {r['mkt_r5']:+.2f} | "
                   f"{r['alpha_r5']:+.2f}pp | {r['nav_r1']:.4f} |\n")
    md.append(f"\n**5 日累计**:\n")
    md.append(f"- 组合 r1: {total_r1_net:+.2f}%\n")
    md.append(f"- 市场 r1: {total_mkt_r1:+.2f}%\n")
    md.append(f"- 累计 α: {total_r1_net - total_mkt_r1:+.2f}pp\n\n")
    md.append(f"## 每日持仓详情\n\n")
    md.extend(md_per_day)

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
