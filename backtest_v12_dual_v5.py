"""V12 双轨 v5: 动态行业 cap + m_excl 自适应分化度.

两个新维度:
  C. 行业 cap: 在 A 轨内, 单行业总仓位 ≤ industry_cap (默认 20%)
     超 cap 的同行业股按 r5_long_rank 升序保留, 多余踢回 B 轨
  D. m_excl 自适应: 按当日行业 60d momentum 的标准差判断市场分化度
     std 大 (高分化) → m_excl 高 (更严格剔除落后行业)
     std 小 (低分化) → m_excl 低 (不需要剔除)

网格:
  cap ∈ [None (无 cap), 0.20, 0.15]
  m_excl_mode ∈ ["static_0.10", "adaptive"]

输出: output/backtest_v12_dual_v5/grid_results.csv + report.md
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

OUT = ROOT / "output" / "backtest_v12_dual_v5"
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
M_EXCL_STATIC = 0.10

# 自适应 m_excl 参数: 按当日行业 mom std 划分
# std 用 (max - min) / 2 近似 (鲁棒于异常值)
ADAPTIVE_HIGH_STD = 0.15   # std > 0.15 (15%): 高分化 → m_excl=0.20
ADAPTIVE_LOW_STD = 0.06    # std < 0.06 (6%): 低分化 → m_excl=0.05
M_EXCL_HIGH = 0.20
M_EXCL_LOW = 0.05
M_EXCL_MID = 0.10


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def compute_industry_60d_momentum(daily: pd.DataFrame) -> pd.DataFrame:
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
    # 当日所有行业 mom 的离散度 (跨行业 std)
    ind_disp = ind_mom.groupby("trade_date")["industry_mom_60d"].agg(
        industry_mom_disp=lambda x: x.std()
    ).reset_index()
    ind_mom = ind_mom.merge(ind_disp, on="trade_date", how="left")
    return ind_mom


def get_adaptive_m_excl(industry_mom_disp: float) -> float:
    """按行业 momentum 分化度自适应 m_excl."""
    if pd.isna(industry_mom_disp):
        return M_EXCL_MID
    if industry_mom_disp > ADAPTIVE_HIGH_STD:
        return M_EXCL_HIGH
    if industry_mom_disp < ADAPTIVE_LOW_STD:
        return M_EXCL_LOW
    return M_EXCL_MID


def apply_industry_cap_a(a_pool: pd.DataFrame, alloc_total_a: float,
                            cap_pct: float) -> pd.DataFrame:
    """A 轨内, 单行业总仓位 ≤ cap_pct.
    超过的, 按 r5_long_rank 升序保留前 max_per_ind 只.
    """
    if a_pool.empty or cap_pct >= 1.0:
        return a_pool
    n = len(a_pool)
    per_stock = alloc_total_a / n
    max_per_ind = max(1, int(np.floor(cap_pct / per_stock)))
    a_pool = a_pool.sort_values("r5_long_rank")
    counts = {}
    keep_idx = []
    for idx, row in a_pool.iterrows():
        ind = str(row.get("industry") or "unknown")
        if counts.get(ind, 0) >= max_per_ind:
            continue
        counts[ind] = counts.get(ind, 0) + 1
        keep_idx.append(idx)
    return a_pool.loc[keep_idx]


def build_dual(daily: pd.DataFrame, ind_mom: pd.DataFrame,
                 industry_cap: float | None, m_excl_mode: str) -> pd.DataFrame:
    daily = daily.merge(ind_mom, on=["trade_date", "industry"], how="left")
    rows = []
    for d_, g in daily.groupby("trade_date"):
        if len(g) < 500: continue
        g = g.copy()

        # m_excl 选择
        if m_excl_mode.startswith("static"):
            m_excl = M_EXCL_STATIC
        else:  # adaptive
            disp = g["industry_mom_disp"].iloc[0] if "industry_mom_disp" in g.columns else None
            m_excl = get_adaptive_m_excl(disp)

        # 行业 momentum 过滤
        if m_excl > 0:
            ind_ok = g["industry_mom_60d_rank"].isna() | \
                      (g["industry_mom_60d_rank"] >= m_excl)
            g = g[ind_ok]

        if len(g) < 100: continue

        # r20 top 5%
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY_STATIC)

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
        a_pool = v7c.head(MAX_A_STOCKS).copy()

        # 行业 cap on A
        if industry_cap is not None and industry_cap < 1.0:
            a_pool = apply_industry_cap_a(a_pool, TRACK_A_PCT, industry_cap)

        # B 轨: 剩余股 (A 已 cap 出的不强制塞 B)
        b_candidates = v7c[~v7c["ts_code"].isin(a_pool["ts_code"])]
        b_pool = b_candidates.head(MAX_B_STOCKS).copy()
        # B 轨可选也加 cap
        if industry_cap is not None and industry_cap < 1.0:
            b_pool = apply_industry_cap_a(b_pool, TRACK_B_PCT, industry_cap)

        for track_tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock = alloc / len(pool)
            for _, row in pool.iterrows():
                rows.append({
                    "entry_date": d_, "ts_code": row["ts_code"], "track": track_tag,
                    "industry": row.get("industry", ""),
                    "actual_m_excl": m_excl,
                    "industry_mom_disp": float(row.get("industry_mom_disp", 0))
                        if pd.notna(row.get("industry_mom_disp")) else 0,
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
    return {
        "monthly_alpha": daily_pnl["alpha"].mean(),
        "sharpe": daily_pnl["alpha"].mean() / (daily_pnl["alpha"].std() + 1e-9) * np.sqrt(12),
        "n_hold_days": daily_pnl["entry_date"].nunique(),
        "bad_months": (monthly["alpha_avg"] < -1.0).sum(),
        "worst_month_alpha": monthly["alpha_avg"].min(),
        "monthly": monthly,
        "max_industry_alloc": None,  # 留位
    }


def diag_max_industry(hold_df: pd.DataFrame) -> dict:
    """每月最大单一行业仓位."""
    hold_df = hold_df.dropna(subset=["r20"]).copy()
    hold_df["month"] = hold_df["entry_date"].str[:6]
    by_date_industry = hold_df.groupby(["entry_date", "industry"])["alloc_pct"].sum().reset_index()
    max_by_date = by_date_industry.groupby("entry_date")["alloc_pct"].max()
    return {
        "max_industry_alloc_daily_avg": float(max_by_date.mean()),
        "max_industry_alloc_overall_max": float(max_by_date.max()),
    }


def main():
    t0 = time.time()
    print(f"\n=== V12 双轨 v5: 动态行业 cap + m_excl 自适应 ===\n", flush=True)

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
    # 看分化度分布
    disp_dist = ind_mom.drop_duplicates("trade_date")["industry_mom_disp"]
    print(f"  industry_mom_disp 分位: p25={disp_dist.quantile(0.25):.3f}, "
           f"p50={disp_dist.quantile(0.50):.3f}, p75={disp_dist.quantile(0.75):.3f}", flush=True)

    # 网格扫描 6 配置
    print(f"\n[3] 网格扫描 (cap × m_excl_mode) ...", flush=True)
    print(f"  {'config':40s} {'cov':5s} {'α':9s} {'Sharpe':8s} {'灾':3s} {'最差':7s} "
           f"{'max行业 alloc':14s}", flush=True)
    grid_rows = []
    for cap in [None, 0.20, 0.15]:
        for mode in ["static_0.10", "adaptive"]:
            label = f"cap={cap if cap else 'None'}, m_excl={mode}"
            hold = build_dual(daily, ind_mom, cap, mode)
            mt = compute_metrics(hold, daily)
            diag = diag_max_industry(hold)
            cov = mt["n_hold_days"] / daily["trade_date"].nunique() * 100
            print(f"  {label:40s} {cov:4.0f}% {mt['monthly_alpha']:+.3f}pp "
                   f"{mt['sharpe']:+.2f}    {mt['bad_months']}    "
                   f"{mt['worst_month_alpha']:+.2f}   avg/max {diag['max_industry_alloc_daily_avg']:.2f}/"
                   f"{diag['max_industry_alloc_overall_max']:.2f}",
                   flush=True)
            grid_rows.append({
                "cap": cap, "m_excl_mode": mode, "cov_pct": cov,
                "monthly_alpha_pp": mt["monthly_alpha"],
                "sharpe": mt["sharpe"],
                "bad_months": mt["bad_months"],
                "worst_month_alpha": mt["worst_month_alpha"],
                "max_ind_alloc_avg": diag["max_industry_alloc_daily_avg"],
                "max_ind_alloc_max": diag["max_industry_alloc_overall_max"],
            })

    grid_df = pd.DataFrame(grid_rows)
    grid_df = grid_df.sort_values(["sharpe", "monthly_alpha_pp"], ascending=False)
    grid_df.to_csv(OUT / "grid_results.csv", index=False)

    best = grid_df.iloc[0]
    print(f"\n--- 最佳 ---")
    print(f"  cap={best['cap']}, m_excl_mode={best['m_excl_mode']}")
    print(f"  α {best['monthly_alpha_pp']:+.3f}pp / Sharpe {best['sharpe']:+.2f}")
    print(f"  灾难月 {int(best['bad_months'])} / 最差 α {best['worst_month_alpha']:+.2f}pp")
    print(f"  最大单行业仓位 avg={best['max_ind_alloc_avg']:.2f}, max={best['max_ind_alloc_max']:.2f}")

    # 报告
    md = [f"# V12 双轨 v5: 行业 cap + m_excl 自适应\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START}-{OOS_END}\n\n",
            f"## 行业 momentum 分化度 (跨行业 std)\n\n",
            f"- p25={disp_dist.quantile(0.25):.3f}, p50={disp_dist.quantile(0.50):.3f}, p75={disp_dist.quantile(0.75):.3f}\n",
            f"- 自适应阈值: high>{ADAPTIVE_HIGH_STD} 用 m_excl={M_EXCL_HIGH}, "
            f"low<{ADAPTIVE_LOW_STD} 用 m_excl={M_EXCL_LOW}, 中间 m_excl={M_EXCL_MID}\n\n",
            "## 网格扫描 (按 Sharpe 降序)\n\n",
            "| cap | m_excl_mode | 覆盖率 % | 月化 α | Sharpe | 灾难月 | 最差月 α | max行业 avg | max行业 max |\n",
            "|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in grid_df.iterrows():
        md.append(f"| {r['cap']} | {r['m_excl_mode']} | {r['cov_pct']:.0f} | "
                   f"{r['monthly_alpha_pp']:+.3f}pp | {r['sharpe']:+.2f} | "
                   f"{int(r['bad_months'])} | {r['worst_month_alpha']:+.2f} | "
                   f"{r['max_ind_alloc_avg']:.2f} | {r['max_ind_alloc_max']:.2f} |\n")
    md.append(f"\n## v4 基准 (cap=None, m_excl=0.10)\n")
    md.append(f"α +0.88pp / Sharpe 1.02 / 0 灾难月\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
