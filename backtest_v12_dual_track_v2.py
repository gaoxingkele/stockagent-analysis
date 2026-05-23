"""V12 双轨架构 v2 (修复锚点漂移): 用 r20_pred pct rank 替代 buy_score [70,85].

旧 V7c 5 铁律: buy_score >= 70 是固定阈值, OOS 期市场分布漂移 → 大部分日子空池.
v2 修复: buy_score 用当日 r20_pred top N% 排名 (相对值, 抗漂移).

新铁律:
  1. r20_pred top P_BUY% (默认 10%, 替代 buy_score [70,85])
  2. pyr_velocity_20_60 < p35
  3. |f1_neg1| < 0.005, |f2_pos1| < 0.005
  4. R5 反向: 在 V7c 池内 pred_r5_long_nost Bot 15%/35%

输出: output/backtest_v12_dual_v2/report.md + monthly_pnl.csv
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

OUT = ROOT / "output" / "backtest_v12_dual_v2"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

OOS_START = "20251001"
OOS_END = "20260331"
HOLD_DAYS = 20
COST_BPS = 35.0 / 10000

# 修复: 相对排名 (跳过锚点)
P_BUY_TOP = 0.10                  # r20_pred top 10% 替代 buy_score >= 70
USE_PYR_VELOCITY = True
USE_F1F2 = False                  # f1/f2 太严, 暂时关掉看效果
PYR_VELOCITY_QUANTILE = 0.35
F1_F2_THRESHOLD = 0.005

R5_REVERSE_BOTTOM_A = 0.15
R5_REVERSE_BOTTOM_B = 0.35
TRACK_A_PCT = 0.70
TRACK_B_PCT = 0.20
CASH_PCT = 0.10
MAX_A_STOCKS = 15
MAX_B_STOCKS = 25


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def main():
    t0 = time.time()
    print(f"\n=== V12 双轨 v2 (相对排名修复) ===\n", flush=True)
    print(f"P_BUY_TOP={P_BUY_TOP}, USE_PYR={USE_PYR_VELOCITY}, USE_F1F2={USE_F1F2}", flush=True)

    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")

    # 推理 (用真 OOS 模型 _long_nost!)
    print("[2] 推理 _long_nost (真 OOS 模型) ...", flush=True)
    for name in ["r10_v16_long_nost", "r20_v16_long_nost", "r5_v17_long_nost"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily[f"pred_{name}"] = b.predict(X)
        print(f"  {name}: pred μ={daily[f'pred_{name}'].mean():+.3f}", flush=True)

    # 5 铁律 v2 (相对)
    print("[3] 每日双轨构建 ...", flush=True)
    days = sorted(daily["trade_date"].unique())
    holdings_log = []

    for d_ in days:
        g = daily[daily["trade_date"] == d_].copy()
        if len(g) < 500: continue

        # 1. r20_pred top P_BUY%
        n_buy_top = max(1, int(len(g) * P_BUY_TOP))
        g["r20_rank"] = g["pred_r20_v16_long_nost"].rank(pct=True, method="first")
        m_buy = g["r20_rank"] >= (1 - P_BUY_TOP)

        # 2. pyr_velocity
        if USE_PYR_VELOCITY and "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(PYR_VELOCITY_QUANTILE)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)

        # 3. f1/f2
        if USE_F1F2 and "f1_neg1" in g.columns and "f2_pos1" in g.columns:
            m_f12 = (g["f1_neg1"].abs() < F1_F2_THRESHOLD) & (g["f2_pos1"].abs() < F1_F2_THRESHOLD)
        else:
            m_f12 = pd.Series(True, index=g.index)

        g["v7c_recommend"] = m_buy & m_pyr & m_f12
        v7c = g[g["v7c_recommend"]].copy()
        if len(v7c) == 0: continue

        # R5 反向
        v7c["r5_long_rank"] = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
        v7c = v7c.sort_values("r5_long_rank")
        a_pool = v7c[v7c["r5_long_rank"] < R5_REVERSE_BOTTOM_A].head(MAX_A_STOCKS)
        b_pool = v7c[(v7c["r5_long_rank"] >= R5_REVERSE_BOTTOM_A) &
                       (v7c["r5_long_rank"] < R5_REVERSE_BOTTOM_B)].head(MAX_B_STOCKS)

        for track_tag, pool, alloc in [("A", a_pool, TRACK_A_PCT), ("B", b_pool, TRACK_B_PCT)]:
            if len(pool) == 0: continue
            per_stock = alloc / len(pool)
            for _, row in pool.iterrows():
                holdings_log.append({
                    "entry_date": d_, "ts_code": row["ts_code"], "track": track_tag,
                    "industry": row.get("industry", ""),
                    "r20_rank": float(row["r20_rank"]),
                    "r5_long_rank": float(row["r5_long_rank"]),
                    "r20": float(row["r20"]) if pd.notna(row.get("r20")) else np.nan,
                    "alloc_pct": per_stock,
                })

    hold_df = pd.DataFrame(holdings_log)
    hold_df.to_csv(OUT / "holdings_log.csv", index=False)
    print(f"  持仓记录: {len(hold_df):,}, 日数 {hold_df['entry_date'].nunique()}", flush=True)

    # PnL
    hold_df = hold_df.dropna(subset=["r20"]).copy()
    hold_df["r20"] = hold_df["r20"].clip(-30, 30)
    hold_df["weighted_r20"] = hold_df["alloc_pct"] * hold_df["r20"]

    total_alloc = TRACK_A_PCT + TRACK_B_PCT
    daily_pnl = hold_df.groupby("entry_date").agg(
        port_r20_gross=("weighted_r20", "sum"),
        n_stocks=("ts_code", "count"),
    ).reset_index()
    daily_pnl["port_r20_net"] = daily_pnl["port_r20_gross"] - total_alloc * COST_BPS * 200

    mkt = daily.groupby("trade_date")["r20"].apply(lambda x: x.clip(-30, 30).mean()).reset_index()
    mkt.columns = ["entry_date", "mkt_r20"]
    daily_pnl = daily_pnl.merge(mkt, on="entry_date", how="left")
    daily_pnl["alpha"] = daily_pnl["port_r20_net"] - daily_pnl["mkt_r20"] * total_alloc

    daily_pnl["month"] = daily_pnl["entry_date"].str[:6]
    monthly = daily_pnl.groupby("month").agg(
        n_days=("entry_date", "count"),
        avg_n_stocks=("n_stocks", "mean"),
        port_avg=("port_r20_net", "mean"),
        mkt_avg=("mkt_r20", "mean"),
        alpha_avg=("alpha", "mean"),
        alpha_std=("alpha", "std"),
    ).reset_index()
    monthly["sharpe"] = monthly["alpha_avg"] / (monthly["alpha_std"] + 1e-9) * np.sqrt(20)
    monthly.to_csv(OUT / "monthly_pnl.csv", index=False)

    avg_net = daily_pnl["port_r20_net"].mean()
    avg_mkt_scaled = daily_pnl["mkt_r20"].mean() * total_alloc
    monthly_alpha = avg_net - avg_mkt_scaled
    sharpe = daily_pnl["alpha"].mean() / (daily_pnl["alpha"].std() + 1e-9) * np.sqrt(12)

    print(f"\n--- 整体 ---")
    print(f"  组合 20 日净均: {avg_net:+.3f}%/期 ({avg_net:+.3f}%/月)")
    print(f"  市场 (按 alloc 缩放): {avg_mkt_scaled:+.3f}%/期")
    print(f"  月化 α: {monthly_alpha:+.3f}pp")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"\n--- 月度 ---")
    for _, r in monthly.iterrows():
        flag = " [BAD]" if r["alpha_avg"] < -1.0 else ""
        print(f"  {r['month']}: n={int(r['n_days']):3d} 股={r['avg_n_stocks']:5.1f} "
               f"组合={r['port_avg']:+.2f}% 市场={r['mkt_avg']:+.2f}% "
               f"α={r['alpha_avg']:+.2f}pp Sharpe={r['sharpe']:+.2f}{flag}")

    # 持仓覆盖率
    n_oos_days = len(daily["trade_date"].unique())
    n_hold_days = daily_pnl["entry_date"].nunique()
    print(f"\n持仓覆盖率: {n_hold_days}/{n_oos_days} = {n_hold_days/n_oos_days*100:.0f}%")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
