#!/usr/bin/env python3
"""组合策略回测 — 模拟实战周频/日频持仓.

策略:
  - 每周一收盘后选 entry_score 最高的 N 只股票
  - 等权持有 5 个交易日 (周频换仓)
  - 计算累计收益, 最大回撤, 夏普比率
  - 对比基准: 等权持有沪深300 / 中证500

关键参数:
  PORTFOLIO_SIZE: 持仓数量 (5 / 10 / 20)
  REBALANCE_DAYS: 换仓周期 (5 = 周频, 1 = 日频)
  TOP_PCT: 选股池比例 (0.05 = top 5%)
  RISK_FILTER: 是否过滤高风险 (risk_score >= 70 排除)

输出:
  output/portfolio_bt/equity_curve.csv
  output/portfolio_bt/trades.csv
  output/portfolio_bt/summary.json
  output/portfolio_bt/regime_perf.csv
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS = ROOT / "output" / "labels" / "max_gain_labels.parquet"
ETF_PATH = ROOT / "output" / "etf_analysis" / "stock_to_etfs.json"
REGIME_PATH = ROOT / "output" / "regimes" / "daily_regime.parquet"
OUT_DIR = ROOT / "output" / "portfolio_bt"
OUT_DIR.mkdir(exist_ok=True)

START = "20250501"
END   = "20260126"

# 策略参数
PORTFOLIO_SIZE = 10
REBALANCE_DAYS = 5
TOP_PCT = 0.05
RISK_THRESHOLD = 70  # risk_score >= 70 排除


def load_oos_data():
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    # 真实收益
    labels = pd.read_parquet(LABELS, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left")

    # 5 日真实收益
    if "r5" not in full.columns:
        full["r5"] = 0
    full = full.dropna(subset=["r5"])

    # merge 各种 features
    for path in [
        ROOT / "output" / "regimes" / "daily_regime.parquet",
        ROOT / "output" / "amount_features" / "amount_features.parquet",
        ROOT / "output" / "regime_extra" / "regime_extra.parquet",
        ROOT / "output" / "moneyflow" / "features.parquet",
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if path.name == "daily_regime.parquet":
            d = d.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                    "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14",
                                    "vol_ratio":"mkt_vol_ratio"})
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")

    return full


def predict_uptrend(full: pd.DataFrame) -> np.ndarray:
    """批量起涨点预测."""
    booster = lgb.Booster(model_file="output/lgbm_uptrend/classifier.txt")
    meta = json.loads(Path("output/lgbm_uptrend/feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    indmap = meta["industry_map"]
    full = full.copy()
    full["industry_id"] = full["industry"].fillna("unknown").map(
        lambda x: indmap.get(str(x), -1))
    for c in feat_cols:
        if c not in full.columns: full[c] = np.nan
    return booster.predict(full[feat_cols].astype(float))


def predict_risk(full: pd.DataFrame) -> np.ndarray:
    """批量风险预测."""
    booster = lgb.Booster(model_file="output/lgbm_risk/classifier.txt")
    meta = json.loads(Path("output/lgbm_risk/feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    indmap = meta["industry_map"]
    full = full.copy()
    full["industry_id"] = full["industry"].fillna("unknown").map(
        lambda x: indmap.get(str(x), -1))
    for c in feat_cols:
        if c not in full.columns: full[c] = np.nan
    return booster.predict(full[feat_cols].astype(float))


def bucket_mv(v):
    if pd.isna(v): return None
    bil = v / 1e4
    if bil < 50: return "20-50亿"
    if bil < 100: return "50-100亿"
    if bil < 300: return "100-300亿"
    if bil < 1000: return "300-1000亿"
    return "1000亿+"


def main():
    t0 = time.time()
    print(f"加载 OOS 数据 {START} → {END}...")
    full = load_oos_data()
    print(f"OOS 样本: {len(full):,}, 日期数: {full['trade_date'].nunique()}")

    print("批量预测 uptrend + risk ...")
    full["uptrend_prob"] = predict_uptrend(full)
    full["risk_prob"] = predict_risk(full)
    full["mv_seg"] = full["total_mv"].apply(bucket_mv)
    etf_holders = set(json.loads(Path(ETF_PATH).read_text(encoding="utf-8")).keys())
    full["etf_held"] = full["ts_code"].isin(etf_holders)

    # 黑名单过滤 (与 _compute_entry_score 一致)
    BANK_INSURE = {"银行", "保险", "保险II"}
    full["is_blacklist"] = (
        full["industry"].isin(BANK_INSURE) & (full["mv_seg"].isin(["1000亿+", "300-1000亿"]))
    ) | ((full["mv_seg"] == "1000亿+") & (~full["etf_held"]))

    print(f"\n=== 组合策略回测 ===")
    print(f"持仓数: {PORTFOLIO_SIZE}, 换仓周期: {REBALANCE_DAYS}天, "
          f"风险过滤: risk_prob >= {RISK_THRESHOLD/100:.2f} 排除")

    dates = sorted(full["trade_date"].unique())
    rebalance_dates = dates[::REBALANCE_DAYS]
    print(f"换仓次数: {len(rebalance_dates)}")

    trades = []
    equity_curve = [{"date": dates[0], "equity": 1.0}]
    cur_equity = 1.0

    for i, rb_date in enumerate(rebalance_dates):
        if i + 1 >= len(rebalance_dates):
            break
        next_rb = rebalance_dates[i + 1]

        # 当日候选
        today = full[full["trade_date"] == rb_date].copy()
        # 过滤
        today = today[~today["is_blacklist"]]
        today = today[today["risk_prob"] < RISK_THRESHOLD / 100]
        today = today.dropna(subset=["uptrend_prob"])
        if len(today) < PORTFOLIO_SIZE:
            continue

        # 选 top N
        selected = today.nlargest(PORTFOLIO_SIZE, "uptrend_prob")
        sel_codes = selected["ts_code"].tolist()

        # 持有期收益: 用 next_rb 的 close / rb_date 的 close - 1
        # 简化: 用 r5 (5 日收益) 作代理
        future = full[(full["trade_date"] == rb_date) & full["ts_code"].isin(sel_codes)]
        if len(future) == 0:
            continue
        period_ret = future["r5"].mean() / 100  # r5 是百分比, 转小数
        if pd.isna(period_ret):
            continue

        # 累加
        cur_equity *= (1 + period_ret)
        equity_curve.append({"date": next_rb, "equity": cur_equity})

        # 当日 regime
        regime = today["regime"].iloc[0] if "regime" in today.columns and len(today) > 0 else None

        for _, row in selected.iterrows():
            trades.append({
                "rb_date": rb_date,
                "next_rb": next_rb,
                "ts_code": row["ts_code"],
                "industry": row.get("industry"),
                "mv_seg": row.get("mv_seg"),
                "uptrend_prob": row.get("uptrend_prob"),
                "risk_prob": row.get("risk_prob"),
                "r5": row.get("r5"),
                "max_gain_20": row.get("max_gain_20"),
                "max_dd_20": row.get("max_dd_20"),
                "regime": regime,
            })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(rebalance_dates)-1}] {rb_date} → equity {cur_equity:.4f}")

    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades)
    eq_df.to_csv(OUT_DIR / "equity_curve.csv", index=False, encoding="utf-8-sig")
    trades_df.to_csv(OUT_DIR / "trades.csv", index=False, encoding="utf-8-sig")

    # 计算指标
    final_ret = cur_equity - 1
    n_periods = len(equity_curve) - 1
    days_held = n_periods * REBALANCE_DAYS
    annualized = (cur_equity ** (252 / days_held) - 1) if days_held > 0 else 0

    # 周频 returns
    eq_df["ret"] = eq_df["equity"].pct_change()
    weekly_rets = eq_df["ret"].dropna().values
    sharpe = (weekly_rets.mean() / weekly_rets.std() * np.sqrt(52)) if weekly_rets.std() > 0 else 0

    # max drawdown
    eq = eq_df["equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # 胜率
    win_rate = (weekly_rets > 0).mean() * 100

    print(f"\n{'='*60}")
    print(f"=== 组合回测结果 ===")
    print(f"{'='*60}")
    print(f"测试期: {START} → {END}")
    print(f"换仓次数: {n_periods}")
    print(f"最终累计收益: {final_ret*100:+.2f}%")
    print(f"年化收益:     {annualized*100:+.2f}%")
    print(f"夏普比率:     {sharpe:.2f}")
    print(f"最大回撤:     {max_dd*100:.2f}%")
    print(f"周频胜率:     {win_rate:.1f}%")

    # 按 regime 表现
    if "regime" in trades_df.columns:
        rg = trades_df.groupby(["rb_date", "regime"], observed=True).agg(
            avg_r5=("r5","mean"),
            n=("ts_code","count")
        ).reset_index()
        rg2 = rg.groupby("regime", observed=True).agg(
            n_periods=("rb_date", "count"),
            avg_period_ret=("avg_r5", "mean"),
            best=("avg_r5", "max"),
            worst=("avg_r5", "min"),
        ).round(3)
        print(f"\n=== 按 regime 表现 ===")
        print(rg2.to_string())
        rg2.to_csv(OUT_DIR / "regime_perf.csv", encoding="utf-8-sig")

    # 按 mv_seg 表现
    if "mv_seg" in trades_df.columns:
        mvp = trades_df.groupby("mv_seg", observed=True).agg(
            n=("ts_code","count"),
            avg_r5=("r5","mean"),
            avg_max_gain=("max_gain_20","mean"),
        ).round(3)
        print(f"\n=== 按 MV 段表现 ===")
        print(mvp.to_string())

    summary = {
        "params": {
            "portfolio_size": PORTFOLIO_SIZE,
            "rebalance_days": REBALANCE_DAYS,
            "risk_threshold": RISK_THRESHOLD / 100,
            "test_period": [START, END],
        },
        "n_periods": n_periods,
        "n_trades": len(trades_df),
        "final_return_pct": float(final_ret * 100),
        "annualized_return_pct": float(annualized * 100),
        "sharpe": float(sharpe),
        "max_dd_pct": float(max_dd * 100),
        "win_rate_pct": float(win_rate),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {OUT_DIR}")
    print(f"耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
