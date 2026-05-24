"""202602 灾难月诊断: 为什么 v10a pump_score 在 2 月 -1.47pp 翻车?

对比 v6 vs v10a 在 202602 的:
  1. 持仓股票差异 (谁选了谁亏了)
  2. pump_score 分布 (是否整体偏低 = 模型失效)
  3. 行业分布 (是否押错行业)
  4. 全市场 pump_score 与实际启动子的相关性 (rank IC)
  5. 跟其他月 (202601 / 202603) 对比, 找 regime 特征

输出: output/diag_202602_pump/report.md
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

OUT = ROOT / "output" / "diag_202602_pump"
OUT.mkdir(parents=True, exist_ok=True)
PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"

DATA_START = "20251201"  # 含 12 月 + 1-4 月做对比
DATA_END = "20260515"
FOCUS_MONTH = "202602"
COMPARE_MONTHS = ["202601", "202602", "202603", "202604"]


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def main():
    t0 = time.time()
    print(f"\n=== 202602 灾难月诊断 ===\n", flush=True)

    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")
    daily["month"] = daily["trade_date"].str[:6]

    # 推理: pump + r5_long + r20_long
    for name in ["r20_v16_long_nost", "r5_v17_long_nost", "r5_pump_lgbm_v1"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        if name == "r5_pump_lgbm_v1":
            daily["pump_score"] = 1 / (1 + np.exp(-b.predict(X)))
        else:
            daily[f"pred_{name}"] = b.predict(X)
    print(f"  推理完成 {time.time()-t0:.0f}s", flush=True)

    # 从 daily cache 加载 OHLCV (load_window 不含)
    print(f"  从 daily cache 加 OHLCV ...", flush=True)
    ddir = ROOT / "output/tushare_cache/daily"
    files = sorted(ddir.glob("*.parquet"))
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high", "low", "close"])
                for f in files]
    ohlcv = pd.concat(parts, ignore_index=True)
    ohlcv["trade_date"] = ohlcv["trade_date"].astype(str)
    ohlcv = ohlcv.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    ohlcv["next_open"] = ohlcv.groupby("ts_code")["open"].shift(-1)
    ohlcv["max_high_5"] = (ohlcv.groupby("ts_code")["high"].apply(
        lambda x: x.rolling(5, min_periods=5).max().shift(-5)).reset_index(level=0, drop=True))
    ohlcv["min_low_5"] = (ohlcv.groupby("ts_code")["low"].apply(
        lambda x: x.rolling(5, min_periods=5).min().shift(-5)).reset_index(level=0, drop=True))
    ohlcv["close_20d"] = ohlcv.groupby("ts_code")["close"].shift(-20)
    ohlcv["upside_5"] = ohlcv["max_high_5"] / ohlcv["next_open"] - 1
    ohlcv["downside_5"] = ohlcv["min_low_5"] / ohlcv["next_open"] - 1
    ohlcv["is_pump"] = ((ohlcv["upside_5"] >= 0.10) & (ohlcv["downside_5"] >= -0.05)).astype(int)
    ohlcv["r20_fresh"] = (ohlcv["close_20d"] / ohlcv["next_open"] - 1) * 100

    daily = daily.merge(ohlcv[["ts_code", "trade_date", "is_pump", "r20_fresh",
                                  "upside_5", "downside_5"]],
                          on=["ts_code", "trade_date"], how="left")

    print(f"\n## 1. 跨月对比 (pump_score 分布 + 实际 pump 率)\n", flush=True)
    print(f"  {'month':10s} {'n_bar':8s} {'pump_score 均/std':22s} {'实际 pump 率':12s} "
           f"{'mkt r20 均':12s} {'V7c 池均 r20':14s}", flush=True)

    # 简化 V7c 选股 (top 5% r20)
    monthly_diag = []
    for m_ in COMPARE_MONTHS:
        sub = daily[daily["month"] == m_]
        if sub.empty: continue
        # V7c 池: r20_pred top 5%
        sub_grp = sub.copy()
        sub_grp["r20_rank"] = sub_grp.groupby("trade_date")["pred_r20_v16_long_nost"].rank(
            pct=True, method="first")
        v7c = sub_grp[sub_grp["r20_rank"] >= 0.95]
        # mkt r20
        mkt_r20 = sub.groupby("trade_date")["r20_fresh"].apply(
            lambda x: x.clip(-30, 30).mean()).mean()
        v7c_r20 = v7c["r20_fresh"].clip(-30, 30).mean()
        info = {
            "month": m_,
            "n_bar": len(sub),
            "pump_mean": sub["pump_score"].mean(),
            "pump_std": sub["pump_score"].std(),
            "pump_p95": sub["pump_score"].quantile(0.95),
            "real_pump_rate": sub["is_pump"].mean(),
            "mkt_r20": mkt_r20,
            "v7c_r20": v7c_r20,
        }
        monthly_diag.append(info)
        print(f"  {m_:10s} {len(sub):>7,d} "
               f"{info['pump_mean']:.3f}/{info['pump_std']:.3f}{'':10s} "
               f"{info['real_pump_rate']*100:5.1f}%       "
               f"{mkt_r20:+6.2f}%      {v7c_r20:+6.2f}%", flush=True)

    print(f"\n## 2. 202602 pump_score Top 推荐池 vs 实际表现\n", flush=True)
    feb = daily[daily["month"] == "202602"].copy()
    # 模拟 v10a 选 A 轨 (V7c top 5% × pyr × f1f2 后 top pump 8 股)
    feb["r20_rank"] = feb.groupby("trade_date")["pred_r20_v16_long_nost"].rank(pct=True, method="first")
    feb_v7c = feb[feb["r20_rank"] >= 0.95]
    if "pyr_velocity_20_60" in feb_v7c.columns:
        feb_v7c = feb_v7c[feb_v7c.groupby("trade_date")["pyr_velocity_20_60"].transform(
            lambda x: x < x.quantile(0.35))]

    # 每日 v10 A 轨 (pump top 8)
    daily_a_rows = []
    for d_, g in feb_v7c.groupby("trade_date"):
        if len(g) == 0: continue
        a = g.nlargest(min(8, len(g)), "pump_score")
        for _, r in a.iterrows():
            daily_a_rows.append({"date": d_, "ts_code": r["ts_code"],
                                   "industry": r.get("industry", ""),
                                   "pump_score": r["pump_score"],
                                   "r5_long_rank": r["pred_r5_v17_long_nost"],
                                   "r20_fresh": r["r20_fresh"],
                                   "is_pump_real": r["is_pump"]})
    a_df = pd.DataFrame(daily_a_rows)
    print(f"  202602 v10a A 轨持仓总数: {len(a_df)}", flush=True)
    print(f"  A 轨 r20 均: {a_df['r20_fresh'].clip(-30,30).mean():+.2f}%", flush=True)
    print(f"  A 轨 实际 pump 命中率: {a_df['is_pump_real'].mean()*100:.1f}%", flush=True)

    # 行业分布
    print(f"\n## 3. 202602 v10 A 轨行业 vs r20 表现\n", flush=True)
    ind_perf = a_df.groupby("industry").agg(
        n=("ts_code", "count"),
        r20_mean=("r20_fresh", lambda x: x.clip(-30, 30).mean()),
        pump_hit=("is_pump_real", "mean"),
    ).reset_index().sort_values("r20_mean")
    print(f"  Worst 5 行业 (拖累):")
    for _, r in ind_perf.head(5).iterrows():
        print(f"    {r['industry']:20s} n={r['n']:3d} r20={r['r20_mean']:+6.2f}% "
               f"pump_hit={r['pump_hit']*100:5.1f}%", flush=True)
    print(f"\n  Best 5 行业 (拉升):")
    for _, r in ind_perf.tail(5).iterrows():
        print(f"    {r['industry']:20s} n={r['n']:3d} r20={r['r20_mean']:+6.2f}% "
               f"pump_hit={r['pump_hit']*100:5.1f}%", flush=True)

    # 跨月 pump_score 与实际 pump 的相关性 (rank IC)
    print(f"\n## 4. pump_score 跨月 rank IC (信号有效性)\n", flush=True)
    for m_ in COMPARE_MONTHS:
        sub = daily[daily["month"] == m_].dropna(subset=["pump_score", "is_pump"])
        if len(sub) < 100: continue
        # 全月 IC (pump_score vs is_pump 0/1)
        ic = sub["pump_score"].corr(sub["is_pump"])
        # 同时算 daily 内 IC, 然后平均
        daily_ics = []
        for d_, g in sub.groupby("trade_date"):
            if len(g) < 50: continue
            if g["is_pump"].sum() == 0: continue
            ic_d = g["pump_score"].corr(g["is_pump"])
            daily_ics.append(ic_d)
        avg_daily_ic = np.mean(daily_ics) if daily_ics else 0
        print(f"  {m_}: pump_score vs is_pump 月度 IC = {ic:.4f}, "
               f"日均 IC = {avg_daily_ic:.4f}", flush=True)

    # market regime 描述
    print(f"\n## 5. 202602 vs 其他月 市场 regime\n", flush=True)
    print(f"  {'month':10s} {'mkt 5d ret 均':14s} {'mkt 20d ret 均':14s} "
           f"{'pump_score 均':14s} {'pump 率(真实)':14s}", flush=True)
    for m_ in COMPARE_MONTHS:
        sub = daily[daily["month"] == m_]
        if sub.empty: continue
        # 简化: 用当月所有 ts_code 的 r5/r20 均值代表 regime
        # 但 r5/r20 是 forward, 这里取 month 内每日的 mkt_ret
        mkt5_avg = sub["upside_5"].clip(-0.5, 0.5).mean() * 100   # 5日 max upside 均
        mkt20_avg = sub["r20_fresh"].clip(-30, 30).mean()
        pump_avg = sub["pump_score"].mean()
        pump_real = sub["is_pump"].mean() * 100
        print(f"  {m_:10s} {mkt5_avg:+6.2f}%       {mkt20_avg:+6.2f}%       "
               f"{pump_avg:.3f}         {pump_real:5.2f}%", flush=True)

    pd.DataFrame(monthly_diag).to_csv(OUT / "monthly_compare.csv", index=False)
    a_df.to_csv(OUT / "v10a_holdings_202602.csv", index=False)
    ind_perf.to_csv(OUT / "v10a_industry_202602.csv", index=False)
    print(f"\n输出: {OUT}/")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
