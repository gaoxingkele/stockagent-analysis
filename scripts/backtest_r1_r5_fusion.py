"""r1 ∩ R5 融合实验 (长 OOS 142 日).

假设: r1 Top 池 ∩ R5 Bottom 池 = 隔夜强 + 短期超跌, alpha 是否相乘?

实验设计:
  - 候选: factors_v3 EOD bar (hour=15), OOS 20251001 至 20260331
  - r1_pred:  r1_next_open_v3_long      (T+1 隔夜信号)
  - r5_pred:  r5_v17_long               (5 日反向信号, 越低越好)

  四个对照组 (各日按 pct_rank):
    A. r1_top10% & r5_bot30%   ← 主测假设 (强隔夜 + 短期超跌)
    B. r1_top10% only
    C. r5_bot30% only
    D. 全市场 (基准)

收益: r1_next_open (capped ±3%) 隔夜
退出: 次日 09:30 (T+1, 一日持有)

输出: output/backtest_fusion/report.md + results.csv
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROD = ROOT / "output" / "production"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
OUT = ROOT / "output" / "backtest_fusion"
OUT.mkdir(parents=True, exist_ok=True)

OOS_START = "20251001"
OOS_END = "20260331"
COST_BPS = 35.0 / 10000
R1_CAP = 3.0
DIST_THRESHOLD = 3.0


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m


def metric(curve: pd.DataFrame, name: str) -> dict:
    if curve.empty:
        return {"strategy": name, "n_days": 0}
    curve = curve.sort_values("date")
    curve["nav_net"] = (1 + curve["ret_net_pct"] / 100).cumprod()
    curve["nav_mkt"] = (1 + curve["ret_mkt_pct"] / 100).cumprod()
    curve["dd"] = curve["nav_net"] / curve["nav_net"].cummax() - 1
    n_days = len(curve)
    monthly = curve["ret_net_pct"].mean() * 20
    std = curve["ret_net_pct"].std() + 1e-9
    sharpe = curve["ret_net_pct"].mean() / std * np.sqrt(252)
    return {
        "strategy": name, "n_days": n_days,
        "total_net_pct": (curve["nav_net"].iloc[-1] - 1) * 100,
        "total_mkt_pct": (curve["nav_mkt"].iloc[-1] - 1) * 100,
        "alpha_total_pct": (curve["nav_net"].iloc[-1] - curve["nav_mkt"].iloc[-1]) * 100,
        "monthly_net_pct": monthly,
        "sharpe": sharpe,
        "mdd_pct": curve["dd"].min() * 100,
        "win_rate_alpha": (curve["alpha_pct"] > 0).mean(),
        "pool_avg": curve["n_pool"].mean(),
    }


def main():
    t0 = time.time()
    print(f"\n=== r1 ∩ R5 融合回测 ({OOS_START}-{OOS_END}) ===\n", flush=True)

    # 1. 1H factor + r1 推理
    print(f"[1] 加载 factors_v3 ...", flush=True)
    df1h = pd.read_parquet(F3)
    df1h["trade_time"] = pd.to_datetime(df1h["trade_time"])
    df1h["trade_date"] = df1h["trade_date"].astype(str)
    eod = df1h[(df1h["trade_time"].dt.hour == 15) &
                 (df1h["trade_date"] >= OOS_START) &
                 (df1h["trade_date"] <= OOS_END)].copy()
    del df1h
    print(f"   EOD bar: {len(eod):,}, 日数 {eod['trade_date'].nunique()}", flush=True)

    print(f"\n[2] r1_next_open_v3_long 推理 ...", flush=True)
    b1, fc1, _ = load_model("r1_next_open_v3_long")
    for c in fc1:
        if c not in eod.columns: eod[c] = 0.0
        eod[c] = eod[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    X1 = eod[fc1].astype("float32")
    eod["pred_r1"] = b1.predict(X1)
    print(f"   r1 pred 均值 {eod['pred_r1'].mean():+.3f}", flush=True)

    # 2. 日线 factor + r5 推理
    print(f"\n[3] 日线 factor + r5_v17_long 推理 ...", flush=True)
    from train_v15_refresh import load_window
    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    meta_r5 = json.loads((PROD / "r5_v17_long" / "feature_meta.json").read_text(encoding="utf-8"))
    ind_map = meta_r5.get("industry_map", {})
    if "industry" in daily.columns:
        daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    else:
        daily["industry_id"] = -1
    b5, fc5, _ = load_model("r5_v17_long")
    miss = [c for c in fc5 if c not in daily.columns]
    for c in miss: daily[c] = 0.0
    X5 = daily[fc5].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    daily["pred_r5"] = b5.predict(X5)
    daily_preds = daily[["ts_code", "trade_date", "pred_r5"]].copy()
    del daily

    # 3. merge
    print(f"\n[4] merge r1 + r5 ...", flush=True)
    m = eod.merge(daily_preds, on=["ts_code", "trade_date"], how="inner")
    print(f"   合并: {len(m):,}, 日数 {m['trade_date'].nunique()}", flush=True)

    # ST 过滤 (实盘必须排除, ST 股波动 alpha 在 r1 模型上有偏见)
    EXCLUDE_ST = True  # 关键开关: 与 backtest_t1_real_long 不一致的地方
    if BASIC_P.exists():
        basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
        m = m.merge(basic, on="ts_code", how="left")
        m["is_st"] = m["name"].fillna("").str.contains("ST", regex=False)
    else:
        m["is_st"] = False

    m["is_near_upper"] = m["dist_to_upper_limit_pct"].fillna(100) < DIST_THRESHOLD
    m["is_at_lower"] = m["dist_to_lower_limit_pct"].fillna(100) < 1.0
    m["bad_filter"] = m["is_near_upper"] | m["is_at_lower"]
    if EXCLUDE_ST:
        m["bad_filter"] = m["bad_filter"] | m["is_st"]

    valid = m.dropna(subset=["pred_r1", "pred_r5", "r1_next_open"]).copy()
    valid = valid[valid["r1_next_open"].abs() <= 20]   # 排除异常
    valid = valid[~valid["bad_filter"]]
    valid["r1_capped"] = valid["r1_next_open"].clip(-R1_CAP, R1_CAP)
    st_flag = " + ST" if EXCLUDE_ST else " (含 ST!)"
    print(f"   有效 bar: {len(valid):,} (排除涨跌停{st_flag})", flush=True)

    # 4. 每日构建对照组 (精确选股可比, 池大小固定 = 10)
    # 6 组, 每组 10 股:
    #   r1_top10           : 基准 (≈ daily_r1_recommend 实战)
    #   r1_top50_r5_bot10  : 融合 = r1 强 → r5 升序 Top 10 (r5 弱, 假设增益)
    #   r1_top50_r5_top10  : 反例 = r1 强 → r5 降序 Top 10 (r5 强, 验证方向)
    #   r1_top100_r5_bot10 : 融合宽 = r1 较强 → r5 升序 Top 10
    #   r5_bot10           : r5 单独 Bot 10
    #   mkt                : 全市场基准
    print(f"\n[5] 6 对照组逐日构建 (固定 10 股/组) ...", flush=True)
    days = sorted(valid["trade_date"].unique())
    rows_top10, rows_fusion50, rows_anti50, rows_fusion100, rows_r5b10, rows_mkt = (
        [], [], [], [], [], [])

    for d_ in days:
        g = valid[valid["trade_date"] == d_].copy()
        if len(g) < 300: continue
        mkt_r1 = g["r1_capped"].mean()

        def add(rows, sub):
            if len(sub) == 0: return
            ret = sub["r1_capped"].mean() - COST_BPS * 100
            rows.append({"date": d_, "n_pool": len(sub),
                          "ret_net_pct": ret, "ret_mkt_pct": mkt_r1,
                          "alpha_pct": ret - mkt_r1})

        # 基准 r1 Top 10
        add(rows_top10, g.nlargest(10, "pred_r1"))

        # r1 Top 50 → r5 最低 10 (融合)
        anchor50 = g.nlargest(50, "pred_r1")
        add(rows_fusion50, anchor50.nsmallest(10, "pred_r5"))
        # r1 Top 50 → r5 最高 10 (反例)
        add(rows_anti50, anchor50.nlargest(10, "pred_r5"))

        # r1 Top 100 → r5 最低 10 (融合宽)
        anchor100 = g.nlargest(100, "pred_r1")
        add(rows_fusion100, anchor100.nsmallest(10, "pred_r5"))

        # r5 Bot 10 (单独)
        add(rows_r5b10, g.nsmallest(10, "pred_r5"))

        # mkt baseline
        rows_mkt.append({"date": d_, "n_pool": len(g),
                          "ret_net_pct": mkt_r1, "ret_mkt_pct": mkt_r1,
                          "alpha_pct": 0})

    summary = []
    for label, rows in [("A_r1top10_baseline", rows_top10),
                          ("B_r1top50_r5bot10_FUSION", rows_fusion50),
                          ("C_r1top50_r5top10_anti", rows_anti50),
                          ("D_r1top100_r5bot10_fusion_wide", rows_fusion100),
                          ("E_r5bot10_only", rows_r5b10),
                          ("F_mkt_baseline", rows_mkt)]:
        cv = pd.DataFrame(rows)
        if not cv.empty:
            cv = cv.sort_values("date").reset_index(drop=True)
            cv.to_csv(OUT / f"curve_{label}.csv", index=False)
        summary.append(metric(cv, label))

    res_df = pd.DataFrame(summary)
    res_df.to_csv(OUT / "results.csv", index=False)

    # 报告
    md = [f"# r1 ∩ R5 融合实验报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START} 至 {OOS_END} (~142 日)\n",
            f"模型: r1_next_open_v3_long (IC 0.77) + r5_v17_long (反向)\n",
            f"约束: cap r1 ±{R1_CAP}%, 成本 {COST_BPS*100:.2f}%, ST/涨跌停过滤\n\n",
            "## 4 对照组净值\n\n",
            "| 策略 | 日数 | 池均 | 净 % | 市场 % | α 累计 % | 月化净 % | Sharpe | MDD % | α 胜率 |\n",
            "|---|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in res_df.iterrows():
        md.append(f"| {r['strategy']} | {int(r['n_days'])} | {r['pool_avg']:.0f} | "
                   f"{r['total_net_pct']:+.1f} | {r['total_mkt_pct']:+.1f} | "
                   f"{r['alpha_total_pct']:+.1f} | {r['monthly_net_pct']:+.2f} | "
                   f"{r['sharpe']:.2f} | {r['mdd_pct']:.1f} | "
                   f"{r['win_rate_alpha']*100:.0f}% |\n")

    # 假设结论
    a = res_df[res_df["strategy"] == "A_r1top10_baseline"].iloc[0]
    b_ = res_df[res_df["strategy"] == "B_r1top50_r5bot10_FUSION"].iloc[0]
    c_ = res_df[res_df["strategy"] == "C_r1top50_r5top10_anti"].iloc[0]
    md.append(f"\n## 假设验证 (10 股/日 固定池)\n\n")
    md.append(f"- 基准 A r1Top10: Sharpe {a['sharpe']:.2f}, 月化 {a['monthly_net_pct']:+.2f}%\n")
    md.append(f"- 融合 B r1Top50→r5Bot10: Sharpe {b_['sharpe']:.2f}, 月化 {b_['monthly_net_pct']:+.2f}%\n")
    md.append(f"- 反例 C r1Top50→r5Top10: Sharpe {c_['sharpe']:.2f}, 月化 {c_['monthly_net_pct']:+.2f}%\n")
    sharpe_delta = b_['sharpe'] - a['sharpe']
    monthly_delta = b_['monthly_net_pct'] - a['monthly_net_pct']
    direction = b_['sharpe'] - c_['sharpe']
    md.append(f"- 融合 vs 基准: Sharpe Δ={sharpe_delta:+.2f}, 月化 Δ={monthly_delta:+.2f}pp\n")
    md.append(f"- 融合 vs 反例 (方向验证): Sharpe Δ={direction:+.2f}\n")
    if sharpe_delta > 1.0 and direction > 1.0:
        md.append(f"- **结论**: 假设成立, r5 反向过滤显著增强 r1\n")
    elif sharpe_delta < -1.0 or direction < -1.0:
        md.append(f"- **结论**: 假设证伪, 融合反而弱化\n")
    else:
        md.append(f"- **结论**: 差异不显著, r1 单用即可\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    for _, r in res_df.iterrows():
        print(f"  {r['strategy']:<22s}: n={int(r['n_days']):3d} 池={r['pool_avg']:5.0f} "
               f"月化={r['monthly_net_pct']:+5.2f}% Sharpe={r['sharpe']:5.2f} "
               f"MDD={r['mdd_pct']:5.1f}% αwin={r['win_rate_alpha']*100:3.0f}%")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
