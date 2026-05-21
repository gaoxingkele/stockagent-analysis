"""P0 长 OOS T+1 真实回测 (用 r1_next_open_v3_long).

vs backtest_t1_real.py:
  - 模型: r1_next_open_v3_long (严格 cut at 20250930)
  - OOS: 20251001 至 20260331 (142 日)
  - cap r1 at +3.0% (更严格的开盘集合竞价滑点真实模拟)
  - 涨跌停 dist 阈值 < 3.0% 排除 (更严格)
  - 成本 0.35%

输出: output/backtest_t1_long/report.md
"""
from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
PROD = ROOT / "output" / "production"
OUT = ROOT / "output" / "backtest_t1_long"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "curves").mkdir(exist_ok=True)

OOS_START = "20251001"
OOS_END = "20260331"
MODEL_NAME = "r1_next_open_v3_long"

COST_BPS = 35.0 / 10000  # 0.35%
R1_CAP = 3.0  # 严格: 集合竞价实际买入价上限 +3%
DIST_THRESHOLD = 3.0  # 距涨停 < 3% 排除

TOP_NS = [5, 10, 20, 50, 100]
HOURS = [10, 11, 13, 14, 15]


def load_model():
    d = PROD / MODEL_NAME
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return booster, meta["feature_cols"]


def main():
    t0 = time.time()
    print(f"\n=== P0 长 OOS T+1 真实回测 ({OOS_START}-{OOS_END}) ===\n")
    print(f"模型: {MODEL_NAME}")
    print(f"r1 cap: ±{R1_CAP}% (严格集合竞价滑点)")
    print(f"涨跌停过滤: dist < {DIST_THRESHOLD}%")
    print(f"成本: 单次 {COST_BPS*100:.2f}%\n")

    print(f"加载 factors_v3...", flush=True)
    df = pd.read_parquet(F3)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"  全量: {len(df):,} 行", flush=True)

    print(f"OOS 切片...", flush=True)
    oos = df[(df["trade_date"] >= OOS_START) & (df["trade_date"] <= OOS_END)].copy()
    print(f"  OOS bar: {len(oos):,}, 日数: {oos['trade_date'].nunique()}", flush=True)
    del df

    print(f"模型推理...", flush=True)
    booster, feat_cols = load_model()
    for c in feat_cols:
        if c not in oos.columns: oos[c] = 0.0
        oos[c] = oos[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    X = oos[feat_cols].astype("float32")
    oos["pred_r1"] = booster.predict(X)
    print(f"  推理完成 {time.time()-t0:.0f}s, pred 均值 {oos['pred_r1'].mean():+.3f}", flush=True)

    # 过滤
    print(f"应用过滤 (严格)...", flush=True)
    oos["is_near_upper"] = oos["dist_to_upper_limit_pct"].fillna(100) < DIST_THRESHOLD
    oos["is_at_lower"] = oos["dist_to_lower_limit_pct"].fillna(100) < 1.0
    oos["bad_filter"] = oos["is_near_upper"] | oos["is_at_lower"]
    print(f"  涨/跌停接近: {oos['bad_filter'].sum():,} ({oos['bad_filter'].mean()*100:.1f}%)", flush=True)

    oos_valid = oos.dropna(subset=["r1_next_open", "pred_r1"])
    oos_valid = oos_valid[oos_valid["r1_next_open"].abs() <= 20]
    oos_valid["r1_capped"] = oos_valid["r1_next_open"].clip(-R1_CAP, R1_CAP)
    print(f"  有效 bar: {len(oos_valid):,}", flush=True)
    print(f"  r1 capped (|r1|>{R1_CAP}%): {(oos_valid['r1_next_open'].abs() > R1_CAP).sum():,}", flush=True)

    # 主回测
    print(f"\n开始 25 组合回测...", flush=True)
    results = []
    for hour in HOURS:
        bar_at_h = oos_valid[oos_valid["trade_time"].dt.hour == hour].copy()
        if len(bar_at_h) == 0: continue
        for top_n in TOP_NS:
            curve = []
            for d_, g in bar_at_h.groupby("trade_date"):
                gf = g[~g["bad_filter"]]
                if len(gf) < top_n: continue
                top = gf.nlargest(top_n, "pred_r1")
                gross = top["r1_capped"].mean()
                net = gross - COST_BPS * 100
                mkt = g["r1_capped"].mean()
                curve.append({"date": d_, "n_pool": len(gf),
                                "ret_net_pct": net, "ret_mkt_pct": mkt,
                                "alpha_pct": net - mkt})
            if not curve: continue
            cv = pd.DataFrame(curve).sort_values("date")
            cv["nav_net"] = (1 + cv["ret_net_pct"] / 100).cumprod()
            cv["nav_mkt"] = (1 + cv["ret_mkt_pct"] / 100).cumprod()
            cv["drawdown_net"] = cv["nav_net"] / cv["nav_net"].cummax() - 1

            n_days = len(cv)
            total_net = cv["nav_net"].iloc[-1] - 1
            total_mkt = cv["nav_mkt"].iloc[-1] - 1
            monthly_net = cv["ret_net_pct"].mean() * 20
            std = cv["ret_net_pct"].std() + 1e-9
            sharpe = cv["ret_net_pct"].mean() / std * np.sqrt(252)
            mdd = cv["drawdown_net"].min()
            win_rate = (cv["alpha_pct"] > 0).mean()

            results.append({
                "hour": hour, "top_n": top_n, "n_days": n_days,
                "total_net_pct": total_net * 100,
                "total_mkt_pct": total_mkt * 100,
                "monthly_net_pct": monthly_net,
                "alpha_total_pct": (total_net - total_mkt) * 100,
                "sharpe": sharpe, "mdd_pct": mdd * 100,
                "win_rate_alpha": win_rate,
            })
            cv.to_csv(OUT / "curves" / f"long_h{hour}_top{top_n}.csv", index=False)
            print(f"  h={hour:02d} Top{top_n:3d}: n_days={n_days}, "
                  f"净 {total_net*100:+.1f}%, 月化 {monthly_net:+.2f}%, "
                  f"Sharpe {sharpe:.2f}, MDD {mdd*100:.1f}%, α 胜率 {win_rate*100:.0f}%",
                  flush=True)

    if not results:
        print("无结果"); return

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "results.csv", index=False)

    # 报告
    md = [f"# P0 长 OOS T+1 r1 真实回测报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START} 至 {OOS_END} (~142 日)\n",
            f"模型: {MODEL_NAME} (IC=0.77 长 OOS)\n",
            f"约束: cap ±{R1_CAP}%, 成本 {COST_BPS*100:.2f}%, dist<{DIST_THRESHOLD}% 排除\n\n",
            "## 25 组合\n\n",
            "| 触发 | TopN | 日数 | 净 % | 市场 % | 月化 % | α 累计 % | Sharpe | MDD % | α 胜率 |\n",
            "|---|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in res_df.iterrows():
        md.append(f"| {int(r['hour']):02d}:30 | {int(r['top_n'])} | {int(r['n_days'])} | "
                   f"{r['total_net_pct']:+.1f} | {r['total_mkt_pct']:+.1f} | "
                   f"{r['monthly_net_pct']:+.2f} | {r['alpha_total_pct']:+.1f} | "
                   f"{r['sharpe']:.2f} | {r['mdd_pct']:.1f} | "
                   f"{r['win_rate_alpha']*100:.0f}% |\n")

    best = res_df.loc[res_df["sharpe"].idxmax()]
    md.append(f"\n## 最佳 Sharpe 组合\n\n")
    md.append(f"- **{int(best['hour']):02d}:30 触发, Top {int(best['top_n'])}**\n")
    md.append(f"- Sharpe: **{best['sharpe']:.2f}**\n")
    md.append(f"- 净收益: {best['total_net_pct']:+.1f}% ({int(best['n_days'])} 日)\n")
    md.append(f"- 月化净: {best['monthly_net_pct']:+.2f}%\n")
    md.append(f"- MDD: {best['mdd_pct']:.1f}%\n")
    md.append(f"- α 胜率: {best['win_rate_alpha']*100:.0f}%\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
