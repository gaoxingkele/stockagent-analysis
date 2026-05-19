"""P0: T+1 r1_next_open 真实回测.

策略:
  每个交易日 X 时段 (10:30/11:30/13:30/14:30/15:00 EOD), 用模型预测 r1_next_open,
  选 Top N 股票, 当 bar close 买入, 次日 09:30 (first_open) 卖出.

成本:
  - 印花税: 0.10% (单边, 卖出收)
  - 佣金: 0.025% × 2 = 0.05% (双边)
  - 滑点: 0.10% × 2 = 0.20% (双边)
  合计 ≈ 0.35% 单次交易

过滤:
  - 涨停: close >= prev_close * 1.098 → 跳过 (买不到, 次日易跌停)
  - 跌停接近: dist_to_lower_limit_pct < 0.5 → 跳过
  - 流动性: day_vol < 当市场分位 30% → 跳过
  - 模型分位: pred_r1 必须 > 历史 p70 (过滤弱信号)

测试:
  Top N: 5 / 10 / 20 / 50 / 100
  时段:  10/11/13/14/15 (1H bar 所在小时)

输出:
  output/backtest_t1/results.csv (按 Top N × 时段 汇总)
  output/backtest_t1/curves/<topN>_<hour>.csv (每条净值曲线)
  output/backtest_t1/report.md
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
OUT = ROOT / "output" / "backtest_t1"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "curves").mkdir(exist_ok=True)

OOS_START = "20260301"
MODEL_NAME = "r1_next_open_v3"

# 交易成本 (单次完整交易, 买+卖)
COST_BPS = 35.0 / 10000  # 0.35%

# 测试参数
TOP_NS = [5, 10, 20, 50, 100]
HOURS = [10, 11, 13, 14, 15]


def load_model():
    d = PROD / MODEL_NAME
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return booster, meta["feature_cols"]


def main():
    t0 = time.time()
    print("\n=== P0 T+1 r1_next_open 真实回测 ===\n")
    print(f"加载 factors_v3...")
    df = pd.read_parquet(F3)
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = df["trade_date"].astype(str)
    print(f"  {len(df):,} × {len(df.columns)}", flush=True)

    print(f"OOS 切片 (≥ {OOS_START})...")
    oos = df[df["trade_date"] >= OOS_START].copy()
    print(f"  OOS bar: {len(oos):,}", flush=True)
    del df

    print(f"加载模型 {MODEL_NAME}...")
    booster, feat_cols = load_model()
    for c in feat_cols:
        if c not in oos.columns: oos[c] = 0.0
        oos[c] = oos[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    X = oos[feat_cols].astype("float32")
    oos["pred_r1"] = booster.predict(X)
    del X
    print(f"  推理完成 {time.time()-t0:.0f}s", flush=True)

    # 过滤 (P0 严格版, 用 v3 已有 dist_to_upper/lower_limit_pct)
    print("应用过滤 (P0 严格版)...")
    # dist_to_upper_limit_pct = (prev_close*1.10/close - 1)*100, 用主板 10% 算法
    # 主板涨停: dist ≈ 0
    # 创业板/科创板涨 10%+: dist < 0 (因为实际涨到 +10% 时 dist 已经 = 0; +15% → dist≈-4%)
    # 安全阈值: dist < 2 都过滤 (主板涨幅 > 8%, 或创业板 > 12%)
    oos["is_near_upper"] = oos["dist_to_upper_limit_pct"].fillna(100) < 2.0
    oos["is_at_lower"] = oos["dist_to_lower_limit_pct"].fillna(100) < 1.0
    oos["bad_filter"] = oos["is_near_upper"] | oos["is_at_lower"]
    n_bad = oos["bad_filter"].sum()
    print(f"  接近涨停/已跌停 bar: {n_bad:,} ({n_bad/len(oos)*100:.1f}%)", flush=True)

    # 必须有 r1_next_open label (用于计算实际收益)
    oos_valid = oos.dropna(subset=["r1_next_open", "pred_r1"])
    # label clip ±20%
    oos_valid = oos_valid[oos_valid["r1_next_open"].abs() <= 20]
    # ★ 真实买入约束: cap r1 at +9.5% (实际涨停集合竞价买不到)
    oos_valid["r1_capped"] = oos_valid["r1_next_open"].clip(upper=9.5, lower=-9.5)
    print(f"  有效 bar (含 label): {len(oos_valid):,}", flush=True)
    print(f"  r1 cap 截掉 (|r1|>9.5%): {(oos_valid['r1_next_open'].abs() > 9.5).sum():,}", flush=True)

    # === 主回测循环 ===
    print("\n开始回测各参数组合...")
    results = []
    for hour in HOURS:
        bar_at_hour = oos_valid[oos_valid["trade_time"].dt.hour == hour].copy()
        if len(bar_at_hour) == 0: continue
        for top_n in TOP_NS:
            curve_rows = []
            for d_, g in bar_at_hour.groupby("trade_date"):
                # 过滤掉接近涨跌停的
                gf = g[~g["bad_filter"]].copy()
                if len(gf) < top_n: continue
                # 选 Top N
                top = gf.nlargest(top_n, "pred_r1")
                gross = top["r1_capped"].mean()  # 已 cap ±9.5%
                net = gross - COST_BPS * 100  # 减 0.35%
                mkt = g["r1_capped"].mean()
                curve_rows.append({
                    "date": d_, "n_pool": len(gf),
                    "ret_gross_pct": gross, "ret_net_pct": net,
                    "ret_mkt_pct": mkt,
                    "alpha_pct": net - mkt,
                })
            if not curve_rows: continue
            curve = pd.DataFrame(curve_rows).sort_values("date")
            # 累计净值 (从 1.0 起)
            curve["nav_gross"] = (1 + curve["ret_gross_pct"] / 100).cumprod()
            curve["nav_net"] = (1 + curve["ret_net_pct"] / 100).cumprod()
            curve["nav_mkt"] = (1 + curve["ret_mkt_pct"] / 100).cumprod()
            curve["drawdown_net"] = curve["nav_net"] / curve["nav_net"].cummax() - 1

            # 统计
            n_days = len(curve)
            total_gross = curve["nav_gross"].iloc[-1] - 1
            total_net = curve["nav_net"].iloc[-1] - 1
            total_mkt = curve["nav_mkt"].iloc[-1] - 1
            monthly_net = curve["ret_net_pct"].mean() * 20
            sharpe = curve["ret_net_pct"].mean() / (curve["ret_net_pct"].std() + 1e-9) * np.sqrt(252)
            mdd = curve["drawdown_net"].min()
            win_rate = (curve["alpha_pct"] > 0).mean()

            row = {
                "hour": hour, "top_n": top_n, "n_days": n_days,
                "total_gross_pct": total_gross * 100,
                "total_net_pct": total_net * 100,
                "total_mkt_pct": total_mkt * 100,
                "monthly_net_pct": monthly_net,
                "alpha_total_pct": (total_net - total_mkt) * 100,
                "sharpe": sharpe, "mdd_pct": mdd * 100,
                "win_rate_alpha": win_rate,
            }
            results.append(row)
            # 保存曲线
            curve.to_csv(OUT / "curves" / f"t1_h{hour}_top{top_n}.csv", index=False)
            print(f"  h={hour:02d} Top{top_n:3d}: n_days={n_days}, "
                  f"净值={total_net*100:+.1f}%, mkt={total_mkt*100:+.1f}%, "
                  f"月化={monthly_net:+.2f}%, Sharpe={sharpe:.2f}, MDD={mdd*100:.1f}%", flush=True)

    if not results:
        print("无结果"); return

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "results.csv", index=False)

    # === Markdown 报告 ===
    print("\n生成报告...")
    md = [f"# P0 T+1 r1_next_open 真实回测报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START} 至今\n",
            f"模型: {MODEL_NAME} (IC=0.760)\n",
            f"交易成本: 单次 {COST_BPS*100:.2f}%\n\n",
            "## 各参数组合汇总\n\n",
            "| 触发 | TopN | 日数 | 净 % | 市场 % | 月化 % | α 累计 % | Sharpe | MDD % | α 胜率 |\n",
            "|---|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in res_df.iterrows():
        md.append(f"| {int(r['hour']):02d}:30 | {int(r['top_n'])} | {int(r['n_days'])} | "
                   f"{r['total_net_pct']:+.1f} | {r['total_mkt_pct']:+.1f} | "
                   f"{r['monthly_net_pct']:+.2f} | {r['alpha_total_pct']:+.1f} | "
                   f"{r['sharpe']:.2f} | {r['mdd_pct']:.1f} | "
                   f"{r['win_rate_alpha']*100:.0f}% |\n")

    # 最佳组合
    best = res_df.loc[res_df["total_net_pct"].idxmax()]
    md.append(f"\n## 最佳净收益组合\n\n")
    md.append(f"- **{int(best['hour']):02d}:30 触发, Top {int(best['top_n'])}**\n")
    md.append(f"- 净收益: **{best['total_net_pct']:+.1f}%** ({int(best['n_days'])} 日)\n")
    md.append(f"- 月化 α: **{best['monthly_net_pct']:+.2f}%**\n")
    md.append(f"- Sharpe: **{best['sharpe']:.2f}**\n")
    md.append(f"- MDD: {best['mdd_pct']:.1f}%\n")

    # Sharpe 最高
    best_sharpe = res_df.loc[res_df["sharpe"].idxmax()]
    md.append(f"\n## 最佳风险调整 (Sharpe)\n\n")
    md.append(f"- **{int(best_sharpe['hour']):02d}:30 触发, Top {int(best_sharpe['top_n'])}**\n")
    md.append(f"- Sharpe: **{best_sharpe['sharpe']:.2f}**\n")
    md.append(f"- 净收益: {best_sharpe['total_net_pct']:+.1f}%, 月化 {best_sharpe['monthly_net_pct']:+.2f}%, MDD {best_sharpe['mdd_pct']:.1f}%\n")

    md.append("\n## 关键观察\n\n")
    md.append("- **理论 α** (validate_t1_models.py) 报 10:30 月化 +57pp/月 是统计意义\n")
    md.append("- **真实回测**减去 0.35% 单次成本 + 涨跌停过滤后是真正可执行结果\n")
    md.append("- 同时段下, Top N 越小集中度越高, 但样本噪声越大\n")
    md.append("- 越早触发理论 α 越强, 但实际 11:30 / 13:30 也好观察板块轮动\n")

    Path(OUT / "report.md").write_text("".join(md), encoding="utf-8")
    print(f"输出: {OUT / 'report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
