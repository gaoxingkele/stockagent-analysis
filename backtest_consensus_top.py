"""1H R20 + 日线 R5 双重共识 Top 池回测.

每日 EOD:
  1H Top X% (按 r20_1h_v2 分排)
  日线 Top X% (按 r5_v17_all 分排)
  共识池 = 两者交集

测试集中度: Top 10% / 5% / 2% / 1%
持有期 label: r1_next_open / r5_close / r20_close
成本: 单次 0.35%

输出: output/backtest_consensus/report.md + curves
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PROD = ROOT / "output" / "production"
F3 = ROOT / "output" / "1h_factors" / "factors_v3.parquet"
OUT = ROOT / "output" / "backtest_consensus"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "curves").mkdir(exist_ok=True)

OOS_START = "20260301"
OOS_END = "20260415"
COST_BPS = 35.0 / 10000

# 测试集中度
TOP_PCTS = [10, 5, 2, 1]


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"]


def main():
    t0 = time.time()
    print("\n=== 双重共识 Top 池回测 ===\n")

    # 1. 日线 factor + 推理
    print("[1] 加载日线 factor + 推理 r5_v17_all...", flush=True)
    from train_v15_refresh import load_window
    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if "industry" in daily.columns:
        daily["industry_id"] = pd.Categorical(daily["industry"].fillna("unknown")).codes
    else:
        daily["industry_id"] = 0
    b, feat_cols = load_model("r5_v17_all")
    miss = [c for c in feat_cols if c not in daily.columns]
    for c in miss: daily[c] = 0.0
    X = daily[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    daily["pred_d_r5"] = b.predict(X)
    print(f"   daily: {len(daily):,} 行, pred 均值 {daily['pred_d_r5'].mean():+.3f}", flush=True)

    # 2. 1H EOD 推理 r20_1h_v2
    print("\n[2] 加载 1H factors_v3 EOD + 推理 r20_1h_v2...", flush=True)
    df1h = pd.read_parquet(F3)
    df1h["trade_time"] = pd.to_datetime(df1h["trade_time"])
    df1h["trade_date"] = df1h["trade_date"].astype(str)
    eod = df1h[(df1h["trade_time"].dt.hour == 15) &
                 (df1h["trade_date"] >= OOS_START) &
                 (df1h["trade_date"] <= OOS_END)].copy()
    del df1h
    b2, feat_cols2 = load_model("r20_1h_v2")
    for c in feat_cols2:
        if c not in eod.columns: eod[c] = 0.0
        eod[c] = eod[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    X2 = eod[feat_cols2].astype("float32")
    eod["pred_1h_r20"] = b2.predict(X2)
    # 涨跌停 + cap r1
    eod["bad_filter"] = (eod["dist_to_upper_limit_pct"].fillna(100) < 2.0) | \
                          (eod["dist_to_lower_limit_pct"].fillna(100) < 1.0)
    print(f"   EOD bar: {len(eod):,}, 涨跌停接近: {eod['bad_filter'].sum():,}", flush=True)
    eod_preds = eod[["ts_code", "trade_date", "pred_1h_r20", "bad_filter",
                       "r1_next_open", "r4_next_morn", "r8_next_day", "r20_1h"]].drop_duplicates(["ts_code","trade_date"])
    del eod

    # 3. merge 日线 + 1H + label
    print("\n[3] merge 日线 + 1H...", flush=True)
    label_cols = [c for c in ["r10", "r20"] if c in daily.columns]
    merged = daily[["ts_code", "trade_date", "pred_d_r5"] + label_cols].merge(
        eod_preds, on=["ts_code", "trade_date"], how="inner"
    )
    # cap r1 / r20 ±9.5% (实际买不到涨停)
    for c in ["r1_next_open", "r5_close" if "r5_close" in merged.columns else None, "r20_1h"]:
        if c and c in merged.columns:
            merged[c+"_cap"] = merged[c].clip(-9.5, 9.5)
    if "r20" in merged.columns:
        merged["r20_cap"] = merged["r20"].clip(-30, 30)  # 20 日波段允许更大
    if "r10" in merged.columns:
        merged["r10_cap"] = merged["r10"].clip(-20, 20)
    print(f"   合并: {len(merged):,}", flush=True)

    # 4. 每日 EOD 构造池子 & 回测
    print("\n[4] 构造共识池 + 回测各集中度...\n", flush=True)
    results = []
    for top_pct in TOP_PCTS:
        curves = {"consensus": [], "only_1h": [], "only_daily": []}
        for d_, g in merged.groupby("trade_date"):
            g = g.dropna(subset=["pred_d_r5", "pred_1h_r20"])
            # 过滤涨跌停
            gf = g[~g["bad_filter"]]
            if len(gf) < 100: continue
            n_top = max(1, int(len(gf) * top_pct / 100))
            top_1h = gf.nlargest(n_top, "pred_1h_r20")
            top_d = gf.nlargest(n_top, "pred_d_r5")
            consensus = top_1h.merge(top_d[["ts_code"]], on="ts_code", how="inner")

            for name, sel in [("consensus", consensus), ("only_1h", top_1h), ("only_daily", top_d)]:
                if len(sel) == 0: continue
                # 用 1H r20_1h_cap (5 日 forward 在 1H label = 等同日线 r5)
                row = {"date": d_, "n": len(sel), "n_pool": len(gf)}
                if "r1_next_open_cap" in sel.columns:
                    row["r1"] = sel["r1_next_open_cap"].mean()
                if "r20_1h_cap" in sel.columns:
                    row["r5"] = sel["r20_1h_cap"].mean()  # 1H r20 = 5 日 forward
                if "r20_cap" in sel.columns:
                    row["r20"] = sel["r20_cap"].mean()
                row["mkt_r5"] = gf["r20_1h_cap"].mean() if "r20_1h_cap" in gf.columns else 0
                row["mkt_r1"] = gf["r1_next_open_cap"].mean() if "r1_next_open_cap" in gf.columns else 0
                row["mkt_r20"] = gf["r20_cap"].mean() if "r20_cap" in gf.columns else 0
                curves[name].append(row)

        for name in ["consensus", "only_1h", "only_daily"]:
            if not curves[name]: continue
            cv = pd.DataFrame(curves[name])
            n_days = len(cv)
            avg_n = cv["n"].mean()
            for ret_col, lab in [("r1", "T+1"), ("r5", "5日"), ("r20", "20日")]:
                if ret_col not in cv.columns: continue
                avg_ret = cv[ret_col].mean()
                avg_mkt = cv[f"mkt_{ret_col}"].mean() if f"mkt_{ret_col}" in cv.columns else 0
                net_per_trade = avg_ret - COST_BPS * 100
                alpha = net_per_trade - avg_mkt
                # 月化: T+1 假设每日交易 ×20 日; 5日 ×4; 20日 ×1
                trades_per_month = {"T+1": 20, "5日": 4, "20日": 1}[lab]
                monthly = net_per_trade * trades_per_month
                # Sharpe
                std = cv[ret_col].std() + 1e-9
                sharpe = avg_ret / std * np.sqrt({"T+1":252, "5日":50, "20日":12}[lab])
                results.append({
                    "top_pct": top_pct, "pool": name, "label": lab,
                    "n_days": n_days, "avg_n_stocks": avg_n,
                    "avg_ret_pct": avg_ret, "avg_mkt_pct": avg_mkt,
                    "net_pct": net_per_trade, "alpha_pct": alpha,
                    "monthly_net_pct": monthly, "sharpe": sharpe,
                })
            cv.to_csv(OUT / "curves" / f"top{top_pct}_{name}.csv", index=False)

        # 打印当前 top_pct 结果
        for name in ["consensus", "only_1h", "only_daily"]:
            for ret_col, lab in [("r1", "T+1"), ("r5", "5日"), ("r20", "20日")]:
                r = [r for r in results if r["top_pct"]==top_pct and r["pool"]==name and r["label"]==lab]
                if not r: continue
                r = r[0]
                print(f"  Top{top_pct:2d}% {name:10s} {lab:3s}: 池均 {r['avg_n_stocks']:6.1f} 股, "
                      f"net={r['net_pct']:+.3f}%, α={r['alpha_pct']:+.3f}pp, "
                      f"月化={r['monthly_net_pct']:+.2f}%, Sharpe={r['sharpe']:.2f}", flush=True)
        print()

    # === 报告 ===
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "results.csv", index=False)

    md = [f"# 双重共识 Top 池回测报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START} 至 {OOS_END}\n",
            f"成本: 单次 {COST_BPS*100:.2f}%, r1/r5 cap ±9.5%\n\n",
            "## 关键对比: 共识池 vs 单一模型 Top\n\n",
            "| Top%% | 池子 | 持有期 | 平均股数 | 单笔净 | α | 月化净 | Sharpe |\n",
            "|---|---|---|---|---|---|---|---|\n"]
    for _, r in res_df.iterrows():
        md.append(f"| {int(r['top_pct'])}% | {r['pool']} | {r['label']} | "
                   f"{r['avg_n_stocks']:.1f} | {r['net_pct']:+.3f}% | "
                   f"{r['alpha_pct']:+.3f}pp | {r['monthly_net_pct']:+.2f}% | "
                   f"{r['sharpe']:.2f} |\n")

    # 共识 vs 单一 best
    cons = res_df[res_df["pool"]=="consensus"]
    only1h = res_df[res_df["pool"]=="only_1h"]
    onlyd = res_df[res_df["pool"]=="only_daily"]
    if len(cons):
        best_cons = cons.loc[cons["sharpe"].idxmax()]
        md.append(f"\n## 共识池最佳组合\n\n")
        md.append(f"- Top {int(best_cons['top_pct'])}%, 持有 {best_cons['label']}\n")
        md.append(f"- 池均 {best_cons['avg_n_stocks']:.1f} 股, 月化净 {best_cons['monthly_net_pct']:+.2f}%, Sharpe {best_cons['sharpe']:.2f}\n")

    p = OUT / "report.md"
    p.write_text("".join(md), encoding="utf-8")
    print(f"\n输出: {p}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
