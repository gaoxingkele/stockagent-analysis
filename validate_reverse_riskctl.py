"""B: 反向风控假设验证.

用户假设: 1H R20 / 日线 R5 sell 评分可作 日线 R20 的逆向风控.
"短期 sell + 长期 buy = 等回调买入"

验证设计:
  锚: 日线 R20 Top 20% (buy 池, ~1100 股/日)
  分子集:
    a) 1H R20 Bottom 30%: 短期超跌 (期望: 20 日实际收益最好)
    b) 1H R20 Mid 40%: 中性
    c) 1H R20 Top 30%: 短期透支 (期望: 20 日实际收益最差, 已被证伪过)
  同样按 日线 R5 分子集

比较 r20_close 真实 20 日收益, 看 sub-pool 差异.
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
OUT = ROOT / "output" / "cross_scale"
OUT.mkdir(parents=True, exist_ok=True)

OOS_START = "20260301"
OOS_END = "20260415"


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"]


def main():
    t0 = time.time()
    print("\n=== B: 反向风控假设验证 ===\n")

    # 1. 日线 factor + r5_v17_all + r20_v16_all 推理
    print("[1] 加载日线 + 推理 r5 / r20...", flush=True)
    from train_v15_refresh import load_window
    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if "industry" in daily.columns:
        daily["industry_id"] = pd.Categorical(daily["industry"].fillna("unknown")).codes
    else:
        daily["industry_id"] = 0
    daily_preds = daily[["ts_code", "trade_date"]].copy()
    label_cols = [c for c in ["r10", "r20"] if c in daily.columns]
    for c in label_cols: daily_preds[c] = daily[c]
    for name in ["r5_v17_all", "r20_v16_all"]:
        b, feat_cols = load_model(name)
        miss = [c for c in feat_cols if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily_preds[f"pred_{name}"] = b.predict(X)
        print(f"   {name} pred 均值 {daily_preds[f'pred_{name}'].mean():+.3f}", flush=True)
    del daily

    # 2. 1H r20_1h_v2 EOD 推理
    print("\n[2] 1H r20_1h_v2 EOD 推理...", flush=True)
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
    eod_preds = eod[["ts_code","trade_date","pred_1h_r20","r20_1h",
                       "dist_to_upper_limit_pct","dist_to_lower_limit_pct"]].drop_duplicates(["ts_code","trade_date"])
    del eod

    # 3. merge
    print("\n[3] merge + 风控测试...", flush=True)
    m = daily_preds.merge(eod_preds, on=["ts_code", "trade_date"], how="inner")
    m["bad_filter"] = (m["dist_to_upper_limit_pct"].fillna(100) < 2.0) | \
                       (m["dist_to_lower_limit_pct"].fillna(100) < 1.0)
    # cap labels
    if "r20" in m.columns:
        m["r20_cap"] = m["r20"].clip(-30, 30)

    # 4. 核心实验: 日线 R20 Top 20% buy 池, 按 1H R20 / 日线 R5 子分位
    print("\n=== 实验 1: 锚日线 R20 Top 20%, 按 1H R20 拆 ===\n")
    results = []
    for d_, g in m.groupby("trade_date"):
        g = g.dropna(subset=["pred_r20_v16_all", "pred_1h_r20", "r20_cap"])
        gf = g[~g["bad_filter"]]
        if len(gf) < 500: continue
        # 日线 R20 Top 20% (buy 池)
        n_top = max(1, int(len(gf) * 0.20))
        anchor = gf.nlargest(n_top, "pred_r20_v16_all").copy()
        # 在 buy 池内, 按 1H R20 拆 3 段
        anchor["1h_rank_pct"] = anchor["pred_1h_r20"].rank(pct=True)
        # Bottom 30% / Mid 40% / Top 30%
        for label, mask in [
            ("1h_bot30",  anchor["1h_rank_pct"] < 0.30),
            ("1h_mid40",  (anchor["1h_rank_pct"] >= 0.30) & (anchor["1h_rank_pct"] < 0.70)),
            ("1h_top30",  anchor["1h_rank_pct"] >= 0.70),
        ]:
            sub = anchor[mask]
            if len(sub) == 0: continue
            results.append({"date": d_, "sub": label, "n": len(sub),
                              "r20_mean": sub["r20_cap"].mean()})

    df_r = pd.DataFrame(results)
    print("--- 按 1H R20 拆 buy 池子集 (持有 20 日 r20 实际) ---")
    for sub, g in df_r.groupby("sub"):
        avg_n = g["n"].mean()
        avg_r = g["r20_mean"].mean()
        std_r = g["r20_mean"].std()
        sharpe = avg_r / (std_r + 1e-9) * np.sqrt(50)
        n_days = len(g)
        print(f"  {sub}: 日数 {n_days}, 池均 {avg_n:.1f} 股, "
              f"r20 均收益 {avg_r:+.3f}%, std {std_r:.3f}, Sharpe={sharpe:.2f}")

    # 实验 2: 锚日线 R20 Top 20%, 按 日线 R5 拆
    print("\n=== 实验 2: 锚日线 R20 Top 20%, 按 日线 R5 拆 ===\n")
    results2 = []
    for d_, g in m.groupby("trade_date"):
        g = g.dropna(subset=["pred_r20_v16_all", "pred_r5_v17_all", "r20_cap"])
        gf = g[~g["bad_filter"]]
        if len(gf) < 500: continue
        n_top = max(1, int(len(gf) * 0.20))
        anchor = gf.nlargest(n_top, "pred_r20_v16_all").copy()
        anchor["d5_rank_pct"] = anchor["pred_r5_v17_all"].rank(pct=True)
        for label, mask in [
            ("d5_bot30",  anchor["d5_rank_pct"] < 0.30),
            ("d5_mid40",  (anchor["d5_rank_pct"] >= 0.30) & (anchor["d5_rank_pct"] < 0.70)),
            ("d5_top30",  anchor["d5_rank_pct"] >= 0.70),
        ]:
            sub = anchor[mask]
            if len(sub) == 0: continue
            results2.append({"date": d_, "sub": label, "n": len(sub),
                              "r20_mean": sub["r20_cap"].mean()})

    df_r2 = pd.DataFrame(results2)
    print("--- 按 日线 R5 拆 buy 池子集 ---")
    for sub, g in df_r2.groupby("sub"):
        avg_n = g["n"].mean()
        avg_r = g["r20_mean"].mean()
        std_r = g["r20_mean"].std()
        sharpe = avg_r / (std_r + 1e-9) * np.sqrt(50)
        n_days = len(g)
        print(f"  {sub}: 日数 {n_days}, 池均 {avg_n:.1f} 股, "
              f"r20 均收益 {avg_r:+.3f}%, std {std_r:.3f}, Sharpe={sharpe:.2f}")

    # 实验 3: 双重过滤 - 日线 R20 Top 20% + 1H R20 Bot30 + 日线 R5 Bot30
    print("\n=== 实验 3: 三重共识 - 日线 R20 高 + 1H R20 低 + 日线 R5 低 ===\n")
    results3 = []
    for d_, g in m.groupby("trade_date"):
        g = g.dropna(subset=["pred_r20_v16_all", "pred_r5_v17_all", "pred_1h_r20", "r20_cap"])
        gf = g[~g["bad_filter"]]
        if len(gf) < 500: continue
        n_top = max(1, int(len(gf) * 0.20))
        anchor = gf.nlargest(n_top, "pred_r20_v16_all").copy()
        anchor["1h_rank"] = anchor["pred_1h_r20"].rank(pct=True)
        anchor["d5_rank"] = anchor["pred_r5_v17_all"].rank(pct=True)
        # 子集 A: 1H 低 + d5 低 (双重短期超跌)
        sub_a = anchor[(anchor["1h_rank"] < 0.30) & (anchor["d5_rank"] < 0.30)]
        # 子集 B: 1H 高 + d5 高 (双重透支, 已知最差)
        sub_b = anchor[(anchor["1h_rank"] >= 0.70) & (anchor["d5_rank"] >= 0.70)]
        # 子集 C: 1H 低 + d5 高 (跨尺度分歧 - 抢钱时机?)
        sub_c = anchor[(anchor["1h_rank"] < 0.30) & (anchor["d5_rank"] >= 0.70)]
        # 子集 D: 1H 高 + d5 低
        sub_d = anchor[(anchor["1h_rank"] >= 0.70) & (anchor["d5_rank"] < 0.30)]
        for nm, sub in [("A_both_low", sub_a), ("B_both_high", sub_b),
                          ("C_1h_low_d5_high", sub_c), ("D_1h_high_d5_low", sub_d)]:
            if len(sub) == 0: continue
            results3.append({"date": d_, "sub": nm, "n": len(sub),
                              "r20_mean": sub["r20_cap"].mean()})

    df_r3 = pd.DataFrame(results3)
    print("--- 四象限 (1H × d5) ---")
    for sub, g in df_r3.groupby("sub"):
        avg_n = g["n"].mean()
        avg_r = g["r20_mean"].mean()
        std_r = g["r20_mean"].std()
        sharpe = avg_r / (std_r + 1e-9) * np.sqrt(50)
        n_days = len(g)
        print(f"  {sub}: 日数 {n_days}, 池均 {avg_n:.1f}, "
              f"r20={avg_r:+.3f}% std {std_r:.3f} Sharpe={sharpe:.2f}")

    # 输出
    rep = [f"# 反向风控假设验证报告\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"OOS: {OOS_START} 至 {OOS_END}\n",
            f"锚: 日线 R20 Top 20% (~1100 股/日 buy 池)\n\n",
            "## 实验 1: 按 1H R20 子分位 (短期信号)\n\n",
            "| 子集 | 平均股数 | 20日均收益 | std | Sharpe |\n|---|---|---|---|---|\n"]
    for sub, g in df_r.groupby("sub"):
        rep.append(f"| {sub} | {g['n'].mean():.1f} | {g['r20_mean'].mean():+.3f}% | "
                    f"{g['r20_mean'].std():.3f} | {g['r20_mean'].mean()/(g['r20_mean'].std()+1e-9)*np.sqrt(50):.2f} |\n")

    rep.append("\n## 实验 2: 按 日线 R5 子分位\n\n")
    rep.append("| 子集 | 平均股数 | 20日均收益 | std | Sharpe |\n|---|---|---|---|---|\n")
    for sub, g in df_r2.groupby("sub"):
        rep.append(f"| {sub} | {g['n'].mean():.1f} | {g['r20_mean'].mean():+.3f}% | "
                    f"{g['r20_mean'].std():.3f} | {g['r20_mean'].mean()/(g['r20_mean'].std()+1e-9)*np.sqrt(50):.2f} |\n")

    rep.append("\n## 实验 3: 四象限 (1H × d5)\n\n")
    rep.append("| 象限 | 平均股数 | 20日均收益 | std | Sharpe |\n|---|---|---|---|---|\n")
    for sub, g in df_r3.groupby("sub"):
        rep.append(f"| {sub} | {g['n'].mean():.1f} | {g['r20_mean'].mean():+.3f}% | "
                    f"{g['r20_mean'].std():.3f} | {g['r20_mean'].mean()/(g['r20_mean'].std()+1e-9)*np.sqrt(50):.2f} |\n")

    Path(OUT / "reverse_riskctl_report.md").write_text("".join(rep), encoding="utf-8")
    print(f"\n输出: {OUT / 'reverse_riskctl_report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
