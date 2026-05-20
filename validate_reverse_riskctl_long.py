"""严格长 OOS 反向风控验证 (142 日, 全用 long 模型).

模型 (全部 cut at 20250930):
  - r20_1h_v2_long: 1H R20
  - r5_v17_long:    日线 R5
  - r20_v16_long:   日线 R20

OOS: 20251001 至 20260331 (留 r20 label 完整)

验证: 实验 1/2/3 同 validate_reverse_riskctl.py 但用 long 模型 + 长 OOS
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

OOS_START = "20251001"
OOS_END = "20260331"  # 留 r20 label 完整 (20260331 + 20 ≈ 20260427 仍在数据内)


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"]


def main():
    t0 = time.time()
    print("\n=== 严格长 OOS 反向风控验证 ===\n")
    print(f"OOS: {OOS_START} 至 {OOS_END}\n")

    # 1. 日线 factor + 推理 long 版本
    print("[1] 加载日线 + 推理 r5_long / r20_long...", flush=True)
    from train_v15_refresh import load_window
    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if "industry" in daily.columns:
        # 用同样的 industry_map (从 r5_v17_long meta)
        meta_r5 = json.loads((PROD / "r5_v17_long" / "feature_meta.json").read_text(encoding="utf-8"))
        ind_map = meta_r5.get("industry_map", {})
        daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
    else:
        daily["industry_id"] = -1
    daily_preds = daily[["ts_code", "trade_date"]].copy()
    label_cols = [c for c in ["r10", "r20"] if c in daily.columns]
    for c in label_cols: daily_preds[c] = daily[c]
    for name in ["r5_v17_long", "r20_v16_long"]:
        b, feat_cols = load_model(name)
        miss = [c for c in feat_cols if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily_preds[f"pred_{name}"] = b.predict(X)
        print(f"   {name} pred 均值 {daily_preds[f'pred_{name}'].mean():+.3f}", flush=True)
    del daily

    # 2. 1H r20_1h_v2_long EOD 推理
    print("\n[2] 1H r20_1h_v2_long EOD 推理...", flush=True)
    df1h = pd.read_parquet(F3)
    df1h["trade_time"] = pd.to_datetime(df1h["trade_time"])
    df1h["trade_date"] = df1h["trade_date"].astype(str)
    eod = df1h[(df1h["trade_time"].dt.hour == 15) &
                 (df1h["trade_date"] >= OOS_START) &
                 (df1h["trade_date"] <= OOS_END)].copy()
    del df1h
    b2, feat_cols2 = load_model("r20_1h_v2_long")
    for c in feat_cols2:
        if c not in eod.columns: eod[c] = 0.0
        eod[c] = eod[c].replace([np.inf, -np.inf], np.nan).clip(-200, 200)
    X2 = eod[feat_cols2].astype("float32")
    eod["pred_1h_r20"] = b2.predict(X2)
    eod_preds = eod[["ts_code","trade_date","pred_1h_r20","r20_1h",
                       "dist_to_upper_limit_pct","dist_to_lower_limit_pct"]].drop_duplicates(["ts_code","trade_date"])
    del eod

    # 3. merge
    print("\n[3] merge...", flush=True)
    m = daily_preds.merge(eod_preds, on=["ts_code", "trade_date"], how="inner")
    m["bad_filter"] = (m["dist_to_upper_limit_pct"].fillna(100) < 2.0) | \
                       (m["dist_to_lower_limit_pct"].fillna(100) < 1.0)
    if "r20" in m.columns:
        m["r20_cap"] = m["r20"].clip(-30, 30)
    print(f"   合并: {len(m):,}, 日数: {m['trade_date'].nunique()}", flush=True)

    # === 实验 1: 按 1H R20 拆 ===
    print("\n=== 实验 1: 按 1H R20 拆 buy 池 ===\n")
    res = []
    for d_, g in m.groupby("trade_date"):
        g = g.dropna(subset=["pred_r20_v16_long", "pred_1h_r20", "r20_cap"])
        gf = g[~g["bad_filter"]]
        if len(gf) < 500: continue
        n_top = max(1, int(len(gf) * 0.20))
        anchor = gf.nlargest(n_top, "pred_r20_v16_long").copy()
        anchor["1h_rank"] = anchor["pred_1h_r20"].rank(pct=True)
        for label, mask in [
            ("1h_bot30", anchor["1h_rank"] < 0.30),
            ("1h_mid40", (anchor["1h_rank"] >= 0.30) & (anchor["1h_rank"] < 0.70)),
            ("1h_top30", anchor["1h_rank"] >= 0.70),
        ]:
            sub = anchor[mask]
            if len(sub): res.append({"date": d_, "sub": label, "n": len(sub),
                                       "r20": sub["r20_cap"].mean()})
    df1 = pd.DataFrame(res)
    for sub, g in df1.groupby("sub"):
        sharpe = g["r20"].mean() / (g["r20"].std()+1e-9) * np.sqrt(50)
        print(f"  {sub}: n_days={len(g)}, 池均 {g['n'].mean():.1f}, "
              f"r20={g['r20'].mean():+.3f}% std={g['r20'].std():.3f} Sharpe={sharpe:.2f}")

    # === 实验 2: 按 日线 R5 拆 ===
    print("\n=== 实验 2: 按 日线 R5 拆 ===\n")
    res2 = []
    for d_, g in m.groupby("trade_date"):
        g = g.dropna(subset=["pred_r20_v16_long", "pred_r5_v17_long", "r20_cap"])
        gf = g[~g["bad_filter"]]
        if len(gf) < 500: continue
        n_top = max(1, int(len(gf) * 0.20))
        anchor = gf.nlargest(n_top, "pred_r20_v16_long").copy()
        anchor["d5_rank"] = anchor["pred_r5_v17_long"].rank(pct=True)
        for label, mask in [
            ("d5_bot30", anchor["d5_rank"] < 0.30),
            ("d5_mid40", (anchor["d5_rank"] >= 0.30) & (anchor["d5_rank"] < 0.70)),
            ("d5_top30", anchor["d5_rank"] >= 0.70),
        ]:
            sub = anchor[mask]
            if len(sub): res2.append({"date": d_, "sub": label, "n": len(sub),
                                       "r20": sub["r20_cap"].mean()})
    df2 = pd.DataFrame(res2)
    for sub, g in df2.groupby("sub"):
        sharpe = g["r20"].mean() / (g["r20"].std()+1e-9) * np.sqrt(50)
        print(f"  {sub}: n_days={len(g)}, 池均 {g['n'].mean():.1f}, "
              f"r20={g['r20'].mean():+.3f}% std={g['r20'].std():.3f} Sharpe={sharpe:.2f}")

    # === 实验 3: 四象限 ===
    print("\n=== 实验 3: 四象限 (1H × d5 在 R20 高分池内) ===\n")
    res3 = []
    for d_, g in m.groupby("trade_date"):
        g = g.dropna(subset=["pred_r20_v16_long", "pred_r5_v17_long", "pred_1h_r20", "r20_cap"])
        gf = g[~g["bad_filter"]]
        if len(gf) < 500: continue
        n_top = max(1, int(len(gf) * 0.20))
        anchor = gf.nlargest(n_top, "pred_r20_v16_long").copy()
        anchor["1h_rank"] = anchor["pred_1h_r20"].rank(pct=True)
        anchor["d5_rank"] = anchor["pred_r5_v17_long"].rank(pct=True)
        sub_a = anchor[(anchor["1h_rank"] < 0.30) & (anchor["d5_rank"] < 0.30)]
        sub_b = anchor[(anchor["1h_rank"] >= 0.70) & (anchor["d5_rank"] >= 0.70)]
        sub_c = anchor[(anchor["1h_rank"] < 0.30) & (anchor["d5_rank"] >= 0.70)]
        sub_d = anchor[(anchor["1h_rank"] >= 0.70) & (anchor["d5_rank"] < 0.30)]
        for nm, sub in [("A_both_low", sub_a), ("B_both_high", sub_b),
                          ("C_1h_low_d5_high", sub_c), ("D_1h_high_d5_low", sub_d)]:
            if len(sub): res3.append({"date": d_, "sub": nm, "n": len(sub),
                                       "r20": sub["r20_cap"].mean()})
    df3 = pd.DataFrame(res3)
    for sub, g in df3.groupby("sub"):
        sharpe = g["r20"].mean() / (g["r20"].std()+1e-9) * np.sqrt(50)
        print(f"  {sub}: n_days={len(g)}, 池均 {g['n'].mean():.1f}, "
              f"r20={g['r20'].mean():+.3f}% std={g['r20'].std():.3f} Sharpe={sharpe:.2f}")

    # 输出报告
    rep = [f"# 严格长 OOS 反向风控报告 ({OOS_START}-{OOS_END}, 142 日)\n\n",
            f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"模型: 全部 cut at 20250930 (r5_v17_long, r20_v16_long, r20_1h_v2_long)\n\n",
            "## 实验 1 (1H R20 拆)\n\n| 子集 | 日数 | 池均股数 | r20 % | std | Sharpe |\n|---|---|---|---|---|---|\n"]
    for sub, g in df1.groupby("sub"):
        rep.append(f"| {sub} | {len(g)} | {g['n'].mean():.1f} | {g['r20'].mean():+.3f} | "
                    f"{g['r20'].std():.3f} | {g['r20'].mean()/(g['r20'].std()+1e-9)*np.sqrt(50):.2f} |\n")
    rep.append("\n## 实验 2 (日线 R5 拆)\n\n| 子集 | 日数 | 池均股数 | r20 % | std | Sharpe |\n|---|---|---|---|---|---|\n")
    for sub, g in df2.groupby("sub"):
        rep.append(f"| {sub} | {len(g)} | {g['n'].mean():.1f} | {g['r20'].mean():+.3f} | "
                    f"{g['r20'].std():.3f} | {g['r20'].mean()/(g['r20'].std()+1e-9)*np.sqrt(50):.2f} |\n")
    rep.append("\n## 实验 3 (四象限)\n\n| 象限 | 日数 | 池均股数 | r20 % | std | Sharpe |\n|---|---|---|---|---|---|\n")
    for sub, g in df3.groupby("sub"):
        rep.append(f"| {sub} | {len(g)} | {g['n'].mean():.1f} | {g['r20'].mean():+.3f} | "
                    f"{g['r20'].std():.3f} | {g['r20'].mean()/(g['r20'].std()+1e-9)*np.sqrt(50):.2f} |\n")

    Path(OUT / "reverse_riskctl_long_report.md").write_text("".join(rep), encoding="utf-8")
    print(f"\n输出: {OUT / 'reverse_riskctl_long_report.md'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
