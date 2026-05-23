"""诊断 V12 双轨持仓覆盖率仅 32% 的原因 + 灾难月 202603 详情.

逐日输出 V7c 5 铁律各步过滤后的池子大小, 找哪一步是瓶颈.
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

PROD = ROOT / "output" / "production"
LONG_FEAT_P = ROOT / "output" / "long_return_features" / "features.parquet"
OUT = ROOT / "output" / "backtest_v12_dual"

OOS_START = "20251001"
OOS_END = "20260331"
R5_ANCHOR = (-0.21, 0.82, 1.80)
R10_ANCHOR = (-3.55, 0.56, 6.91)
R20_ANCHOR = (-7.10, 2.45, 14.08)


def map_anchored(v, p5, p50, p95):
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, 50.0)
    out = np.where(v <= p5, 0, out)
    out = np.where(v >= p95, 100, out)
    mask_lo = (v > p5) & (v <= p50)
    out = np.where(mask_lo, (v - p5) / (p50 - p5 + 1e-9) * 50, out)
    mask_hi = (v > p50) & (v < p95)
    out = np.where(mask_hi, 50 + (v - p50) / (p95 - p50 + 1e-9) * 50, out)
    return out


def load_model(name):
    d = PROD / name
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    m = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    return b, m["feature_cols"], m.get("industry_map", {})


def main():
    t0 = time.time()
    print("\n=== 诊断 V12 双轨覆盖率 + 202603 灾难月 ===\n", flush=True)

    daily = load_window(OOS_START, OOS_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if LONG_FEAT_P.exists():
        lf = pd.read_parquet(LONG_FEAT_P)
        lf["trade_date"] = lf["trade_date"].astype(str)
        daily = daily.merge(lf, on=["ts_code", "trade_date"], how="left")

    # 推理
    for name in ["r5_v17_all_nost", "r10_v16_all_nost", "r20_v16_all_nost",
                   "r5_v17_long_nost"]:
        b, fc, ind_map = load_model(name)
        if ind_map and "industry" in daily.columns:
            daily["industry_id"] = daily["industry"].fillna("unknown").map(ind_map).fillna(-1).astype(int)
        miss = [c for c in fc if c not in daily.columns]
        for c in miss: daily[c] = 0.0
        X = daily[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
        daily[f"pred_{name}"] = b.predict(X)

    daily["buy_r10_score"] = map_anchored(daily["pred_r10_v16_all_nost"].values, *R10_ANCHOR)
    daily["buy_r20_score"] = map_anchored(daily["pred_r20_v16_all_nost"].values, *R20_ANCHOR)
    daily["buy_score"] = 0.5 * daily["buy_r10_score"] + 0.5 * daily["buy_r20_score"]

    # 逐日逐步过滤诊断
    rows = []
    for d_, g in daily.groupby("trade_date"):
        n_all = len(g)
        # Step 1: buy_score [70, 85]
        n_b = (g["buy_score"].between(70, 85)).sum()
        # Step 2: 加 pyr_velocity < p35
        if "pyr_velocity_20_60" in g.columns:
            p35 = g["pyr_velocity_20_60"].quantile(0.35)
            m_pyr = g["pyr_velocity_20_60"] < p35
        else:
            m_pyr = pd.Series(True, index=g.index)
        n_bp = (g["buy_score"].between(70, 85) & m_pyr).sum()
        # Step 3: 加 f1/f2
        if "f1_neg1" in g.columns and "f2_pos1" in g.columns:
            m_f = (g["f1_neg1"].abs() < 0.005) & (g["f2_pos1"].abs() < 0.005)
        else:
            m_f = pd.Series(True, index=g.index)
        n_bpf = (g["buy_score"].between(70, 85) & m_pyr & m_f).sum()
        # Step 4: 加 R5 反向 Bot 15% + Bot 35%
        v7c = g[g["buy_score"].between(70, 85) & m_pyr & m_f]
        if len(v7c) > 0:
            r5_rank = v7c["pred_r5_v17_long_nost"].rank(pct=True, method="first")
            n_a = (r5_rank < 0.15).sum()
            n_b_t = ((r5_rank >= 0.15) & (r5_rank < 0.35)).sum()
        else:
            n_a, n_b_t = 0, 0
        rows.append({"date": d_, "total": n_all, "buy_score": n_b,
                       "+pyr": n_bp, "+f1f2": n_bpf, "trackA": n_a, "trackB": n_b_t})
    diag = pd.DataFrame(rows)
    diag.to_csv(OUT / "diag_daily.csv", index=False)

    # 月度统计
    diag["month"] = diag["date"].str[:6]
    print(f"\n## 逐月过滤漏斗 (median 池子 size 每步)\n")
    print(f"  {'month':10s} {'total':6s} {'b_score':8s} {'+pyr':6s} {'+f1f2':6s} "
           f"{'trackA':7s} {'trackB':7s}", flush=True)
    for m_, g in diag.groupby("month"):
        print(f"  {m_:10s} {g['total'].median():6.0f} {g['buy_score'].median():8.0f} "
               f"{g['+pyr'].median():6.0f} {g['+f1f2'].median():6.0f} "
               f"{g['trackA'].median():7.0f} {g['trackB'].median():7.0f}", flush=True)

    # 空池天数
    empty_a = (diag["trackA"] == 0).sum()
    empty_b = (diag["trackB"] == 0).sum()
    empty_both = ((diag["trackA"] == 0) & (diag["trackB"] == 0)).sum()
    print(f"\n## 空池统计 (共 {len(diag)} 日)", flush=True)
    print(f"  轨 A 空池: {empty_a} 日 ({empty_a/len(diag)*100:.0f}%)", flush=True)
    print(f"  轨 B 空池: {empty_b} 日 ({empty_b/len(diag)*100:.0f}%)", flush=True)
    print(f"  两轨全空: {empty_both} 日 ({empty_both/len(diag)*100:.0f}%)", flush=True)

    # 找瓶颈步骤
    print(f"\n## 漏斗瓶颈分析 (每步保留比例)", flush=True)
    avg = diag.mean(numeric_only=True)
    print(f"  total → buy_score [70,85]: {avg['buy_score']/avg['total']*100:.1f}%", flush=True)
    print(f"  buy_score → +pyr<p35: {avg['+pyr']/avg['buy_score']*100:.1f}%", flush=True)
    print(f"  +pyr → +f1f2<0.005: {avg['+f1f2']/avg['+pyr']*100:.1f}%", flush=True)
    print(f"  +f1f2 → trackA(R5 Bot15): {avg['trackA']/(avg['+f1f2']+1e-9)*100:.1f}%", flush=True)

    print(f"\n输出: {OUT / 'diag_daily.csv'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
