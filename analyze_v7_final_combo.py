#!/usr/bin/env python3
"""V7 终极组合: pyr_velocity_20_60 + f1_neg1 双过滤 (无 look-ahead).

V6.5 baseline: buy_70_85+SELL_v6≤30+pyr_velocity_10_60<p35 → net α +4.06pp
V7 update:    替换 _10_60 → _20_60 + 加 f1_neg1 静默过滤
"""
from __future__ import annotations
import os, struct, datetime as dt, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
TDX = os.getenv("TDX_DIR", "D:/tdx")
TEST_START = "20250501"
TEST_END   = "20260126"
ETFS_KEY = ["sh000300", "sh000905", "sz399006"]
R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)
STOCK_COST = 0.5
ETF_COST = 0.2


def read_tdx_close(market_code):
    market = market_code[:2]; code = market_code[2:]
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    data = p.read_bytes()
    n = len(data) // 32
    rows = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append((d.strftime("%Y%m%d"), f[4]/100.0))
    return rows


def compute_etf_future_r20(rows):
    df = pd.DataFrame(rows, columns=["trade_date", "close"])
    df["entry"] = df["close"].shift(-1)
    df["exit"] = df["close"].shift(-21)
    df["r20"] = (df["exit"] / df["entry"] - 1) * 100
    return df[["trade_date", "r20"]].dropna()


def _map_anchored(v, p5, p50, p95):
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, 50.0)
    out = np.where(v <= p5, 0, out)
    out = np.where(v >= p95, 100, out)
    mask_lo = (v > p5) & (v <= p50)
    out = np.where(mask_lo, (v - p5) / (p50 - p5) * 50, out)
    mask_hi = (v > p50) & (v < p95)
    out = np.where(mask_hi, 50 + (v - p50) / (p95 - p50) * 50, out)
    return out


def predict_model(df, name):
    d = PROD_DIR / name
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta.get("industry_map", {})
    df = df.copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), -1)
    )
    return booster.predict(df[feat_cols])


def load_oos():
    PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
    LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
    LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    l20 = pd.read_parquet(LABELS_20, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")
    for path in [
        ROOT / "output" / "amount_features" / "amount_features.parquet",
        ROOT / "output" / "regime_extra" / "regime_extra.parquet",
        ROOT / "output" / "moneyflow" / "features.parquet",
        ROOT / "output" / "moneyflow_v2" / "features.parquet",
        ROOT / "output" / "cogalpha_features" / "features.parquet",
        ROOT / "output" / "mfk_features" / "features.parquet",
        ROOT / "output" / "pyramid_v2" / "features.parquet",
        ROOT / "output" / "v7_extras" / "features.parquet",
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")
    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path,
              columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        full = full.merge(rg, on="trade_date", how="left")
    return full


def alpha_seg(g, etf_col):
    sub = g[["r20", etf_col]].dropna()
    if len(sub) < 30: return None
    alpha = sub["r20"] - sub[etf_col]
    net = (sub["r20"] - STOCK_COST) - (sub[etf_col] - ETF_COST)
    return {"n": len(sub), "stock_r20": sub["r20"].mean(),
            "etf_r20": sub[etf_col].mean(),
            "alpha": alpha.mean(), "net_alpha": net.mean(),
            "win_pct": (alpha > 0).mean() * 100,
            "p_value": stats.ttest_1samp(alpha, 0)[1] if alpha.std() > 0 else np.nan}


def main():
    t0 = time.time()
    etfs = {}
    for code in ETFS_KEY:
        rows = read_tdx_close(code)
        if rows:
            ed = compute_etf_future_r20(rows)
            ed = ed[(ed["trade_date"] >= TEST_START) & (ed["trade_date"] <= TEST_END)]
            ed = ed.rename(columns={"r20": f"r20_{code}"})
            etfs[code] = ed
    df = load_oos()
    df["r10_pred"] = predict_model(df, "r10_v4_all")
    df["r20_pred"] = predict_model(df, "r20_v4_all")
    df["sell_10_v6_prob"] = predict_model(df, "sell_10_v6")
    df["sell_20_v6_prob"] = predict_model(df, "sell_20_v6")
    s10 = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df["r20_pred"].values, *R20_ANCHOR)
    df["buy_score"] = 0.5 * s10 + 0.5 * s20
    s10s = _map_anchored(df["sell_10_v6_prob"].values, *SELL10_V6)
    s20s = _map_anchored(df["sell_20_v6_prob"].values, *SELL20_V6)
    df["sell_score"] = 0.5 * s10s + 0.5 * s20s
    for code in ETFS_KEY:
        if code in etfs:
            df = df.merge(etfs[code], on="trade_date", how="left")
    df = df.dropna(subset=["r20"])

    base_mask = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30))

    p35_v20 = df["pyr_velocity_20_60"].quantile(0.35)
    p35_v10 = df["pyr_velocity_10_60"].quantile(0.35)
    p20_v20 = df["pyr_velocity_20_60"].quantile(0.20)

    combos = [
        ("V6.5 baseline (pyr_v10_60<p35)",
         base_mask & (df["pyr_velocity_10_60"] < p35_v10)),
        ("V7a (pyr_v20_60<p35)",
         base_mask & (df["pyr_velocity_20_60"] < p35_v20)),
        ("V7b (pyr_v20_60<p35 + f1_neg1 静默)",
         base_mask & (df["pyr_velocity_20_60"] < p35_v20) &
         (df["f1_neg1"].abs() < 0.005)),
        ("V7c (pyr_v20_60<p35 + f1_neg1 + f2_pos1 双静默)",
         base_mask & (df["pyr_velocity_20_60"] < p35_v20) &
         (df["f1_neg1"].abs() < 0.005) & (df["f2_pos1"].abs() < 0.005)),
        ("V7d 激进 (pyr_v20_60<p20)",
         base_mask & (df["pyr_velocity_20_60"] < p20_v20)),
        ("V7e 激进 + 双静默 (pyr_v20_60<p20 + f1_neg1+f2_pos1)",
         base_mask & (df["pyr_velocity_20_60"] < p20_v20) &
         (df["f1_neg1"].abs() < 0.005) & (df["f2_pos1"].abs() < 0.005)),
    ]

    print("=== V7 终极组合实验 (按 vs hs300 net α 排序) ===\n", flush=True)
    rows = []
    for label, mask in combos:
        seg = df[mask]
        if len(seg) < 50: continue
        for code in ETFS_KEY:
            col = f"r20_{code}"
            if col not in seg.columns: continue
            stat = alpha_seg(seg, col)
            if stat:
                rows.append({"combo": label, "etf": code, **stat})
    res = pd.DataFrame(rows)

    # 按 hs300 排序
    hs = res[res["etf"]=="sh000300"].sort_values("net_alpha", ascending=False)
    show = ["combo","n","stock_r20","alpha","net_alpha","win_pct","p_value"]
    print(hs[show].round(2).to_string(), flush=True)

    # 同一 combo 在 3 ETF 上 net_alpha
    print(f"\n各 combo vs 3 ETF net α:", flush=True)
    pivot = res.pivot(index="combo", columns="etf", values="net_alpha").round(2)
    n_per = res.groupby("combo")["n"].first()
    pivot["n"] = n_per
    pivot = pivot.sort_values("sh000300", ascending=False)
    print(pivot.to_string(), flush=True)

    res.to_csv(ROOT / "output" / "v7" / "final_combo.csv",
                 index=False, encoding="utf-8-sig")
    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
