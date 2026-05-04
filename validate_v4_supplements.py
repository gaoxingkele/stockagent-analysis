#!/usr/bin/env python3
"""V4/V6.5 主推荐段在不同维度的稳定性验证.

主推荐: buy_70_85 + SELL_v6≤30 + pyr_velocity_10_60 < p35

测试维度:
  1. 月度稳定性 (9 个月各自 alpha)
  2. mv_bucket 分层 (大/中/小盘)
  3. regime_id 分层 (市场状态)
  4. industry 分层 (top/bot 5)

关键问题: alpha 是否集中在某月/某 bucket/某 regime/某行业?
若集中 → 系统脆弱; 若分散 → 鲁棒.

输出: output/v6/supplements/{monthly.csv, mv_bucket.csv, regime.csv, industry.csv}
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
OUT_DIR = ROOT / "output" / "v6" / "supplements"
OUT_DIR.mkdir(exist_ok=True, parents=True)
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
    booster = lgb.Booster(model_file=str(d / "classifier.txt"))
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


def alpha_segment(g, etf_col):
    sub = g[["r20", etf_col]].dropna()
    if len(sub) < 30: return None
    alpha = sub["r20"] - sub[etf_col]
    net = (sub["r20"] - STOCK_COST) - (sub[etf_col] - ETF_COST)
    return {
        "n": len(sub),
        "stock_r20": sub["r20"].mean(),
        f"etf_r20": sub[etf_col].mean(),
        "alpha": alpha.mean(),
        "net_alpha": net.mean(),
        "win_pct": (alpha > 0).mean() * 100,
        "p_value": stats.ttest_1samp(alpha, 0)[1] if alpha.std() > 0 else np.nan,
    }


def main():
    t0 = time.time()
    print(f"[{int(time.time()-t0)}s] 加载 ETF + OOS...", flush=True)

    etfs = {}
    for code in ETFS_KEY:
        rows = read_tdx_close(code)
        if rows:
            ed = compute_etf_future_r20(rows)
            ed = ed[(ed["trade_date"] >= TEST_START) & (ed["trade_date"] <= TEST_END)]
            ed = ed.rename(columns={"r20": f"r20_{code}"})
            etfs[code] = ed

    df = load_oos()
    print(f"[{int(time.time()-t0)}s] OOS shape: {df.shape}", flush=True)

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

    # 主推荐段
    pyr_p35 = df["pyr_velocity_10_60"].quantile(0.35)
    rec_mask = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30) &
                  (df["pyr_velocity_10_60"] < pyr_p35))
    rec = df[rec_mask].copy()
    print(f"[{int(time.time()-t0)}s] 主推荐 (buy_70_85+SELL≤30+pyr_v<p35): n={len(rec):,}, r20={rec['r20'].mean():.2f}%",
          flush=True)
    rec["yyyymm"] = rec["trade_date"].astype(str).str[:6]

    # ── 1. 月度稳定性 ──
    print(f"\n[{int(time.time()-t0)}s] === 1. 月度稳定性 ===", flush=True)
    monthly_rows = []
    for m, g in rec.groupby("yyyymm"):
        for code in ETFS_KEY:
            col = f"r20_{code}"
            stat = alpha_segment(g, col)
            if stat:
                monthly_rows.append({"month": m, "etf": code, **stat})
    mdf = pd.DataFrame(monthly_rows)
    pivot_m = mdf.pivot_table(index="month", columns="etf",
                                values="net_alpha", aggfunc="first").round(2)
    pivot_n = mdf.pivot_table(index="month", columns="etf",
                                values="n", aggfunc="first")
    pivot_m["n"] = pivot_n.iloc[:, 0]
    print(pivot_m.to_string(), flush=True)
    mdf.to_csv(OUT_DIR / "monthly.csv", index=False, encoding="utf-8-sig")

    # ── 2. mv_bucket 分层 ──
    print(f"\n[{int(time.time()-t0)}s] === 2. mv_bucket 分层 (市值) ===", flush=True)
    mv_rows = []
    if "mv_bucket" in rec.columns:
        for mv, g in rec.groupby("mv_bucket"):
            for code in ETFS_KEY:
                col = f"r20_{code}"
                stat = alpha_segment(g, col)
                if stat:
                    mv_rows.append({"mv_bucket": mv, "etf": code, **stat})
    mvdf = pd.DataFrame(mv_rows)
    if not mvdf.empty:
        pivot_mv = mvdf.pivot_table(index="mv_bucket", columns="etf",
                                       values="net_alpha", aggfunc="first").round(2)
        pivot_mv["n"] = mvdf.groupby("mv_bucket")["n"].first()
        print(pivot_mv.to_string(), flush=True)
    mvdf.to_csv(OUT_DIR / "mv_bucket.csv", index=False, encoding="utf-8-sig")

    # ── 3. regime_id 分层 ──
    print(f"\n[{int(time.time()-t0)}s] === 3. regime_id 分层 ===", flush=True)
    rg_rows = []
    if "regime_id" in rec.columns:
        for rg, g in rec.groupby("regime_id"):
            for code in ETFS_KEY:
                col = f"r20_{code}"
                stat = alpha_segment(g, col)
                if stat:
                    rg_rows.append({"regime_id": int(rg), "etf": code, **stat})
    rgdf = pd.DataFrame(rg_rows)
    if not rgdf.empty:
        pivot_rg = rgdf.pivot_table(index="regime_id", columns="etf",
                                       values="net_alpha", aggfunc="first").round(2)
        pivot_rg["n"] = rgdf.groupby("regime_id")["n"].first()
        print(pivot_rg.to_string(), flush=True)
    rgdf.to_csv(OUT_DIR / "regime.csv", index=False, encoding="utf-8-sig")

    # ── 4. industry 分层 ──
    print(f"\n[{int(time.time()-t0)}s] === 4. industry 分层 (按样本量 top 15) ===", flush=True)
    ind_rows = []
    if "industry" in rec.columns:
        for ind, g in rec.groupby("industry"):
            if len(g) < 100: continue
            for code in ETFS_KEY:
                col = f"r20_{code}"
                stat = alpha_segment(g, col)
                if stat:
                    ind_rows.append({"industry": ind, "etf": code, **stat})
    inddf = pd.DataFrame(ind_rows)
    if not inddf.empty:
        # 按 hs300 net alpha 排序
        hs300 = inddf[inddf["etf"] == "sh000300"].copy()
        hs300 = hs300.sort_values("net_alpha", ascending=False)
        n_per_ind = inddf.groupby("industry")["n"].first()
        hs300["n"] = hs300["industry"].map(n_per_ind)
        # top 5 + bot 5
        print(f"\nTop 5 行业 (vs hs300 net α 最高):")
        print(hs300.head(5)[["industry", "n", "stock_r20", "net_alpha", "win_pct"]].round(2).to_string())
        print(f"\nBot 5 行业 (vs hs300 net α 最低):")
        print(hs300.tail(5)[["industry", "n", "stock_r20", "net_alpha", "win_pct"]].round(2).to_string())
    inddf.to_csv(OUT_DIR / "industry.csv", index=False, encoding="utf-8-sig")

    print(f"\n总耗时 {time.time()-t0:.0f}s, 输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
