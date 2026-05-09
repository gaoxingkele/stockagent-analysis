#!/usr/bin/env python3
"""V6 P2 评估: 多窗口 pyramid 单因子 IC + 过滤实验.

对比 6 个 pyramid 因子 (4 窗口 + 2 派生) 在:
  1. 单因子 RankIC vs r20 (OOS 920K)
  2. 作为 low_pyramid 过滤在 buy_70_85+SELL≤30 段的 alpha vs hs300

输出: output/v6/pyramid_p2.csv
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
OUT_DIR = ROOT / "output" / "v6"
TDX = os.getenv("TDX_DIR", "D:/tdx")
TEST_START = "20250501"
TEST_END   = "20260126"

ETFS_KEY = ["sh000001", "sh000300", "sh000905", "sz399006"]

R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)

PYR_COLS = ["pyr_5d", "pyr_10d", "pyr_20d", "pyr_60d", "pyr_acc_5_20", "pyr_velocity_10_60"]
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
        ROOT / "output" / "pyramid_v2" / "features.parquet",  # NEW
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


def main():
    t0 = time.time()
    print("[0s] 加载 ETF + OOS...", flush=True)
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

    # 评分 (V4 r10/r20 + V6 sell)
    print(f"[{int(time.time()-t0)}s] 预测...", flush=True)
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

    # ── 1. 单因子 RankIC vs r20 ──
    print(f"\n[{int(time.time()-t0)}s] === 1. 6 pyramid 因子单因子 RankIC ===", flush=True)
    print(f"  对照: V6 mfk_pyramid_top_heavy (5d 默认) RankIC=-0.081", flush=True)
    ic_rows = []
    for fc in PYR_COLS:
        if fc not in df.columns: continue
        m = df["r20"].notna() & df[fc].notna() & np.isfinite(df[fc])
        if m.sum() < 1000: continue
        d_clip = df.loc[m, fc].clip(df.loc[m, fc].quantile(0.001),
                                       df.loc[m, fc].quantile(0.999))
        rank_ic = stats.spearmanr(d_clip, df.loc[m, "r20"])[0]
        ic_rows.append({"factor": fc, "n": int(m.sum()),
                          "RankIC_r20": rank_ic,
                          "mean": d_clip.mean(),
                          "std": d_clip.std()})
    ic_df = pd.DataFrame(ic_rows).sort_values("RankIC_r20", key=abs, ascending=False)
    print(ic_df.round(4).to_string(), flush=True)

    # ── 2. 各窗口作为 low_pyramid 过滤的 alpha ──
    print(f"\n[{int(time.time()-t0)}s] === 2. low_pyramid_X 各窗口过滤 alpha ===", flush=True)
    base_mask = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30))
    base = df[base_mask]
    n_base = len(base)
    print(f"  Baseline (buy_70_85 + SELL_v6≤30): n={n_base:,}, r20={base['r20'].mean():.2f}%", flush=True)

    filter_rows = []
    # 各窗口的多个分位阈值: 取 p20, p35, p50 (越严格样本越少)
    for fc in PYR_COLS:
        if fc not in df.columns: continue
        for q in [0.20, 0.35, 0.50]:
            thresh = df[fc].quantile(q)
            f_mask = df[fc] < thresh
            seg = df[base_mask & f_mask]
            if len(seg) < 100: continue
            row = {"filter": f"{fc}<p{int(q*100)}",
                    "thresh": float(thresh), "n": len(seg),
                    "stock_r20": seg["r20"].mean()}
            for code in ETFS_KEY:
                col = f"r20_{code}"
                if col not in seg.columns: continue
                sub = seg[["r20", col]].dropna()
                if len(sub) < 50: continue
                alpha = sub["r20"] - sub[col]
                net_alpha = (sub["r20"] - STOCK_COST) - (sub[col] - ETF_COST)
                row[f"α_vs_{code}"] = alpha.mean()
                row[f"net_α_vs_{code}"] = net_alpha.mean()
                row[f"win_{code}"] = (alpha > 0).mean() * 100
                row[f"p_{code}"] = stats.ttest_1samp(alpha, 0)[1]
            filter_rows.append(row)
    filt_df = pd.DataFrame(filter_rows)
    filt_df.to_csv(OUT_DIR / "pyramid_p2.csv", index=False, encoding="utf-8-sig")

    # 关键透视: 各因子在 p35 阈值下 vs hs300 net alpha 排名
    print(f"\n[{int(time.time()-t0)}s] === p35 阈值下 各窗口 vs hs300 net α (扣交易成本) ===", flush=True)
    p35 = filt_df[filt_df["filter"].str.contains("p35")].copy()
    p35 = p35.sort_values("net_α_vs_sh000300", ascending=False)
    show_cols = ["filter", "n", "stock_r20"]
    for code in ETFS_KEY:
        col = f"net_α_vs_{code}"
        if col in p35.columns: show_cols.append(col)
    print(p35[show_cols].round(2).to_string(), flush=True)

    # baseline (V6 P3 中已有) 对比
    print(f"\n  V6 P3 已知: low_pyramid (mfk_pyramid_top_heavy<0.45) net α vs hs300 = +3.19pp", flush=True)
    print(f"  V6 P3 baseline (buy_70_85+SELL_v6≤30 无过滤) net α vs hs300 = +2.90pp", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
