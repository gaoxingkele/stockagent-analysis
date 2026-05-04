#!/usr/bin/env python3
"""V7 P1+P2+P3 综合评估.

P1 pyr_velocity 5 变种过滤对比:
  pyr_velocity_5_60, _10_30, _20_60, _5_30, _10_60 (V6.5 baseline)

P2 industry score:
  在每天每个行业内部, 按 buy_score 排名, 选 top 30%/50% 看 alpha

P3 f1/f2 阈值变种:
  f1: ±1%, ±3%, ±5%, ±8% (4 阈值)
  f2: 同上
  作为过滤器 (取负值, "无强吸筹/派发事件" 段)

输出 output/v7/*.csv
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
OUT_DIR = ROOT / "output" / "v7"
OUT_DIR.mkdir(exist_ok=True)
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

PYR_VEL_VARS = ["pyr_velocity_5_60", "pyr_velocity_10_30",
                  "pyr_velocity_20_60", "pyr_velocity_5_30",
                  "pyr_velocity_10_60"]  # 后者来自 pyramid_v2
F1_VARS = ["f1_main_in_red", "f1_neg1", "f1_neg5", "f1_neg8"]  # 第一个是 V2 默认 (-3%)
F2_VARS = ["f2_main_out_green", "f2_pos1", "f2_pos5", "f2_pos8"]


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
        ROOT / "output" / "v7_extras" / "features.parquet",  # NEW
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

    # 主推荐基础段
    base_mask = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30))
    base = df[base_mask]
    print(f"  Baseline (buy_70_85+SELL_v6≤30): n={len(base):,}, r20={base['r20'].mean():.2f}%", flush=True)

    # ── P1: pyr_velocity 5 变种过滤 ──
    print(f"\n[{int(time.time()-t0)}s] === P1: pyr_velocity 5 变种 (p35 阈值) ===", flush=True)
    p1_rows = []
    for var in PYR_VEL_VARS:
        if var not in df.columns:
            print(f"  ⚠ {var} 缺失"); continue
        thresh = df[var].quantile(0.35)
        seg_mask = base_mask & (df[var] < thresh)
        seg = df[seg_mask]
        if len(seg) < 100: continue
        for code in ETFS_KEY:
            col = f"r20_{code}"
            if col not in seg.columns: continue
            stat = alpha_seg(seg, col)
            if stat:
                p1_rows.append({"variant": var, "thresh_pct": 35,
                                 "thresh": float(thresh), "etf": code, **stat})
    p1_df = pd.DataFrame(p1_rows)
    if not p1_df.empty:
        pivot1 = p1_df[p1_df["etf"]=="sh000300"].sort_values("net_alpha", ascending=False)
        print(pivot1[["variant","n","stock_r20","alpha","net_alpha","win_pct","p_value"]].round(2).to_string(), flush=True)
    p1_df.to_csv(OUT_DIR / "p1_pyr_velocity_variants.csv", index=False, encoding="utf-8-sig")

    # ── P3: f1/f2 阈值条件化 ──
    print(f"\n[{int(time.time()-t0)}s] === P3: f1/f2 阈值变种 (作过滤: |f|<0.005 静默段) ===", flush=True)
    p3_rows = []
    for var in F1_VARS + F2_VARS:
        if var not in df.columns:
            continue
        # 过滤 = 该因子绝对值低 (无强事件) 段
        seg_mask = base_mask & (df[var].abs() < 0.005)
        seg = df[seg_mask]
        if len(seg) < 100: continue
        for code in ETFS_KEY:
            col = f"r20_{code}"
            if col not in seg.columns: continue
            stat = alpha_seg(seg, col)
            if stat:
                p3_rows.append({"variant": var, "etf": code, **stat})
    p3_df = pd.DataFrame(p3_rows)
    if not p3_df.empty:
        pivot3 = p3_df[p3_df["etf"]=="sh000300"].sort_values("net_alpha", ascending=False)
        print(pivot3[["variant","n","stock_r20","alpha","net_alpha","win_pct","p_value"]].round(2).to_string(), flush=True)
    p3_df.to_csv(OUT_DIR / "p3_f1f2_threshold.csv", index=False, encoding="utf-8-sig")

    # ── P2: industry score (cross-section 内排名) ──
    print(f"\n[{int(time.time()-t0)}s] === P2: industry score 设计 ===", flush=True)
    # 在每天每个行业内, 按 buy_score 排名 (越高越好), 取 top 30% 作为 industry-leader
    df["ind_buy_rank"] = df.groupby(["trade_date", "industry"])["buy_score"].rank(pct=True)
    # 只在主推荐基础段上加 industry score 过滤
    p2_rows = []
    for q in [0.7, 0.5, 0.3]:  # top 30%, 50%, 70%
        seg_mask = base_mask & (df["ind_buy_rank"] >= q)
        seg = df[seg_mask]
        if len(seg) < 100: continue
        for code in ETFS_KEY:
            col = f"r20_{code}"
            if col not in seg.columns: continue
            stat = alpha_seg(seg, col)
            if stat:
                p2_rows.append({"filter": f"ind_buy_rank>={q*100:.0f}%",
                                 "q": q, "etf": code, **stat})
    # 同时 try industry-aware 过滤: 只选半导体/电子/化工等 top 行业 (来自 V4 supplements)
    top_industries = ["半导体","元件","化学制品","染料涂料","塑胶","能源金属","半导体设备"]
    seg_mask_ti = base_mask & (df["industry"].isin(top_industries))
    seg_ti = df[seg_mask_ti]
    print(f"  Top 7 行业 baseline: n={len(seg_ti):,}", flush=True)
    for code in ETFS_KEY:
        col = f"r20_{code}"
        if col not in seg_ti.columns: continue
        stat = alpha_seg(seg_ti, col)
        if stat:
            p2_rows.append({"filter": "top7_industry_only", "q": -1, "etf": code, **stat})

    # 也试: top 行业 + pyr_velocity_10_60<p35 双过滤
    if "pyr_velocity_10_60" in df.columns:
        thr = df["pyr_velocity_10_60"].quantile(0.35)
        seg_mask_combo = base_mask & (df["industry"].isin(top_industries)) & (df["pyr_velocity_10_60"] < thr)
        seg_combo = df[seg_mask_combo]
        for code in ETFS_KEY:
            col = f"r20_{code}"
            if col not in seg_combo.columns: continue
            stat = alpha_seg(seg_combo, col)
            if stat:
                p2_rows.append({"filter": "top7_ind + pyr_v10_60<p35", "q": -1, "etf": code, **stat})

    p2_df = pd.DataFrame(p2_rows)
    if not p2_df.empty:
        pivot2 = p2_df[p2_df["etf"]=="sh000300"].sort_values("net_alpha", ascending=False)
        print(pivot2[["filter","n","stock_r20","alpha","net_alpha","win_pct","p_value"]].round(2).to_string(), flush=True)
    p2_df.to_csv(OUT_DIR / "p2_industry_score.csv", index=False, encoding="utf-8-sig")

    print(f"\n总耗时 {time.time()-t0:.0f}s, 输出: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
