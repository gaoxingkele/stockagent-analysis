#!/usr/bin/env python3
"""理想多象限 vs ETF 配对实验.

核心问题: 在每个 (ts_code, trade_date) 时点, 我们的"理想多"信号是否跑赢同时点买入 ETF?

5 ETF/指数 baseline:
  sh000001 上证指数
  sh000300 沪深300
  sh000905 中证500
  sh000852 中证1000
  sz399006 创业板指
  sz399303 北证50

输出 output/v4_compare/etf_pairwise.csv:
  对每个 ETF, 对每个段 (理想多 / 中性区 / sell_top 等), 计算:
    - 平均 alpha (stock_r20 - etf_r20)
    - 配对 t-test p 值
    - 胜率 (alpha > 0 比例)
"""
from __future__ import annotations
import os, struct, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import json
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
OUT_DIR = ROOT / "output" / "v4_compare"
TDX = os.getenv("TDX_DIR", "D:/tdx")
TEST_START = "20250501"
TEST_END   = "20260126"

ETFS = [
    ("sh000001", "上证指数"),
    ("sh000300", "沪深300"),
    ("sh000905", "中证500"),
    ("sh000852", "中证1000"),
    ("sz399006", "创业板指"),
    ("sz399303", "北证50"),
]

R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_ANCHOR = (0.27, 0.48, 0.70)
SELL20_ANCHOR = (0.04, 0.43, 0.88)


def read_tdx_close(market_code):
    """读 TDX day 文件返回 (date, close) list."""
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


def compute_etf_future_r20(etf_rows):
    """对每个日期, 计算未来 20 日 close 收益 (close[t+20]/close[t+1] - 1) %"""
    df = pd.DataFrame(etf_rows, columns=["trade_date", "close"])
    # entry = 第二日开盘 (近似用收盘)
    df["entry"] = df["close"].shift(-1)
    df["exit_20"] = df["close"].shift(-21)
    df["etf_r20"] = (df["exit_20"] / df["entry"] - 1) * 100
    return df[["trade_date", "etf_r20"]].dropna()


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
        ROOT / "output" / "cogalpha_features" / "features.parquet",
        ROOT / "output" / "mfk_features" / "features.parquet",
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
    import time
    t0 = time.time()

    # ── ETF future r20 ──
    print("加载 6 个 ETF 未来 r20...")
    etf_dfs = []
    for code, name in ETFS:
        rows = read_tdx_close(code)
        if not rows:
            print(f"  ❌ {code} {name} 无数据"); continue
        etf_df = compute_etf_future_r20(rows)
        etf_df = etf_df.rename(columns={"etf_r20": f"r20_{code}"})
        etf_df = etf_df[(etf_df["trade_date"] >= TEST_START) & (etf_df["trade_date"] <= TEST_END)]
        etf_dfs.append(etf_df)
        print(f"  ✓ {code} {name}: {len(etf_df)} 天, r20 mean={etf_df[f'r20_{code}'].mean():.2f}%")

    # 合并 ETF
    etf_all = etf_dfs[0]
    for d in etf_dfs[1:]:
        etf_all = etf_all.merge(d, on="trade_date", how="outer")

    # ── OOS + 评分 ──
    print(f"[{int(time.time()-t0)}s] 加载 OOS + 预测 + 评分...")
    df = load_oos()
    df["r10_pred"] = predict_model(df, "r10_v4_all")
    df["r20_pred"] = predict_model(df, "r20_v4_all")
    df["sell_10_prob"] = predict_model(df, "sell_10_v4")
    df["sell_20_prob"] = predict_model(df, "sell_20_v4")
    s10 = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df["r20_pred"].values, *R20_ANCHOR)
    df["buy_score"] = 0.5 * s10 + 0.5 * s20
    s10s = _map_anchored(df["sell_10_prob"].values, *SELL10_ANCHOR)
    s20s = _map_anchored(df["sell_20_prob"].values, *SELL20_ANCHOR)
    df["sell_score"] = 0.5 * s10s + 0.5 * s20s

    # 合并 ETF future r20
    df = df.merge(etf_all, on="trade_date", how="left")
    df = df.dropna(subset=["r20"])
    print(f"  样本: {len(df):,}")

    # ── 段级配对 alpha ──
    print(f"\n[{int(time.time()-t0)}s] === 段级 vs ETF 配对 alpha ===")
    bp80, bp20 = df["buy_score"].quantile(0.80), df["buy_score"].quantile(0.20)
    sp80, sp20 = df["sell_score"].quantile(0.80), df["sell_score"].quantile(0.20)
    segments = [
        ("ALL_OOS",     df.index >= 0),
        ("理想多 (BUY≥70+SELL≤30)", (df["buy_score"] >= 70) & (df["sell_score"] <= 30)),
        ("buy_70_85",   (df["buy_score"] >= 70) & (df["buy_score"] <= 85) & (df["sell_score"] <= 30)),
        ("buy_85+",     (df["buy_score"] > 85) & (df["sell_score"] <= 30)),
        ("中性区",       ~((df["buy_score"] >= 70) | (df["buy_score"] <= 30) |
                          (df["sell_score"] >= 70) | (df["sell_score"] <= 30))),
        ("sell_top20",  df["sell_score"] >= sp80),
        ("矛盾 (双高)",  (df["buy_score"] >= 70) & (df["sell_score"] >= 70)),
    ]

    rows = []
    for label, mask in segments:
        g = df[mask].copy()
        n = len(g)
        if n < 100: continue
        row = {"segment": label, "n": n,
               "stock_r20_mean": g["r20"].mean(),
               "stock_r20_win": (g["r20"] > 0).mean() * 100}
        for code, name in ETFS:
            col = f"r20_{code}"
            if col not in g.columns: continue
            sub = g[["r20", col]].dropna()
            if len(sub) < 50: continue
            alpha = (sub["r20"] - sub[col])
            mean_alpha = alpha.mean()
            t_stat, p_val = stats.ttest_1samp(alpha, 0)
            win_vs_etf = (alpha > 0).mean() * 100
            etf_r20_mean = sub[col].mean()
            row[f"{code}_etf_mean"] = etf_r20_mean
            row[f"{code}_alpha_mean"] = mean_alpha
            row[f"{code}_alpha_winrate"] = win_vs_etf
            row[f"{code}_p"] = p_val
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "etf_pairwise.csv", index=False, encoding="utf-8-sig")

    # 打印简洁表
    print(f"\n=== 段级 vs ETF 平均 alpha (alpha = stock_r20 - etf_r20) ===")
    print(f"{'segment':<35} {'n':>8} {'stk_r20':>8} {'win%':>5} | "
          f"{'hs300_α':>9} {'wn%':>4} | {'zz500_α':>9} {'wn%':>4} | "
          f"{'cyb_α':>9} {'wn%':>4} | {'zz1000_α':>9} {'wn%':>4}")
    for _, r in res.iterrows():
        print(f"{r['segment']:<35} {int(r['n']):>8} {r['stock_r20_mean']:>+8.2f} {r['stock_r20_win']:>5.1f} | "
              f"{r.get('sh000300_alpha_mean', 0):>+9.2f} {r.get('sh000300_alpha_winrate', 0):>4.0f} | "
              f"{r.get('sh000905_alpha_mean', 0):>+9.2f} {r.get('sh000905_alpha_winrate', 0):>4.0f} | "
              f"{r.get('sz399006_alpha_mean', 0):>+9.2f} {r.get('sz399006_alpha_winrate', 0):>4.0f} | "
              f"{r.get('sh000852_alpha_mean', 0):>+9.2f} {r.get('sh000852_alpha_winrate', 0):>4.0f}")

    print(f"\n=== p 值 (alpha 显著性, p<0.001 即极显著) ===")
    print(f"{'segment':<35} | {'hs300':>9} {'zz500':>9} {'cyb':>9} {'zz1000':>9} {'sh':>9} {'bj50':>9}")
    for _, r in res.iterrows():
        print(f"{r['segment']:<35} | "
              f"{r.get('sh000300_p', 1):>9.4f} "
              f"{r.get('sh000905_p', 1):>9.4f} "
              f"{r.get('sz399006_p', 1):>9.4f} "
              f"{r.get('sh000852_p', 1):>9.4f} "
              f"{r.get('sh000001_p', 1):>9.4f} "
              f"{r.get('sz399303_p', 1):>9.4f}")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
