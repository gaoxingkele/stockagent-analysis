"""V7c zombie filter OOS 验证 (2025-05 → 2026-01, 9 月).

对比:
  V7c 5 铁律       基线
  V7c + zombie过滤  新版

数据源:
  - factor_lab_3y/factor_groups (OOS 期因子)
  - labels: labels_10d.parquet + max_gain_labels.parquet
  - daily cache: 算 zombie 用 (MA60 + 横盘度)
  - Tushare 指数: hs300 / cyb / zz500 benchmark
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.zombie_filter import compute_zombie_factors

PROD_DIR = ROOT / "output" / "production"
TEST_START = "20250501"
TEST_END   = "20260126"
R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)
STOCK_COST = 0.5
ETF_COST = 0.2


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
    print("[1/4] 加载 OOS factor + labels...", flush=True)
    PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    print(f"  factor: {len(full)} 行 / {full['ts_code'].nunique()} 股", flush=True)

    l10 = pd.read_parquet(ROOT / "output/cogalpha_features/labels_10d.parquet",
                           columns=["ts_code","trade_date","r10"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    l20 = pd.read_parquet(ROOT / "output/labels/max_gain_labels.parquet",
                           columns=["ts_code","trade_date","r20_close","max_gain_20","max_dd_20"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    l20 = l20.rename(columns={"r20_close": "r20"})
    # 删除 full 里可能已存在的同名列 (cogalpha_features 等可能含 r20/max_gain_20)
    for c in ["r20","max_gain_20","max_dd_20"]:
        if c in full.columns:
            full = full.drop(columns=[c])
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")

    for path in [
        "output/amount_features/amount_features.parquet",
        "output/regime_extra/regime_extra.parquet",
        "output/moneyflow/features.parquet",
        "output/cogalpha_features/features.parquet",
        "output/mfk_features/features.parquet",
        "output/pyramid_v2/features.parquet",
        "output/v7_extras/features.parquet",
    ]:
        ap = ROOT / path
        if not ap.exists(): continue
        d = pd.read_parquet(ap)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")

    rg = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet",
                          columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
    rg["trade_date"] = rg["trade_date"].astype(str)
    rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                              "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
    full = full.merge(rg, on="trade_date", how="left")
    return full


def compute_zombie_all(df_oos: pd.DataFrame) -> pd.DataFrame:
    """对 OOS 期所有 (ts_code, trade_date) 计算 zombie 因子.
    用 daily cache 取每股的完整时序, 算 zombie, 再 merge 回 df.
    """
    print("[2/4] 计算 zombie 因子 (全市场遍历 + MA60 滚动)...", flush=True)
    daily_cache = ROOT / "output/tushare_cache/daily"
    files = sorted(daily_cache.glob("*.parquet"))
    # 只读 OOS_START - 80 → OOS_END
    start_int = int(TEST_START) - 200  # 给 MA60 留够历史
    end_int = int(TEST_END)
    files = [f for f in files if int(f.stem) <= end_int and int(f.stem) >= start_int - 10000]
    parts = [pd.read_parquet(f, columns=["ts_code","trade_date","close"]) for f in files]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    print(f"  daily 数据: {len(big):,} 行 / {big['ts_code'].nunique()} 股", flush=True)

    target_ts = set(df_oos["ts_code"].unique())
    big = big[big["ts_code"].isin(target_ts)]

    out_parts = []
    n_ts = big["ts_code"].nunique()
    t0 = time.time()
    for i, (ts, g) in enumerate(big.groupby("ts_code"), 1):
        if len(g) < 80: continue
        z = compute_zombie_factors(g)
        z = z[(z["trade_date"] >= TEST_START) & (z["trade_date"] <= TEST_END)]
        if z.empty: continue
        out_parts.append(z[["ts_code","trade_date","is_zombie","zombie_days_pct","ma60_slope_short"]])
        if i % 500 == 0:
            print(f"  [{i}/{n_ts}] {time.time()-t0:.0f}s", flush=True)
    zdf = pd.concat(out_parts, ignore_index=True)
    print(f"  zombie 计算完成: {len(zdf):,} 行, 总耗时 {time.time()-t0:.0f}s", flush=True)
    return zdf


def fetch_etf_benchmark():
    """拉 hs300/cyb/zz500 ETF 净 r20."""
    print("[3/4] 拉 ETF benchmark (hs300/cyb/zz500)...", flush=True)
    from dotenv import load_dotenv
    load_dotenv()
    import tushare as ts
    ts.set_token(os.environ['TUSHARE_TOKEN'])
    pro = ts.pro_api()

    INDEX_MAP = {"hs300": "000300.SH", "cyb": "399006.SZ", "zz500": "000905.SH"}
    out = {}
    for name, code in INDEX_MAP.items():
        df = pro.index_daily(ts_code=code, start_date="20250101", end_date="20260301")
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["trade_date"] = df["trade_date"].astype(str)
        # 算 r20 (next-day entry, exit 21 days later)
        df["entry"] = df["close"].shift(-1)
        df["exit"]  = df["close"].shift(-21)
        df[f"r20_{name}"] = (df["exit"] / df["entry"] - 1) * 100
        out[name] = df[["trade_date", f"r20_{name}"]].dropna()
        print(f"  {name}: {len(out[name])} 行", flush=True)
    return out


def alpha_seg(g, etf_col):
    sub = g[["r20", etf_col]].dropna()
    if len(sub) < 30: return None
    alpha = sub["r20"] - sub[etf_col]
    net = (sub["r20"] - STOCK_COST) - (sub[etf_col] - ETF_COST)
    p = stats.ttest_1samp(alpha, 0)[1] if alpha.std() > 0 else np.nan
    return {"n": len(sub), "stock_r20": sub["r20"].mean(),
             "etf_r20": sub[etf_col].mean(),
             "alpha": alpha.mean(), "net_alpha": net.mean(),
             "win_pct": (alpha > 0).mean() * 100, "p_value": p}


def main():
    t0 = time.time()
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

    # zombie
    zdf = compute_zombie_all(df)
    df = df.merge(zdf, on=["ts_code","trade_date"], how="left")
    df["is_zombie"] = df["is_zombie"].fillna(False).astype(bool)

    # ETF benchmark
    etfs = fetch_etf_benchmark()
    for name, ed in etfs.items():
        df = df.merge(ed, on="trade_date", how="left")
    df = df.dropna(subset=["r20"])

    # 5 vs 6 铁律
    p35_v20 = df["pyr_velocity_20_60"].quantile(0.35)
    base_v7c = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30) &
                  (df["pyr_velocity_20_60"] < p35_v20) &
                  (df["f1_neg1"].abs() < 0.005) & (df["f2_pos1"].abs() < 0.005))
    base_v7c_z = base_v7c & (~df["is_zombie"])

    print(f"\n[4/4] 对比 V7c (5 铁律) vs V7c+zombie (6 铁律)\n")
    print(f"OOS 期: {TEST_START} → {TEST_END} (9 月, n_stocks≈{df['ts_code'].nunique()})\n")

    combos = [
        ("V7c 5 铁律 (baseline)", base_v7c),
        ("V7c+zombie 6 铁律 (新)", base_v7c_z),
    ]

    rows = []
    for label, mask in combos:
        seg = df[mask]
        print(f"━━━ {label} ━━━")
        print(f"  样本数: {len(seg):,} (占 OOS {len(seg)/len(df)*100:.2f}%)")
        if len(seg) < 50:
            print("  样本不足"); continue
        for name in ["hs300","cyb","zz500"]:
            col = f"r20_{name}"
            stat = alpha_seg(seg, col)
            if stat:
                print(f"  vs {name:5s}: stock_r20={stat['stock_r20']:+.2f}%, etf_r20={stat['etf_r20']:+.2f}%, "
                      f"alpha={stat['alpha']:+.2f}pp, net_α={stat['net_alpha']:+.2f}pp, "
                      f"胜率={stat['win_pct']:.0f}%, p={stat['p_value']:.3f}")
                rows.append({"combo": label, "etf": name, **stat})
        print()

    # 月度细分
    print("\n━━━ 月度细分 (vs hs300 net α) ━━━")
    df["month"] = df["trade_date"].str[:6]
    print(f'{"月份":<8} {"V7c 5铁律":>14} {"V7c+zombie":>14} {"提升":>8}')
    print('-'*50)
    for m, g in df.groupby("month"):
        s5 = g[base_v7c.loc[g.index]]
        s6 = g[base_v7c_z.loc[g.index]]
        a5 = alpha_seg(s5, "r20_hs300") if len(s5)>30 else None
        a6 = alpha_seg(s6, "r20_hs300") if len(s6)>30 else None
        a5s = f"{a5['net_alpha']:+.2f}pp (n={a5['n']})" if a5 else "—"
        a6s = f"{a6['net_alpha']:+.2f}pp (n={a6['n']})" if a6 else "—"
        diff = f"{a6['net_alpha']-a5['net_alpha']:+.2f}pp" if (a5 and a6) else "—"
        print(f"{m:<8} {a5s:>14} {a6s:>14} {diff:>8}")

    print(f"\n总耗时: {time.time()-t0:.0f}s")
    pd.DataFrame(rows).to_csv(ROOT / "output/v12_inference/zombie_oos_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
