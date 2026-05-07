#!/usr/bin/env python3
"""CogAlpha 启发的 8 个流动性/尾部风险因子 + r10 标签.

新增因子 (从 TDX OHLC 计算):
  upward_impact_per_amount  : (high-close) / amount, 上行价格冲击
  range_per_amount          : (high-low) / amount, 流动性脆弱度
  amihud_illiq_5d           : mean(|ret_pct| / amount) 5日, Amihud 非流动性
  kyle_lambda_20d           : abs(ret) 与 volume 20日相关性 (Kyle's lambda 代理)
  tail_risk_5d              : 5日内 low 相对 close 的极端跌幅
  range_compression_60d     : std(range_5d) / mean(range_60d), 波动收敛
  drawdown_recovery_score   : 回撤恢复速度 (days_since_max_dd / dd_magnitude)
  price_impact_asymmetry    : upward_impact - downward_impact, 不对称冲击

新增标签:
  r10: 10 日 close-to-close 收益 (close[+10]/close[0]-1) %
  max_gain_10: 10 日内期间最高涨幅
  max_dd_10: 10 日内期间最大回撤

输出:
  output/cogalpha_features/features.parquet (8 个新因子)
  output/cogalpha_features/labels_10d.parquet (r10/max_gain_10/max_dd_10)
"""
from __future__ import annotations
import os, struct, time, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
OUT_DIR = ROOT / "output" / "cogalpha_features"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")
START = "20230101"
END   = "20260420"


def read_tdx(market, code):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    try: data = p.read_bytes()
    except: return None
    n = len(data) // 32
    if n == 0: return None
    rows = []
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        try: d = dt.date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append((d.strftime("%Y%m%d"),
                     f[1]/100.0, f[2]/100.0, f[3]/100.0, f[4]/100.0,
                     float(f[6])))
    return rows


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def compute_features_and_labels(daily, start, end):
    """单股计算因子 + 10/20 日标签."""
    n = len(daily)
    if n < 70: return [], []

    closes = np.array([r[4] for r in daily])
    opens  = np.array([r[1] for r in daily])
    highs  = np.array([r[2] for r in daily])
    lows   = np.array([r[3] for r in daily])
    vols   = np.array([r[5] for r in daily])
    dates  = [r[0] for r in daily]

    # amount = vol * close (千元单位简化)
    amounts = vols * closes / 1e3

    # 日内冲击因子
    upward_impact   = (highs - closes) / np.maximum(amounts, 1)
    downward_impact = (closes - lows) / np.maximum(amounts, 1)
    range_per_amt   = (highs - lows) / np.maximum(amounts, 1)

    # 滚动统计
    s_close = pd.Series(closes)
    s_amt   = pd.Series(amounts)
    rets    = s_close.pct_change()
    ranges  = pd.Series(highs - lows)

    amihud_5d = (rets.abs() / np.maximum(amounts, 1)).rolling(5, min_periods=3).mean().values
    # Kyle lambda 代理: |ret| 与 volume 的 20 日相关
    abs_ret = rets.abs()
    kyle_corr = abs_ret.rolling(20, min_periods=10).corr(s_amt).values

    # 尾部风险: 5 日内 (low - close[t]) / close[t] 的最低值
    tail_5d = []
    for i in range(n):
        if i < 5: tail_5d.append(np.nan); continue
        recent_low = lows[i-4:i+1].min()
        tail_5d.append((recent_low - closes[i]) / closes[i] * 100)
    tail_5d = np.array(tail_5d)

    # range compression: std(range_5d) / mean(range_60d)
    range_5d_std  = ranges.rolling(5, min_periods=3).std().values
    range_60d_avg = ranges.rolling(60, min_periods=30).mean().values
    range_compress = range_5d_std / np.maximum(range_60d_avg, 1e-6)

    # drawdown recovery: 计算最近 20 日 max_dd 的位置 + 大小
    dd_recovery = []
    for i in range(n):
        if i < 20: dd_recovery.append(np.nan); continue
        window = closes[i-19:i+1]
        peak_idx = int(np.argmax(window))
        peak = window[peak_idx]
        # 当前价对峰值的回撤
        cur_dd = (closes[i] / peak - 1) * 100
        days_since_peak = 19 - peak_idx
        if cur_dd >= -1:  # 回到峰值或没回撤
            dd_recovery.append(0)
        else:
            dd_recovery.append(days_since_peak / abs(cur_dd))
    dd_recovery = np.array(dd_recovery)

    # 不对称冲击 (5日均)
    up_5d = pd.Series(upward_impact).rolling(5, min_periods=3).mean().values
    dn_5d = pd.Series(downward_impact).rolling(5, min_periods=3).mean().values
    asymmetry = up_5d - dn_5d

    # ── 计算每行特征 + 标签 ──
    feat_rows = []; label_rows = []
    LH_10, LH_20 = 10, 20
    for i in range(n - LH_20 - 1):
        td = dates[i]
        if td < start or td > end: continue
        # 标签
        entry_open = opens[i+1] if i+1 < n else None
        if not entry_open or entry_open <= 0: continue
        # r10
        if i + LH_10 < n:
            future_10_close = closes[i + LH_10]
            future_10_high  = highs[i+1: i+1+LH_10].max()
            future_10_low   = lows[i+1: i+1+LH_10].min()
            r10 = (future_10_close / entry_open - 1) * 100
            mg10 = (future_10_high / entry_open - 1) * 100
            md10 = (future_10_low / entry_open - 1) * 100
        else:
            r10 = mg10 = md10 = None
        # r20
        future_20_close = closes[i + LH_20]
        r20 = (future_20_close / entry_open - 1) * 100

        label_rows.append({
            "trade_date": td, "entry_open": round(entry_open, 3),
            "r10": round(r10, 4) if r10 is not None else None,
            "max_gain_10": round(mg10, 4) if mg10 is not None else None,
            "max_dd_10": round(md10, 4) if md10 is not None else None,
            "r20": round(r20, 4),
        })

        feat_rows.append({
            "trade_date": td,
            "upward_impact_per_amount":  round(float(up_5d[i] if not np.isnan(up_5d[i]) else 0), 6),
            "range_per_amount_5d": round(float(pd.Series(range_per_amt).rolling(5).mean().iloc[i] if i >= 5 else 0), 6),
            "amihud_illiq_5d":     round(float(amihud_5d[i] if not np.isnan(amihud_5d[i]) else 0), 8),
            "kyle_lambda_20d":     round(float(kyle_corr[i] if not np.isnan(kyle_corr[i]) else 0), 4),
            "tail_risk_5d":        round(float(tail_5d[i] if not np.isnan(tail_5d[i]) else 0), 3),
            "range_compression":   round(float(range_compress[i] if not np.isnan(range_compress[i]) else 1), 3),
            "drawdown_recovery":   round(float(dd_recovery[i] if not np.isnan(dd_recovery[i]) else 0), 3),
            "price_impact_asym":   round(float(asymmetry[i] if not np.isnan(asymmetry[i]) else 0), 6),
        })
    return feat_rows, label_rows


def main():
    t0 = time.time()
    print("加载 stock list...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code"])
        parts.append(df)
    ts_codes = sorted(pd.concat(parts)["ts_code"].unique())
    print(f"股票数: {len(ts_codes)}")

    all_feat = []; all_label = []
    for i, ts in enumerate(ts_codes):
        if (i+1) % 500 == 0:
            print(f"  [{i+1}/{len(ts_codes)}] {int(time.time()-t0)}s, "
                  f"feat {len(all_feat):,}, label {len(all_label):,}")
        c, m = code_market(ts)
        daily = read_tdx(m, c)
        if not daily: continue
        feat, label = compute_features_and_labels(daily, START, END)
        for r in feat: r["ts_code"] = ts
        for r in label: r["ts_code"] = ts
        all_feat.extend(feat)
        all_label.extend(label)

    feat_df = pd.DataFrame(all_feat)
    label_df = pd.DataFrame(all_label)
    feat_df.to_parquet(OUT_DIR / "features.parquet", index=False)
    label_df.to_parquet(OUT_DIR / "labels_10d.parquet", index=False)
    print(f"\nfeatures: {len(feat_df):,} 行")
    print(f"labels: {len(label_df):,} 行")
    print()
    print("=== 8 因子分布 ===")
    feat_cols = [c for c in feat_df.columns if c not in ("ts_code","trade_date")]
    print(feat_df[feat_cols].describe(percentiles=[0.05, 0.5, 0.95]).round(4).to_string())
    print()
    print("=== 标签分布 ===")
    print(label_df[["r10","r20","max_gain_10","max_dd_10"]].describe(
        percentiles=[0.05, 0.5, 0.95]).round(2).to_string())
    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
