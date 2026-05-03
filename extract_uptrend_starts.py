#!/usr/bin/env python3
"""提取"干净起涨点 + 20 天有效上涨段" 训练样本.

筛选规则:
  起涨点 (t0):
    A. close[t0] > MA5[t0]              (站上 5 日均线)
    B. MA5[t0] > MA5[t0-1]              (MA5 向上)
    C. MA5[t0-5..t0-1] 整体平/向下      (从向下/平转向上的拐点)
    D. 成交额[t0] > MA(成交额,5)*1.2    (放量启动)

  20 天有效上涨段 (t0+1 → t0+20):
    E. min(low) >= entry_open * 0.97    (未破起涨点 -3%)
    F. max(high) / entry_open - 1 >= 10% (至少 10% 涨幅)
    G. MA5[t0+20] / MA5[t0+1] - 1 >= 5%  (MA5 整体向上 5%+)
    H. MA5 不能连续 >= 3 天向下         (允许走平)

输出:
  output/uptrend_starts/starts.parquet
    columns: ts_code, trade_date, entry_open, max_gain_20, min_dd_20,
             ma5_overall_pct, max_consec_ma5_down, amount_ratio_5d,
             total_mv, pe, mv_seg, pe_seg, etf_held, industry
  分桶汇总: 按 mv × pe × ETF 看有效样本量
"""
from __future__ import annotations
import os, struct, time, json
import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
ETF_PATH = ROOT / "output" / "etf_analysis" / "stock_to_etfs.json"
OUT_DIR = ROOT / "output" / "uptrend_starts"
OUT_DIR.mkdir(exist_ok=True)

TDX = os.getenv("TDX_DIR", "D:/tdx")

# 时间范围
START = "20230101"
END   = "20260126"

# 起涨点参数
LOOKAHEAD       = 20
DD_TOLERANCE    = -0.03   # 不破起涨点 -3%
MIN_MAX_GAIN    = 0.10    # 至少涨 10%
MIN_MA5_GROWTH  = 0.05    # MA5 期间整体 +5%
MAX_CONSEC_DOWN = 3       # MA5 最多连续走低 2 天 (>=3 reject)
MA5_FLAT_THRESH = 0.003   # MA5 变化 <0.3% 算走平
VOL_BREAKOUT_RATIO = 1.2  # 当日成交额/5日均


def read_tdx(market: str, code: str):
    """读 TDX 日线, 返回 list of (date, open, high, low, close, volume)."""
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
                      float(f[6])))   # volume
    return rows


def code_market(ts: str):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def bucket_mv(v):
    if pd.isna(v): return None
    bil = v / 1e4
    if bil < 50: return "20-50亿"
    if bil < 100: return "50-100亿"
    if bil < 300: return "100-300亿"
    if bil < 1000: return "300-1000亿"
    return "1000亿+"


def bucket_pe(v):
    if pd.isna(v): return None
    if v < 0: return "亏损"
    if v < 15: return "0-15"
    if v < 30: return "15-30"
    if v < 50: return "30-50"
    if v < 100: return "50-100"
    return "100+"


def find_uptrend_starts(daily: list, start_date: str, end_date: str) -> list[dict]:
    """单股扫描所有干净起涨点."""
    if len(daily) < 30: return []
    n = len(daily)
    closes = np.array([r[4] for r in daily])
    opens  = np.array([r[1] for r in daily])
    highs  = np.array([r[2] for r in daily])
    lows   = np.array([r[3] for r in daily])
    volumes = np.array([r[5] for r in daily])
    dates  = [r[0] for r in daily]

    # MA5
    ma5 = pd.Series(closes).rolling(5).mean().values
    ma5_diff = np.concatenate([[0], np.diff(ma5)])
    # 5 日均量
    vol_ma5 = pd.Series(volumes).rolling(5).mean().values

    starts = []
    for t0 in range(10, n - LOOKAHEAD - 1):
        td = dates[t0]
        if td < start_date or td > end_date:
            continue
        if np.isnan(ma5[t0]) or np.isnan(vol_ma5[t0]):
            continue

        # 起涨点条件
        # A. close > MA5
        if closes[t0] <= ma5[t0]: continue
        # B. MA5 向上
        if ma5_diff[t0] <= 0: continue
        # C. MA5 在 t0-5..t0-1 期间整体平/向下 (避免已经在上涨中)
        prev_ma5_chg = (ma5[t0-1] / ma5[t0-5] - 1) if ma5[t0-5] > 0 else 0
        if prev_ma5_chg > 0.005:   # 已经在涨 (>0.5%) 不算起涨
            continue
        # D. 放量
        if vol_ma5[t0] <= 0: continue
        if volumes[t0] / vol_ma5[t0] < VOL_BREAKOUT_RATIO:
            continue

        # 验证未来 20 天
        if t0 + 1 + LOOKAHEAD >= n: continue
        entry_open = opens[t0 + 1]
        if entry_open <= 0: continue
        future_low  = lows[t0+1: t0+1+LOOKAHEAD]
        future_high = highs[t0+1: t0+1+LOOKAHEAD]
        future_ma5  = ma5[t0+1: t0+1+LOOKAHEAD]

        min_low = future_low.min()
        max_high = future_high.max()
        max_gain = max_high / entry_open - 1
        min_dd   = min_low / entry_open - 1

        # E. 不破起涨点 -3%
        if min_dd < DD_TOLERANCE: continue
        # F. 至少 10% 涨幅
        if max_gain < MIN_MAX_GAIN: continue
        # G. MA5 整体向上 >=5%
        if future_ma5[0] <= 0 or future_ma5[-1] / future_ma5[0] - 1 < MIN_MA5_GROWTH:
            continue
        # H. MA5 不能连续 >=3 天向下 (走平 ok)
        consec_down = 0; max_consec = 0
        for i in range(1, len(future_ma5)):
            change = (future_ma5[i] - future_ma5[i-1]) / future_ma5[i-1] if future_ma5[i-1] > 0 else 0
            if change < -MA5_FLAT_THRESH:   # 显著向下
                consec_down += 1
                max_consec = max(max_consec, consec_down)
            else:
                consec_down = 0   # 走平或向上都重置
        if max_consec >= MAX_CONSEC_DOWN: continue

        starts.append({
            "trade_date":     td,
            "entry_open":     round(entry_open, 3),
            "max_gain_20":    round(max_gain * 100, 2),
            "min_dd_20":      round(min_dd * 100, 2),
            "ma5_overall_pct": round((future_ma5[-1]/future_ma5[0] - 1)*100, 2),
            "max_consec_ma5_down": int(max_consec),
            "vol_breakout":   round(volumes[t0] / vol_ma5[t0], 2),
        })
    return starts


def main():
    t0 = time.time()
    print("加载 parquet meta...")
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["ts_code","trade_date","total_mv",
                                          "pe","pe_ttm","industry"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        parts.append(df)
    meta = pd.concat(parts, ignore_index=True)
    print(f"meta: {len(meta)} 行, {meta['ts_code'].nunique()} 股")

    # ETF 持仓
    etf_holders = set()
    if ETF_PATH.exists():
        etf_holders = set(json.loads(ETF_PATH.read_text(encoding="utf-8")).keys())
    print(f"ETF 持仓: {len(etf_holders)} 股")

    print(f"\n扫起涨点 ({START} → {END})...")
    all_starts = []
    ts_codes = meta["ts_code"].unique()
    for i, ts in enumerate(ts_codes):
        if (i+1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(ts_codes)}] {int(elapsed)}s, 起涨点 {len(all_starts)}")
        c, m = code_market(ts)
        daily = read_tdx(m, c)
        if not daily: continue
        starts = find_uptrend_starts(daily, START, END)
        for s in starts:
            s["ts_code"] = ts
            all_starts.append(s)

    if not all_starts:
        print("没找到任何起涨点!")
        return

    df = pd.DataFrame(all_starts)
    print(f"\n总起涨点数: {len(df)}, {df['ts_code'].nunique()} 股, {df['trade_date'].nunique()} 天")

    # 合并 meta
    df = df.merge(meta, on=["ts_code","trade_date"], how="left")
    df["mv_seg"] = df["total_mv"].apply(bucket_mv)
    df["pe_seg"] = df["pe_ttm"].fillna(df["pe"]).apply(bucket_pe)
    df["etf_held"] = df["ts_code"].isin(etf_holders)

    df.to_parquet(OUT_DIR / "starts.parquet", index=False)
    print(f"\n写出 {OUT_DIR / 'starts.parquet'}")

    # 时间分布
    print("\n=== 按年月分布 ===")
    df["_ym"] = df["trade_date"].str[:6]
    print(df.groupby("_ym").size().to_string())

    # 按 max_gain 分级
    print("\n=== 按 max_gain 分级 ===")
    df["gain_tier"] = pd.cut(df["max_gain_20"],
                              bins=[0, 15, 20, 30, 50, 100, 1e9],
                              labels=["10-15%","15-20%","20-30%","30-50%","50-100%","100%+"])
    print(df["gain_tier"].value_counts().sort_index().to_string())

    # 三维分桶
    print("\n=== 三维分桶: MV × PE × ETF ===")
    pivot = df.pivot_table(index=["mv_seg","pe_seg"], columns="etf_held",
                            values="ts_code", aggfunc="count", fill_value=0)
    pivot.columns = ["non_ETF", "ETF_held"]
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.to_string())

    # 行业 Top 15
    print("\n=== 行业 Top 15 ===")
    print(df["industry"].value_counts().head(15).to_string())

    # 综合统计
    print("\n=== 综合统计 ===")
    summary = {
        "total_starts": int(len(df)),
        "n_stocks":     int(df["ts_code"].nunique()),
        "n_dates":      int(df["trade_date"].nunique()),
        "data_period":  [START, END],
        "max_gain_avg": round(df["max_gain_20"].mean(), 2),
        "max_gain_p50": round(df["max_gain_20"].median(), 2),
        "max_gain_p90": round(df["max_gain_20"].quantile(0.9), 2),
        "min_dd_avg":   round(df["min_dd_20"].mean(), 2),
        "ma5_growth_avg": round(df["ma5_overall_pct"].mean(), 2),
        "max_consec_down_avg": round(df["max_consec_ma5_down"].mean(), 2),
        "by_year": df.groupby(df["trade_date"].str[:4]).size().to_dict(),
        "by_mv":   df["mv_seg"].value_counts().to_dict(),
        "by_pe":   df["pe_seg"].value_counts().to_dict(),
        "by_etf":  df["etf_held"].value_counts().to_dict(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    Path(OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
