#!/usr/bin/env python3
"""市场 regime 自动标注 (3 年).

用沪深300 / 创业板 / 中证500 三指数 OHLC 计算每日 regime 标签:

  bull_policy        政策催化牛: 5 日累计 ≥+8% AND 当日成交放量 1.5x
  bull_fast          快牛       : 20 日斜率 +15%+, RSI>70
  bull_slow_diverge  慢牛分化    : 沪深300 20d 横盘 ±3% AND 中证500 20d > +5%
  bear               熊市       : 60 日累计 ≤ -10%, MA60 向下
  sideways           震荡       : 20 日斜率 ±2% AND 布林带宽 <6%
  mixed              过渡       : 不满足以上任一

输出:
  output/regimes/daily_regime.parquet
    columns: trade_date, regime, regime_id, hs300_ret_5d, hs300_ret_20d,
             cyb_ret_20d, zz500_ret_20d, dispersion, vol_zscore
"""
from __future__ import annotations
import os, struct, json, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "regimes"
OUT_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

START = "20230101"
END   = "20260126"

INDICES = {
    "hs300":  ("sh", "000300"),  # 沪深300
    "cyb":    ("sz", "399006"),  # 创业板指
    "zz500":  ("sh", "000905"),  # 中证500
}

REGIME_ID = {
    "bull_policy":       1,
    "bull_fast":         2,
    "bull_slow_diverge": 3,
    "bear":              4,
    "sideways":          5,
    "mixed":             0,
}


def read_tdx_index(market, code):
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
        rows.append({
            "date": d.strftime("%Y%m%d"),
            "open": f[1] / 100.0, "high": f[2] / 100.0,
            "low":  f[3] / 100.0, "close": f[4] / 100.0,
            "volume": float(f[6]),
        })
    df = pd.DataFrame(rows)
    df["date"] = df["date"].astype(str)
    return df


def main():
    print("加载三大指数...")
    idx = {}
    for name, (m, c) in INDICES.items():
        df = read_tdx_index(m, c)
        if df is None:
            print(f"  ❌ {name} 加载失败")
            continue
        df = df[(df["date"] >= START) & (df["date"] <= END)].sort_values("date").reset_index(drop=True)
        idx[name] = df
        print(f"  {name}: {len(df)} 行 ({df['date'].min()} → {df['date'].max()})")

    if "hs300" not in idx:
        print("❌ 沪深300 不可用, 退出"); return

    # 用沪深300 trade_date 作为基准
    hs = idx["hs300"].copy()
    hs["close"] = hs["close"].astype(float)
    hs["volume"] = hs["volume"].astype(float)

    # 计算指标
    hs["ret_1d"]  = hs["close"].pct_change()
    hs["ret_5d"]  = hs["close"].pct_change(5)
    hs["ret_20d"] = hs["close"].pct_change(20)
    hs["ret_60d"] = hs["close"].pct_change(60)
    hs["ma60"]    = hs["close"].rolling(60).mean()
    hs["ma60_slope"] = hs["ma60"].pct_change(10)
    hs["std20"]   = hs["close"].pct_change().rolling(20).std()
    hs["bband_w"] = (4 * hs["std20"]) * 100
    # RSI 14
    delta = hs["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    hs["rsi14"] = 100 - 100 / (1 + rs)
    # 量能
    hs["vol_ma20"] = hs["volume"].rolling(20).mean()
    hs["vol_ratio"] = hs["volume"] / hs["vol_ma20"]
    hs["vol_z60"]  = (hs["volume"] - hs["volume"].rolling(60).mean()) / hs["volume"].rolling(60).std()

    # 创业板 / 中证500 20 日收益, 用于慢牛分化判定
    if "cyb" in idx:
        cyb = idx["cyb"].set_index("date")
        hs["cyb_ret_20d"] = hs["date"].map(cyb["close"].pct_change(20))
    else:
        hs["cyb_ret_20d"] = np.nan
    if "zz500" in idx:
        zz = idx["zz500"].set_index("date")
        hs["zz500_ret_20d"] = hs["date"].map(zz["close"].pct_change(20))
    else:
        hs["zz500_ret_20d"] = np.nan

    # ── 分类逻辑 ──────────────────────────────────────────────
    def classify(row) -> str:
        r5  = row["ret_5d"]; r20 = row["ret_20d"]; r60 = row["ret_60d"]
        rsi = row["rsi14"]; vol_z = row["vol_z60"]; vol_r = row["vol_ratio"]
        ma60_slope = row["ma60_slope"]; bband = row["bband_w"]
        cyb20 = row["cyb_ret_20d"]; zz20 = row["zz500_ret_20d"]

        # 任意 NaN 退化
        if pd.isna(r20): return "mixed"

        # 1. 政策催化牛: 5 日 +8%, 放量
        if pd.notna(r5) and r5 >= 0.08 and pd.notna(vol_r) and vol_r >= 1.5:
            return "bull_policy"

        # 2. 快牛: 20 日 +15%+, RSI>70
        if r20 >= 0.15 and pd.notna(rsi) and rsi > 70:
            return "bull_fast"

        # 3. 慢牛分化: 沪深300 横盘 ±3% 但中证500 / 创业板涨 +5%+
        hs300_flat = abs(r20) < 0.03
        small_up = ((pd.notna(zz20) and zz20 > 0.05) or
                    (pd.notna(cyb20) and cyb20 > 0.05))
        if hs300_flat and small_up:
            return "bull_slow_diverge"

        # 4. 熊市: 60 日 -10% 且 MA60 向下
        if pd.notna(r60) and r60 <= -0.10 and pd.notna(ma60_slope) and ma60_slope < -0.005:
            return "bear"
        # 单一条件较松版熊市
        if pd.notna(r60) and r60 <= -0.15:
            return "bear"

        # 5. 震荡: 20 日斜率 ±2% AND 布林带宽 <6%
        if abs(r20) < 0.02 and pd.notna(bband) and bband < 6:
            return "sideways"

        # 较宽松的震荡
        if abs(r20) < 0.04:
            return "sideways"

        return "mixed"

    hs["regime"] = hs.apply(classify, axis=1)
    hs["regime_id"] = hs["regime"].map(REGIME_ID)

    # 保留有用列
    out = hs[["date", "regime", "regime_id",
              "ret_5d", "ret_20d", "ret_60d", "rsi14", "vol_ratio", "vol_z60",
              "cyb_ret_20d", "zz500_ret_20d"]].rename(columns={"date":"trade_date"})
    out.to_parquet(OUT_DIR / "daily_regime.parquet", index=False)

    # 分布
    print("\n=== Regime 整体分布 ===")
    print(out["regime"].value_counts().to_string())
    print()
    print("=== 各年 regime 分布 ===")
    out["_year"] = out["trade_date"].str[:4]
    yp = out.pivot_table(index="_year", columns="regime", values="trade_date",
                          aggfunc="count", fill_value=0)
    print(yp.to_string())

    # 最长各 regime 持续段
    print("\n=== 各 regime 持续天数统计 ===")
    out["_block"] = (out["regime"] != out["regime"].shift()).cumsum()
    blocks = out.groupby(["_block","regime"]).agg(
        start=("trade_date","first"),
        end=("trade_date","last"),
        n=("trade_date","count"),
    ).reset_index()
    for r in REGIME_ID:
        sub = blocks[blocks["regime"] == r].sort_values("n", ascending=False).head(3)
        if len(sub) > 0:
            print(f"\n[{r}] Top 3 持续段:")
            for _, row in sub.iterrows():
                print(f"  {row['start']} → {row['end']} ({row['n']} 天)")

    # 保存元数据
    Path(OUT_DIR / "regime_summary.json").write_text(json.dumps({
        "regime_distribution": out["regime"].value_counts().to_dict(),
        "by_year": yp.to_dict(),
        "regime_id_map": REGIME_ID,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n写出 {OUT_DIR / 'daily_regime.parquet'}")


if __name__ == "__main__":
    main()
