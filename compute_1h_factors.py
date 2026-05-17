"""1H 因子计算 + r20_1h forward label (2 年全市场).

输入: output/tushare_cache/1h/{ts_code}.parquet
输出: output/1h_factors/factors.parquet (全市场 1H × 因子)

因子设计 (~30 个 1H 特征):
  基础: MA5/10/20/60 / close_to_ma20 / ma20_slope / RSI6/14 / MACD / BOLL_position
  节奏: streak_n / amp_5bar / vol_ratio_20 / gap_open
  突破: break_20bar_high / break_20bar_low
  波动: ret_std_20bar / upper_wick / lower_wick

Label:
  r20_1h = close[t+20] / open[t+1] - 1  (5 交易日 = 20 个 1H bar)
  r4_1h = close[t+4] / open[t+1] - 1   (1 交易日)
"""
from __future__ import annotations
import time, gc
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "tushare_cache" / "1h"
OUT = ROOT / "output" / "1h_factors"
OUT.mkdir(parents=True, exist_ok=True)


def compute_factors_one_stock(df: pd.DataFrame) -> pd.DataFrame:
    """对单股 1H 时序计算因子 + forward label."""
    if len(df) < 60: return pd.DataFrame()
    df = df.sort_values("trade_time").reset_index(drop=True)
    n = len(df)

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["vol"].values
    s = pd.Series(c)

    # MA
    ma5 = s.rolling(5, min_periods=2).mean()
    ma10 = s.rolling(10, min_periods=5).mean()
    ma20 = s.rolling(20, min_periods=10).mean()
    ma60 = s.rolling(60, min_periods=30).mean()

    out = pd.DataFrame({
        "ts_code": df.get("ts_code", "?"),
        "trade_time": df["trade_time"],
        "trade_date": df["trade_date"].astype(str),
        # 基础
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "close": c,
        "close_to_ma20": (s / ma20 - 1) * 100,
        "ma5_to_ma20": (ma5 / ma20 - 1) * 100,
        "ma20_slope_10": (ma20 / ma20.shift(10) - 1) * 100,
    })

    # RSI(6/14)
    delta = s.diff()
    for w in (6, 14):
        gain = delta.where(delta > 0, 0).rolling(w, min_periods=w//2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w, min_periods=w//2).mean()
        rs = gain / loss.replace(0, np.nan)
        out[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    # MACD (12/26/9 在 1H 上)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # BOLL position
    std20 = s.rolling(20, min_periods=10).std()
    out["boll_pos"] = (s - ma20) / (2 * std20)
    out["boll_width"] = (4 * std20) / ma20 * 100

    # 振幅
    out["amp_pct"] = (h - l) / o * 100
    out["amp_5bar_avg"] = pd.Series((h - l) / o).rolling(5).mean() * 100

    # 量能
    vs = pd.Series(v.astype(float))
    out["vol_ma20"] = vs.rolling(20, min_periods=10).mean()
    out["vol_ratio_20"] = vs / out["vol_ma20"]
    out["vol_z60"] = (vs - vs.rolling(60, min_periods=30).mean()) / vs.rolling(60, min_periods=30).std()

    # 连续阳/阴
    is_red = (c > o).astype(int)
    # 用 cumsum 切组找最近连续
    out["bar_is_red"] = is_red
    # 简化版: 最近 5 bar 阳 K 数
    out["red_count_5bar"] = pd.Series(is_red).rolling(5).sum()
    out["red_count_20bar"] = pd.Series(is_red).rolling(20).sum()

    # 突破 (20bar 新高/新低)
    out["break_high_20"] = (c > s.shift(1).rolling(20).max()).astype(int)
    out["break_low_20"] = (c < s.shift(1).rolling(20).min()).astype(int)

    # 上下影线
    bar_range = h - l
    body = np.abs(c - o)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    out["upper_wick_ratio"] = np.where(bar_range > 0, upper / bar_range, 0)
    out["lower_wick_ratio"] = np.where(bar_range > 0, lower / bar_range, 0)
    out["body_ratio"] = np.where(bar_range > 0, body / bar_range, 0)

    # 跳空 (今 bar open vs 昨 bar close)
    out["gap_pct"] = (s.shift(0) / s.shift(1).rolling(1).max() - 1) * 100

    # === Forward Labels ===
    # 入场 = 下一 bar open, 退出 = N bar 后 close
    entry_open = pd.Series(o).shift(-1)
    out["r4_1h"] = (s.shift(-4) / entry_open - 1) * 100      # 1 日
    out["r20_1h"] = (s.shift(-20) / entry_open - 1) * 100    # 5 日 (主 label)
    out["r40_1h"] = (s.shift(-40) / entry_open - 1) * 100    # 10 日

    return out


def main():
    t0 = time.time()
    files = sorted(SRC.glob("*.parquet"))
    print(f"加载 1H 数据: {len(files)} 个文件")
    if len(files) == 0:
        print("无 1H 数据, 退出"); return

    # 分批处理 (每 500 股一批, 控制内存)
    BATCH = 500
    all_parts = []
    n_done = 0
    for i in range(0, len(files), BATCH):
        batch_files = files[i:i+BATCH]
        batch_parts = []
        for f in batch_files:
            df = pd.read_parquet(f)
            df["ts_code"] = f.stem
            feat = compute_factors_one_stock(df)
            if not feat.empty: batch_parts.append(feat)
        if batch_parts:
            batch_df = pd.concat(batch_parts, ignore_index=True)
            all_parts.append(batch_df)
        n_done += len(batch_files)
        print(f"  [{n_done}/{len(files)}] 因子算完, {time.time()-t0:.0f}s, "
              f"内存累计 {sum(len(p) for p in all_parts):,} 行", flush=True)

    big = pd.concat(all_parts, ignore_index=True)
    print(f"\n全市场 1H 因子: {len(big):,} 行 × {len(big.columns)} 列")
    print(f"  时间范围: {big['trade_time'].min()} → {big['trade_time'].max()}")
    print(f"  独立股票: {big['ts_code'].nunique()}")
    print(f"  独立交易日: {big['trade_date'].nunique()}")

    # 输出 parquet
    out_p = OUT / "factors.parquet"
    big.to_parquet(out_p, index=False, compression="snappy")
    sz = out_p.stat().st_size / 1024 / 1024
    print(f"\n输出: {out_p} ({sz:.0f} MB)")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
