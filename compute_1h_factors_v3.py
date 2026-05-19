"""1H 因子 v3 - T+1 交易员适配版 (Sprint 4.6).

vs v2: v2 用 r20_1h (5 日 forward) label, 不适合 T+1 实战.
v3 加 15 个 T+1 因子 + 4 个新 label:
  - r1_next_open: 次日开盘 / 今 bar close - 1 (最短 T+1)
  - r4_next_morn: 次日 11:30 / 今 bar close - 1
  - r8_next_day: 次日 15:00 / 今 bar close - 1
  - r20_1h: 保留 (5 日波段)

输入: output/tushare_cache/1h/{ts_code}.parquet
输出: output/1h_factors/factors_v3.parquet
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "tushare_cache" / "1h"
OUT = ROOT / "output" / "1h_factors"
OUT.mkdir(parents=True, exist_ok=True)

# 复用 v2 的两个函数
import sys
sys.path.insert(0, str(ROOT))
from compute_1h_factors_v2 import compute_v1_factors, compute_v2_factors


def compute_t1_factors(df: pd.DataFrame) -> dict:
    """v3 - 15 个 T+1 因子, 全 vectorized (向量化, 比 v3 原版快 ~50x)."""
    out = {}
    o = df["open"].values; h = df["high"].values; l = df["low"].values
    c = df["close"].values; v = df["vol"].values.astype(float)
    dates = df["trade_date"].astype(str).values

    df_calc = pd.DataFrame({"date": dates, "o": o, "h": h, "l": l, "c": c, "v": v})
    grp = df_calc.groupby("date", sort=True)

    day_open_s = grp["o"].first()
    day_high_s = grp["h"].max()
    day_low_s = grp["l"].min()
    day_close_s = grp["c"].last()
    day_vol_s = grp["v"].sum()

    # === 一次性算所有日级序列 (vectorized via shift) ===
    prev_close_s = day_close_s.shift(1)
    prev_high_s = day_high_s.shift(1)
    prev_low_s = day_low_s.shift(1)
    prev_range = prev_high_s - prev_low_s
    prev_last_pos_s = ((prev_close_s - prev_low_s) / prev_range).where(prev_range > 0, 0.5)

    # 隔夜跳空 = day_open / prev_close - 1
    gap_s = (day_open_s / prev_close_s - 1) * 100

    # 各类滚动统计
    gap_vol_20d_s = gap_s.rolling(20, min_periods=10).std()
    gap_down_freq_s = (gap_s < -0.5).astype(float).rolling(20, min_periods=10).mean()
    gap_up_freq_s = (gap_s > 0.5).astype(float).rolling(20, min_periods=10).mean()
    gap_min_20d_s = gap_s.rolling(20, min_periods=10).min()
    gap_cum_60d_s = gap_s.rolling(60, min_periods=30).sum()

    # 昨日量 vs 前 5 日均
    day_vol_ma5_s = day_vol_s.shift(1).rolling(5, min_periods=3).mean()
    day_vol_spike_s = (day_vol_s / day_vol_ma5_s).shift(1)

    # 涨停 binary (近 20 日封板频率)
    is_limit_s = (day_close_s >= prev_close_s * 1.098).astype(float)
    limit_freq_20d_s = is_limit_s.rolling(20, min_periods=10).mean()

    # 连续 3 日尾盘强势 (基于 prev_last_pos > 0.7)
    daily_strong_close_s = (prev_last_pos_s > 0.7).astype(float)
    consec_strong_s = daily_strong_close_s.rolling(3, min_periods=2).sum()

    # === 单次 map 回 per-bar 数组 (替代 dict + list comp) ===
    # 用 dates 直接 .reindex 或 .loc 一次拿全部
    date_idx = pd.Index(dates)
    def to_bar(s):
        return s.reindex(date_idx).values

    out["overnight_gap_vol_20d"] = to_bar(gap_vol_20d_s)
    out["gap_down_freq_20d"] = to_bar(gap_down_freq_s)
    out["gap_up_freq_20d"] = to_bar(gap_up_freq_s)
    out["gap_neg_max_20d"] = to_bar(gap_min_20d_s)
    out["overnight_cum_60d"] = to_bar(gap_cum_60d_s)

    last_pos_bar = to_bar(prev_last_pos_s)
    out["yesterday_last_close_pos"] = last_pos_bar
    out["yesterday_vol_spike"] = to_bar(day_vol_spike_s)
    out["yesterday_strong_close"] = (last_pos_bar > 0.7).astype(int)
    out["yesterday_weak_close"] = (last_pos_bar < 0.3).astype(int)

    prev_close_bar = to_bar(prev_close_s)
    upper_limit = prev_close_bar * 1.10
    lower_limit = prev_close_bar * 0.90
    out["dist_to_upper_limit_pct"] = (upper_limit / c - 1) * 100
    out["dist_to_lower_limit_pct"] = (c / lower_limit - 1) * 100
    out["limit_freq_20d"] = to_bar(limit_freq_20d_s)

    day_open_bar = grp["o"].transform("first").values
    day_high_bar = grp["h"].transform("max").values
    day_low_bar = grp["l"].transform("min").values
    day_vol_bar = grp["v"].transform("sum").values
    intraday_amp = np.where(day_open_bar > 0, (day_high_bar - day_low_bar) / day_open_bar * 100, np.nan)
    out["intraday_amp_today"] = intraday_amp
    out["vol_per_amp_today"] = np.where(intraday_amp > 0, day_vol_bar / intraday_amp, 0) / 1e6

    out["consec_strong_close_3d"] = to_bar(consec_strong_s)

    return out


def compute_t1_labels(df: pd.DataFrame) -> dict:
    """T+1 交易员真实可执行 label.

    r1_next_open  = 次日 09:30 (第一根 1H 的 open) / 今 bar close - 1
    r4_next_morn  = 次日 11:30 (第二根 1H 的 close) / 今 bar close - 1
    r8_next_day   = 次日 15:00 (最后一根 1H 的 close) / 今 bar close - 1
    """
    out = {}
    dates = df["trade_date"].astype(str).values
    c = df["close"].values
    o = df["open"].values

    df_calc = pd.DataFrame({
        "date": dates, "o": o, "c": c,
    })
    grp = df_calc.groupby("date")

    unique_dates = sorted(set(dates))
    # 次日 first_open / morn_close (第二根) / last_close (最后一根)
    day_first_open_s = grp["o"].first()
    day_last_close_s = grp["c"].last()
    # 第二根 K close (= 上午最后一根, 因为 A 股上午 2 根 1H: 10:30 close + 11:30 close)
    def get_morn_close(g):
        g_sorted = g.sort_values("date")  # 已按 date 分组, 内部按 trade_time 顺序
        if len(g) >= 2: return g["c"].iloc[1]
        return g["c"].iloc[-1]
    day_morn_close_s = grp["c"].apply(lambda g: g.iloc[1] if len(g) >= 2 else g.iloc[-1])

    # vectorized: 次日数据 = shift(-1)
    next_open_s = day_first_open_s.shift(-1)
    next_morn_s = day_morn_close_s.shift(-1)
    next_close_s = day_last_close_s.shift(-1)
    date_idx = pd.Index(dates)
    next_open = next_open_s.reindex(date_idx).values
    next_morn = next_morn_s.reindex(date_idx).values
    next_close = next_close_s.reindex(date_idx).values

    # 以今 bar close 作为入场价
    out["r1_next_open"] = (next_open / c - 1) * 100
    out["r4_next_morn"] = (next_morn / c - 1) * 100
    out["r8_next_day"] = (next_close / c - 1) * 100
    return out


def compute_factors_one_stock(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 60: return pd.DataFrame()
    df = df.sort_values("trade_time").reset_index(drop=True)
    o = df["open"].values; h = df["high"].values; l = df["low"].values
    c = df["close"].values; v = df["vol"].values
    s = pd.Series(c)

    v1_feats = compute_v1_factors(o, h, l, c, v, s)
    v2_feats = compute_v2_factors(df)
    t1_feats = compute_t1_factors(df)
    t1_labs = compute_t1_labels(df)

    out = pd.DataFrame({
        "ts_code": df.get("ts_code", "?"),
        "trade_time": df["trade_time"],
        "trade_date": df["trade_date"].astype(str),
    })
    for k, v_ in v1_feats.items(): out[k] = v_
    for k, v_ in v2_feats.items(): out[k] = v_
    for k, v_ in t1_feats.items(): out[k] = v_

    # 旧 forward labels (r20_1h 等) 也保留
    entry_open = pd.Series(o).shift(-1)
    out["r4_1h"] = (s.shift(-4) / entry_open - 1) * 100
    out["r20_1h"] = (s.shift(-20) / entry_open - 1) * 100
    out["r40_1h"] = (s.shift(-40) / entry_open - 1) * 100
    # T+1 新 labels
    for k, v_ in t1_labs.items(): out[k] = v_

    return out


def main():
    t0 = time.time()
    files = sorted(SRC.glob("*.parquet"))
    print(f"加载 1H 数据: {len(files)} 个文件")
    BATCH = 500
    all_parts = []
    n_done = 0; n_fail = 0
    for i in range(0, len(files), BATCH):
        batch = files[i:i+BATCH]
        parts = []
        for f in batch:
            try:
                df = pd.read_parquet(f)
                df["ts_code"] = f.stem
                feat = compute_factors_one_stock(df)
                if not feat.empty: parts.append(feat)
            except Exception as e:
                n_fail += 1
                if n_fail < 5: print(f"  {f.stem} 失败: {e}")
        if parts: all_parts.append(pd.concat(parts, ignore_index=True))
        n_done += len(batch)
        print(f"  [{n_done}/{len(files)}] {time.time()-t0:.0f}s, "
              f"累计 {sum(len(p) for p in all_parts):,} 行, fail={n_fail}", flush=True)

    big = pd.concat(all_parts, ignore_index=True)
    print(f"\n全市场 1H v3 因子: {len(big):,} × {len(big.columns)}")
    out_p = OUT / "factors_v3.parquet"
    big.to_parquet(out_p, index=False, compression="snappy")
    sz = out_p.stat().st_size / 1024 / 1024
    print(f"输出: {out_p} ({sz:.0f} MB)  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
