"""1H 因子 v2 - 日内独有信号 (用户 A 方案: 重新设计因子工程).

vs v1: v1 是日线技术指标 (MA/RSI/MACD/BOLL) 的 1H 翻版.
v2 新增 30+ 个 1H 独有因子, 捕捉日内序列 / 主动方向 / 跨日跳空 / 龙头特征.

输入: output/tushare_cache/1h/{ts_code}.parquet
输出: output/1h_factors/factors_v2.parquet

设计 (v1 23 个 + v2 30 个 = 53 个):

A. 日内序列 (12个) - 当日 4 根 1H K 的相对结构
   - intra_morning_ret / aft_ret / aft_stronger
   - intra_first/last_bar_amp
   - intra_tail_pump (14-15点涨)
   - intra_high_at_close / low_at_open (binary)
   - intra_close_pos_in_day = (close - low) / (high - low)
   - intra_max_body_bar_idx / max_vol_bar_idx (0-3)
   - intra_day_drift (4 bar slope)

B. 量价同步 (8个) - 主动方向 / 量价配合
   - vp_corr_5bar = corr(bar_ret, bar_vol)
   - bar_active_buy_pct = (close - low) / (high - low)
   - active_buy_ma_5bar
   - vol_explosion_ratio = 单 bar / 前 10 bar 均量
   - vol_per_amp 当日 (流动性)
   - shrink_rise_3bar (binary 缩量上涨)
   - expand_drop_3bar (binary 放量下跌)
   - vol_top_signal (binary 顶部抛压)

C. 跨日序列 (6个)
   - overnight_gap_pct
   - gap_continue_signal / gap_fade_signal
   - multi_day_drift_3d (3 日 EOD slope)
   - yesterday_tail_pump (昨日尾盘强)
   - close_vs_yesterday_high (突破)

D. 龙头/强势 (4个)
   - consec_red_bars (跨日连续阳 K)
   - new_high_count_5bar
   - close_at_day_high_streak
   - pullback_then_rise (3-5bar 回调-2%+反弹)
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


def compute_v1_factors(o, h, l, c, v, s):
    """复用 v1 的 23 个特征 (除已排除的 ma5/10/20/60/vol_ma20/close)."""
    out = {}
    ma5 = s.rolling(5, min_periods=2).mean()
    ma10 = s.rolling(10, min_periods=5).mean()
    ma20 = s.rolling(20, min_periods=10).mean()
    out["close_to_ma20"] = (s / ma20 - 1) * 100
    out["ma5_to_ma20"] = (ma5 / ma20 - 1) * 100
    out["ma20_slope_10"] = (ma20 / ma20.shift(10) - 1) * 100

    delta = s.diff()
    for w in (6, 14):
        gain = delta.where(delta > 0, 0).rolling(w, min_periods=w//2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w, min_periods=w//2).mean()
        rs = gain / loss.replace(0, np.nan)
        out[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    std20 = s.rolling(20, min_periods=10).std()
    out["boll_pos"] = (s - ma20) / (2 * std20)
    out["boll_width"] = (4 * std20) / ma20 * 100

    out["amp_pct"] = (h - l) / np.where(o > 0, o, 1) * 100
    out["amp_5bar_avg"] = pd.Series((h - l) / np.where(o > 0, o, 1)).rolling(5).mean() * 100

    vs = pd.Series(v.astype(float))
    out["vol_ratio_20"] = vs / vs.rolling(20, min_periods=10).mean()
    out["vol_z60"] = (vs - vs.rolling(60, min_periods=30).mean()) / vs.rolling(60, min_periods=30).std()

    is_red = (c > o).astype(int)
    out["bar_is_red"] = is_red
    out["red_count_5bar"] = pd.Series(is_red).rolling(5).sum()
    out["red_count_20bar"] = pd.Series(is_red).rolling(20).sum()

    out["break_high_20"] = (c > s.shift(1).rolling(20).max()).astype(int)
    out["break_low_20"] = (c < s.shift(1).rolling(20).min()).astype(int)

    bar_range = h - l
    body = np.abs(c - o)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    out["upper_wick_ratio"] = np.where(bar_range > 0, upper / bar_range, 0)
    out["lower_wick_ratio"] = np.where(bar_range > 0, lower / bar_range, 0)
    out["body_ratio"] = np.where(bar_range > 0, body / bar_range, 0)

    return out


def compute_v2_factors(df: pd.DataFrame) -> dict:
    """v2 新增 30 个 1H 独有因子. df 已按 trade_time 排序."""
    out = {}
    o = df["open"].values; h = df["high"].values; l = df["low"].values
    c = df["close"].values; v = df["vol"].values.astype(float)
    n = len(df)
    s = pd.Series(c)

    # 提取小时 (假设 trade_time 是 datetime)
    tt = pd.to_datetime(df["trade_time"])
    hours = tt.dt.hour.values
    dates = df["trade_date"].astype(str).values

    # === A. 日内序列 (12) ===
    # 对每个 bar, 找它所在的当日所有 bar 信息. 用 groupby 高效计算.
    df_calc = pd.DataFrame({
        "date": dates, "hour": hours, "idx": np.arange(n),
        "o": o, "h": h, "l": l, "c": c, "v": v,
    })
    # 给每个 (date, bar) 找当日的 day_open / day_high / day_low / day_close
    grp_d = df_calc.groupby("date")
    day_open = grp_d["o"].transform("first").values
    day_high = grp_d["h"].transform("max").values
    day_low = grp_d["l"].transform("min").values
    day_close = grp_d["c"].transform("last").values
    day_n = grp_d["o"].transform("size").values

    # 当日内 bar 索引 (0 = 第一根 1H, 一般 9:30-10:30 → hour=10 实际是 10:30 closeed)
    bar_in_day = grp_d.cumcount().values

    # 上午 (hour <= 11, A股 1H 是 10:30/11:30 closed) + 下午 (hour >= 13/14/15)
    is_morn = (hours <= 11).astype(int)
    is_aft = (hours >= 13).astype(int)

    # 上午最后 bar close & 下午第一 bar open
    morn_only = df_calc[is_morn == 1].groupby("date")
    morn_last_close = morn_only["c"].last() if len(morn_only) else pd.Series(dtype=float)
    morn_first_open = morn_only["o"].first()
    aft_only = df_calc[is_aft == 1].groupby("date")
    aft_last_close = aft_only["c"].last() if len(aft_only) else pd.Series(dtype=float)
    aft_first_open = aft_only["o"].first()
    # map back
    morn_ret_map = ((morn_last_close / morn_first_open - 1) * 100).to_dict() if len(morn_only) else {}
    aft_ret_map = ((aft_last_close / aft_first_open - 1) * 100).to_dict() if len(aft_only) else {}
    morn_ret = np.array([morn_ret_map.get(d_, np.nan) for d_ in dates])
    aft_ret = np.array([aft_ret_map.get(d_, np.nan) for d_ in dates])

    out["intra_morning_ret"] = morn_ret
    out["intra_afternoon_ret"] = aft_ret
    out["intra_aft_stronger"] = (aft_ret > morn_ret).astype(int)

    # 第一/最后 bar 振幅
    first_bar_amp = grp_d["h"].transform("first").values - grp_d["l"].transform("first").values
    first_bar_open = day_open
    out["intra_first_bar_amp"] = first_bar_amp / np.where(first_bar_open > 0, first_bar_open, 1) * 100

    last_bar_amp = grp_d["h"].transform("last").values - grp_d["l"].transform("last").values
    out["intra_last_bar_amp"] = last_bar_amp / np.where(day_open > 0, day_open, 1) * 100

    # 尾盘涨幅 = 当日最后 bar close / 倒数第二 bar close
    # 简化: 当日下午涨幅 (已有 aft_ret)
    out["intra_tail_pump"] = aft_ret  # 同 aft_ret

    # 高/低点是否在 close / open
    high_at_close = (np.abs(c - day_high) < 1e-6) & (bar_in_day == day_n - 1)
    low_at_open = (np.abs(c - day_low) < 1e-6) & (bar_in_day == 0)
    out["intra_high_at_close"] = high_at_close.astype(int)
    out["intra_low_at_open"] = low_at_open.astype(int)

    # 收盘在当日范围内位置 (0 = 最低, 1 = 最高)
    day_range = day_high - day_low
    close_pos = np.where(day_range > 0, (c - day_low) / day_range, 0.5)
    out["intra_close_pos_in_day"] = close_pos

    # 当日 max body / max vol bar idx (0 = morning, larger = afternoon)
    body = np.abs(c - o)
    df_calc["body"] = body
    max_body_idx = df_calc.groupby("date")["body"].transform("idxmax")
    max_body_bar = (max_body_idx - grp_d["idx"].transform("first")).values
    out["intra_max_body_bar_idx"] = max_body_bar

    max_vol_idx = df_calc.groupby("date")["v"].transform("idxmax")
    max_vol_bar = (max_vol_idx - grp_d["idx"].transform("first")).values
    out["intra_max_vol_bar_idx"] = max_vol_bar

    # 当日趋势 (4 bar 简单 slope: last_close - first_open)
    out["intra_day_drift"] = (day_close / day_open - 1) * 100

    # === B. 量价同步 (8) ===
    # vp_corr_5bar: 最近 5 bar (close-open) vs vol 的 corr
    bar_ret = pd.Series((c - o) / np.where(o > 0, o, 1))
    bar_vol = pd.Series(v)
    out["vp_corr_5bar"] = bar_ret.rolling(5).corr(bar_vol).values

    # 主动买入近似
    bar_range_h = h - l
    active_buy = np.where(bar_range_h > 0, (c - l) / bar_range_h, 0.5)
    out["bar_active_buy_pct"] = active_buy
    out["active_buy_ma_5bar"] = pd.Series(active_buy).rolling(5, min_periods=3).mean().values

    # 量比突变 (单 bar / 前 10 bar avg)
    out["vol_explosion_ratio"] = pd.Series(v) / pd.Series(v).shift(1).rolling(10, min_periods=5).mean()

    # 当日 vol_per_amp
    day_amp = (day_high - day_low) / np.where(day_open > 0, day_open, 1) * 100
    day_vol = grp_d["v"].transform("sum")
    out["intra_vol_per_amp"] = np.where(day_amp > 0, day_vol / day_amp, 0)

    # 缩量上涨 / 放量下跌 (连续 3 bar)
    rises = (c > pd.Series(c).shift(1)).astype(int).values
    vol_shrink = (pd.Series(v) < pd.Series(v).shift(1).rolling(3, min_periods=2).mean()).astype(int).values
    rise_3bar = pd.Series(rises).rolling(3).sum().values
    vol_shrink_3bar = pd.Series(vol_shrink).rolling(3).sum().values
    out["shrink_rise_3bar"] = ((rise_3bar >= 2) & (vol_shrink_3bar >= 2)).astype(int)
    drops = (c < pd.Series(c).shift(1)).astype(int).values
    vol_expand = (pd.Series(v) > pd.Series(v).shift(1).rolling(3, min_periods=2).mean() * 1.2).astype(int).values
    drop_3bar = pd.Series(drops).rolling(3).sum().values
    vol_expand_3bar = pd.Series(vol_expand).rolling(3).sum().values
    out["expand_drop_3bar"] = ((drop_3bar >= 2) & (vol_expand_3bar >= 2)).astype(int)

    # 顶部信号: 量比 > 3 + 收盘在 K 下半部 + 涨幅 < 0
    vol_ratio_10 = pd.Series(v) / pd.Series(v).shift(1).rolling(10, min_periods=5).mean()
    out["vol_top_signal"] = ((vol_ratio_10 > 3) &
                                (active_buy < 0.4) &
                                (bar_ret < 0)).astype(int).values

    # === C. 跨日序列 (6) ===
    # 隔日跳空 = 今日第一 bar open / 昨日最后 bar close
    # 取 unique date 的 first_open 和 last_close
    day_first_open_s = grp_d["o"].first()
    day_last_close_s = grp_d["c"].last()
    unique_dates = day_first_open_s.index.tolist()
    overnight_map = {}
    prev_d = None; prev_close_v = None
    for d_ in unique_dates:
        first_o = day_first_open_s[d_]
        if prev_close_v is not None and prev_close_v > 0:
            overnight_map[d_] = (first_o / prev_close_v - 1) * 100
        prev_close_v = day_last_close_s[d_]
    overnight_gap = np.array([overnight_map.get(d_, np.nan) for d_ in dates])
    out["overnight_gap_pct"] = overnight_gap

    # 跳空高开 + 第一 bar 仍涨
    first_bar_ret = (grp_d["c"].transform("first") / grp_d["o"].transform("first") - 1) * 100
    out["gap_continue_signal"] = ((overnight_gap > 0.5) & (first_bar_ret > 0)).astype(int)
    out["gap_fade_signal"] = ((overnight_gap > 0.5) & (first_bar_ret < -0.5)).astype(int)

    # 3 日 EOD 趋势 (EOD = 最后 bar close)
    # 取每日 close, 算 slope
    day_close_s = grp_d["c"].last()
    # 简化: 3 日动量 = today / 3day前 - 1
    day_close_3 = day_close_s.shift(2)  # 3 日前 (含今日则 shift 2)
    drift_map = ((day_close_s / day_close_3 - 1) * 100).to_dict()
    out["multi_day_drift_3d"] = np.array([drift_map.get(d_, np.nan) for d_ in dates])

    # 昨日尾盘强 = 昨日下午涨幅 > 1%
    yesterday_aft_map = {}
    prev_aft = None
    for d_ in unique_dates:
        yesterday_aft_map[d_] = prev_aft
        prev_aft = aft_ret_map.get(d_)
    out["yesterday_tail_pump"] = np.array([
        1 if (yesterday_aft_map.get(d_) is not None and yesterday_aft_map.get(d_) > 1) else 0
        for d_ in dates
    ])

    # 收盘 vs 昨日最高 (突破)
    day_high_s = grp_d["h"].max()
    yesterday_high_map = {}
    prev_high = None
    for d_ in unique_dates:
        yesterday_high_map[d_] = prev_high
        prev_high = day_high_s[d_]
    yh = np.array([yesterday_high_map.get(d_) if yesterday_high_map.get(d_) else np.nan for d_ in dates])
    out["close_vs_yesterday_high"] = (c / yh - 1) * 100

    # === D. 龙头/强势 (4) ===
    # 连续阳 K (跨日)
    is_red = (c > o).astype(int)
    streak = np.zeros(n, dtype=int)
    cur = 0
    for i in range(n):
        cur = cur + 1 if is_red[i] else 0
        streak[i] = cur
    out["consec_red_bars"] = streak

    # 最近 5 bar 创新高次数
    new_high = (c > pd.Series(c).shift(1).rolling(20).max()).astype(int)
    out["new_high_count_5bar"] = pd.Series(new_high).rolling(5).sum().values

    # 连续 N 日收盘 = 当日最高 (尾盘强势封板近似)
    close_eq_high = ((c == day_high) & (bar_in_day == day_n - 1)).astype(int)
    # 这个是按 bar 给, 大多 0. 取 5 bar rolling sum 作近似
    out["close_at_day_high_streak"] = pd.Series(close_eq_high).rolling(5).sum().values

    # 回调后反弹: 5 bar 前曾跌 -2%, 之后 5 bar 涨回原位
    ret_5bar = (s / s.shift(5) - 1) * 100
    ret_min_5bar = s.rolling(5).min().values
    min_drop_pct = (ret_min_5bar / s.shift(5).values - 1) * 100
    pullback_then_rise = ((ret_5bar > -0.5) & (min_drop_pct < -2.0)).astype(int)
    out["pullback_then_rise"] = pullback_then_rise

    return out


def compute_factors_one_stock(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 60: return pd.DataFrame()
    df = df.sort_values("trade_time").reset_index(drop=True)
    n = len(df)
    o = df["open"].values; h = df["high"].values; l = df["low"].values
    c = df["close"].values; v = df["vol"].values
    s = pd.Series(c)

    # v1 因子 (除排除的)
    v1_feats = compute_v1_factors(o, h, l, c, v, s)

    # v2 因子
    v2_feats = compute_v2_factors(df)

    out = pd.DataFrame({
        "ts_code": df.get("ts_code", "?"),
        "trade_time": df["trade_time"],
        "trade_date": df["trade_date"].astype(str),
    })
    for k, v_ in v1_feats.items(): out[k] = v_
    for k, v_ in v2_feats.items(): out[k] = v_

    # Forward labels
    entry_open = pd.Series(o).shift(-1)
    out["r4_1h"] = (s.shift(-4) / entry_open - 1) * 100
    out["r20_1h"] = (s.shift(-20) / entry_open - 1) * 100
    out["r40_1h"] = (s.shift(-40) / entry_open - 1) * 100
    return out


def main():
    t0 = time.time()
    files = sorted(SRC.glob("*.parquet"))
    print(f"加载 1H 数据: {len(files)} 个文件")
    BATCH = 500
    all_parts = []
    n_done = 0
    n_fail = 0
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
    print(f"\n全市场 1H v2 因子: {len(big):,} × {len(big.columns)}")
    out_p = OUT / "factors_v2.parquet"
    big.to_parquet(out_p, index=False, compression="snappy")
    sz = out_p.stat().st_size / 1024 / 1024
    print(f"输出: {out_p} ({sz:.0f} MB)  耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
