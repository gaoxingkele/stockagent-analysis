"""1H 短线差异化因子 (V12.18, Sprint 4.5 升级版).

vs v1 (intraday_filter.py): 加入 1H 独有的"日内行为"特征,
不再复用日线那套 MA/RSI 基础指标.

新增 1H 差异化因子:
1. 日内动量: 上午 vs 下午表现, 当日开盘 vs 当前
2. 量价节奏: 上午成交占比, 单 bar 量比突变 (>2x), 缩量上涨
3. 盘口节奏: 连续 1H 阳/阴 K 数, 1H 平均振幅
4. 微观结构: 涨停/跌停后回吐, 高低点突破时段
5. 跨日跳空: 开盘 vs 昨收 (反映情绪转折)
"""
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def fetch_1h_data(ts_code: str, lookback_days: int = 10,
                   end_date: Optional[str] = None) -> pd.DataFrame:
    """复用 v1 拉数据."""
    import tushare as ts
    from datetime import datetime, timedelta
    if "TUSHARE_TOKEN" not in os.environ:
        from dotenv import load_dotenv; load_dotenv()
    ts.set_token(os.environ["TUSHARE_TOKEN"])
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=lookback_days + 14)
    start_str = start_dt.strftime("%Y%m%d") + " 09:00:00"
    end_str = end_dt.strftime("%Y%m%d") + " 15:00:00"
    try:
        df = ts.pro_bar(ts_code=ts_code, freq="60min",
                         start_date=start_str, end_date=end_str)
        if df is None or df.empty: return pd.DataFrame()
        return df.sort_values("trade_time").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def compute_intraday_features_v2(df: pd.DataFrame) -> dict:
    """计算 1H 差异化短线因子."""
    if df.empty or len(df) < 20:
        return {"valid": False, "n_bars": len(df)}

    df = df.copy()
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["date"] = df["trade_time"].dt.date
    df["hour"] = df["trade_time"].dt.hour

    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["vol"].values
    n = len(df)

    feat = {"valid": True, "n_bars": n, "last_bar_time": str(df["trade_time"].iloc[-1])}

    # ──── 1. 当日动量结构 ────
    today = df[df["date"] == df["date"].iloc[-1]]
    if len(today) >= 2:
        morning = today[today["hour"] <= 11]
        afternoon = today[today["hour"] >= 13]
        if len(morning) and len(afternoon):
            mor_ret = (morning["close"].iloc[-1] / morning["open"].iloc[0] - 1) * 100
            aft_ret = (afternoon["close"].iloc[-1] / afternoon["open"].iloc[0] - 1) * 100
            feat["intraday_morn_ret_pct"] = round(float(mor_ret), 2)
            feat["intraday_aft_ret_pct"] = round(float(aft_ret), 2)
            feat["intraday_aft_strong"] = bool(aft_ret > mor_ret)

    # 当日开盘到当前涨幅
    if len(today):
        day_ret = (today["close"].iloc[-1] / today["open"].iloc[0] - 1) * 100
        feat["intraday_day_ret_pct"] = round(float(day_ret), 2)

    # ──── 2. 量价节奏 ────
    # 单 bar 量比突变 (最后 1 bar vs 前 5 bar avg)
    if n >= 6:
        recent_vol = float(vol[-1])
        prev_avg = float(np.mean(vol[-6:-1]))
        vol_spike = recent_vol / prev_avg if prev_avg > 0 else 1
        feat["vol_spike_ratio"] = round(vol_spike, 2)
        feat["has_vol_spike"] = bool(vol_spike >= 2.0)

    # 上午 vs 下午成交占比
    if len(today) >= 4:
        mor_vol = today[today["hour"] <= 11]["vol"].sum()
        aft_vol = today[today["hour"] >= 13]["vol"].sum()
        total = mor_vol + aft_vol
        feat["morn_vol_ratio"] = round(float(mor_vol / total), 2) if total > 0 else 0.5

    # 缩量上涨 / 放量下跌 (最近 3 bar)
    if n >= 4:
        prices = close[-3:]; vols = vol[-3:]
        prices_prev = close[-6:-3] if n >= 6 else close[:-3]
        vols_prev = vol[-6:-3] if n >= 6 else vol[:-3]
        if len(prices_prev) > 0:
            cur_avg_p = float(np.mean(prices)); cur_avg_v = float(np.mean(vols))
            prev_avg_p = float(np.mean(prices_prev)); prev_avg_v = float(np.mean(vols_prev))
            price_rise = cur_avg_p > prev_avg_p
            vol_shrink = cur_avg_v < prev_avg_v * 0.8
            feat["shrink_rise_signal"] = bool(price_rise and vol_shrink)

    # ──── 3. 盘口节奏 ────
    # 连续阳/阴 K 数
    is_red = close > open_
    last_streak = 1
    for i in range(n-2, -1, -1):
        if is_red[i] == is_red[-1]: last_streak += 1
        else: break
    feat["last_streak_n"] = int(last_streak)
    feat["last_streak_dir"] = "up" if is_red[-1] else "down"

    # 1H 振幅 (最近 5 bar 平均)
    if n >= 5:
        recent_amp = np.mean((high[-5:] - low[-5:]) / open_[-5:]) * 100
        feat["amp_5bar_avg_pct"] = round(float(recent_amp), 2)

    # ──── 4. 微观 / 跨日跳空 ────
    # 隔日跳空: 今日 open vs 昨日 close
    if len(today) >= 1:
        today_open = float(today["open"].iloc[0])
        prev_day = df[df["date"] < df["date"].iloc[-1]]
        if len(prev_day) >= 1:
            prev_close = float(prev_day["close"].iloc[-1])
            gap_pct = (today_open / prev_close - 1) * 100
            feat["gap_pct"] = round(gap_pct, 2)
            feat["has_gap_up"] = bool(gap_pct >= 0.5)
            feat["has_gap_down"] = bool(gap_pct <= -0.5)
            # 跳空后是否回补 (低点是否到 prev_close)
            if "has_gap_up" in feat and feat["has_gap_up"]:
                day_low = float(today["low"].min())
                feat["gap_filled"] = bool(day_low <= prev_close)

    # 突破最近 N bar 新高/新低
    if n >= 20:
        cur_close = float(close[-1])
        recent_max = float(np.max(close[-20:-1]))
        recent_min = float(np.min(close[-20:-1]))
        feat["break_20bar_high"] = bool(cur_close > recent_max)
        feat["break_20bar_low"] = bool(cur_close < recent_min)

    return feat


def score_intraday_v2(f: dict) -> dict:
    """1H v2 综合分 [-1, +1] + 子分项."""
    if not f.get("valid"):
        return {"v2_score": 0.0, "valid": False}

    score = 0.0; reasons = []

    # 1. 日内动量 (权重 0.30)
    if "intraday_day_ret_pct" in f:
        r = f["intraday_day_ret_pct"]
        if r > 1.5: score += 0.20; reasons.append(f"当日涨{r}%")
        elif r < -1.5: score -= 0.20; reasons.append(f"当日跌{r}%")
    if f.get("intraday_aft_strong"): score += 0.10; reasons.append("下午强于上午")

    # 2. 量价节奏 (权重 0.25)
    if f.get("shrink_rise_signal"): score += 0.20; reasons.append("缩量上涨")
    if f.get("has_vol_spike"):
        if f.get("intraday_day_ret_pct", 0) > 0:
            score += 0.10; reasons.append("放量上涨")
        else:
            score -= 0.15; reasons.append("放量下跌")

    # 3. 盘口节奏 (权重 0.20)
    streak = f.get("last_streak_n", 0); dir_ = f.get("last_streak_dir", "")
    if dir_ == "up" and streak >= 3: score += 0.15; reasons.append(f"连阳{streak}H")
    elif dir_ == "down" and streak >= 3: score -= 0.15; reasons.append(f"连阴{streak}H")

    # 4. 跨日 (权重 0.15)
    if f.get("has_gap_up"):
        if not f.get("gap_filled"): score += 0.15; reasons.append("跳空未回补")
        else: score -= 0.05; reasons.append("跳空已回补")
    elif f.get("has_gap_down"):
        score -= 0.15; reasons.append("跳空向下")

    # 5. 突破 (权重 0.10)
    if f.get("break_20bar_high"): score += 0.10; reasons.append("突破20bar高")
    elif f.get("break_20bar_low"): score -= 0.10; reasons.append("跌破20bar低")

    score = float(np.clip(score, -1.0, 1.0))
    return {
        "v2_score": round(score, 3),
        "reasons": "; ".join(reasons[:5]),
        "valid": True,
        **f,
    }


def drill_down_v2(ts_codes: list[str], end_date: str,
                   lookback_days: int = 10, progress_cb=None) -> pd.DataFrame:
    """对一批股做 1H v2 钻取."""
    rows = []
    t0 = time.time()
    for i, ts in enumerate(ts_codes, 1):
        try:
            df = fetch_1h_data(ts, lookback_days, end_date)
            feat = compute_intraday_features_v2(df)
            score = score_intraday_v2(feat)
            rows.append({"ts_code": ts, **score})
        except Exception as e:
            rows.append({"ts_code": ts, "error": str(e)[:80], "valid": False})
        if progress_cb and i % 10 == 0:
            progress_cb(i, len(ts_codes), time.time() - t0)
        time.sleep(0.3)
    return pd.DataFrame(rows)
