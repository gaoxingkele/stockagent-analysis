# -*- coding: utf-8 -*-
"""各Agent独立回测 — 逐日滚动计算指标+评分，与前瞻收益做IC和分数段分析。

用法:
    python backtest_agents.py
"""
import sys, os, io

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from pathlib import Path
from stockagent_analysis.data_backend import DataBackend


# ── 数据加载 ──────────────────────────────────────────────────

def load_tdx_daily(symbol: str) -> pd.DataFrame | None:
    """从TDX本地加载日线数据。"""
    try:
        backend = DataBackend(mode="combined", default_sources=["tdx", "akshare"])
        df = backend._fetch_kline_tdx(symbol, "day", limit=500)
        if df is not None and not df.empty:
            df = df.rename(columns={"ts": "date"})
            df["date"] = pd.to_datetime(df["date"])
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.reset_index(drop=True)
    except Exception:
        pass
    return None


# ── 滚动指标计算 ──────────────────────────────────────────────

def rolling_indicators(df: pd.DataFrame, min_bars: int = 120) -> pd.DataFrame:
    """对每根K线计算所有需要的技术指标（滚动窗口）。"""
    n = len(df)
    if n < min_bars:
        return pd.DataFrame()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    open_ = df["open"]

    results = []
    for i in range(min_bars, n):
        row = {"idx": i, "close": close.iloc[i]}
        sl = slice(0, i + 1)  # 用到当前bar的所有历史数据
        c = close.iloc[sl]
        h = high.iloc[sl]
        l = low.iloc[sl]
        v = volume.iloc[sl]
        o = open_.iloc[sl]
        ni = len(c)

        # === 基础指标 ===
        # RSI
        rsi = DataBackend._calc_rsi(c)
        row["rsi"] = rsi

        # MACD
        macd = DataBackend._calc_macd(c)
        if macd:
            row["macd_dif"], row["macd_dea"], row["macd_hist"] = macd
        else:
            row["macd_dif"] = row["macd_dea"] = row["macd_hist"] = 0

        # KDJ
        kdj = DataBackend._calc_kdj(h, l, c)
        if kdj:
            row["kdj_k"], row["kdj_d"], row["kdj_j"] = kdj
        else:
            row["kdj_k"] = 50

        # MA system
        ma_vals = {}
        for p in (5, 10, 20, 60, 120):
            if ni >= p:
                mv = float(c.rolling(p).mean().iloc[-1])
                ma_vals[p] = mv
                pct_above = (float(c.iloc[-1]) / mv - 1) * 100 if mv > 0 else 0
                row[f"ma{p}_val"] = mv
                row[f"ma{p}_pct"] = pct_above
        row["ma_vals"] = ma_vals

        # Trend slope
        slope = DataBackend._calc_trend_slope(c)
        row["trend_slope_pct"] = slope or 0

        # ADX
        adx = DataBackend._calc_adx(h, l, c)
        row["adx"] = adx

        # ATR
        atr = DataBackend._calc_atr(h, l, c)
        row["atr"] = atr
        # ATR% (相对价格的波动率)
        cur_price = float(c.iloc[-1])
        row["atr_pct"] = (atr / cur_price * 100) if atr and cur_price > 0 else 0

        # ── quant_alpha 指标 ──
        # ATR5 / ATR20 波动收敛
        atr5 = DataBackend._calc_atr(h, l, c, 5) if ni >= 6 else atr
        atr20 = DataBackend._calc_atr(h, l, c, 20) if ni >= 21 else atr
        row["atr5"] = atr5 or 0
        row["atr20"] = atr20 or 0
        row["atr_converge"] = (atr5 / atr20) if atr5 and atr20 and atr20 > 0 else 1.0
        # 近60日高点距离
        if ni >= 60:
            high60 = float(h.iloc[-60:].max())
            row["near_high60"] = (cur_price / high60) if high60 > 0 else 0
        else:
            row["near_high60"] = 1.0
        # 20日平均成交额
        if ni >= 20:
            amt = (c * v).iloc[-20:]
            row["avg_amount20"] = float(amt.mean())
        else:
            row["avg_amount20"] = float((c * v).iloc[-1]) if ni >= 1 else 0
        # 量能比 (当日成交额 / 20日均额)
        cur_amount = float(c.iloc[-1] * v.iloc[-1]) if ni >= 1 else 0
        row["amount_ratio"] = (cur_amount / row["avg_amount20"]) if row["avg_amount20"] > 0 else 1.0
        # momentum_5
        if ni >= 5:
            row["momentum_5"] = (float(c.iloc[-1]) / float(c.iloc[-5]) - 1) * 100
        else:
            row["momentum_5"] = 0
        # MA20斜率 (已有 trend_slope_pct，这里单独命名以语义清晰)
        row["ma20_slope"] = row["trend_slope_pct"]

        # 一目均衡图 (Ichimoku)
        if ni >= 52:
            # 转换线 (Tenkan-sen): (9日最高+9日最低)/2
            tenkan = (float(h.iloc[-9:].max()) + float(l.iloc[-9:].min())) / 2
            # 基准线 (Kijun-sen): (26日最高+26日最低)/2
            kijun = (float(h.iloc[-26:].max()) + float(l.iloc[-26:].min())) / 2
            # 先行带A (Senkou Span A): (转换线+基准线)/2
            senkou_a = (tenkan + kijun) / 2
            # 先行带B (Senkou Span B): (52日最高+52日最低)/2
            senkou_b = (float(h.iloc[-52:].max()) + float(l.iloc[-52:].min())) / 2
            row["ichi_tenkan"] = tenkan
            row["ichi_kijun"] = kijun
            row["ichi_senkou_a"] = senkou_a
            row["ichi_senkou_b"] = senkou_b
            # 价格相对云的位置
            cloud_top = max(senkou_a, senkou_b)
            cloud_bot = min(senkou_a, senkou_b)
            row["ichi_above_cloud"] = cur_price > cloud_top
            row["ichi_below_cloud"] = cur_price < cloud_bot
            row["ichi_in_cloud"] = cloud_bot <= cur_price <= cloud_top
            # TK交叉: 转换线在基准线上方=多头
            row["ichi_tk_bull"] = tenkan > kijun
            # 价格距云顶/底的百分比
            row["ichi_cloud_dist_pct"] = ((cur_price - cloud_top) / cloud_top * 100) if cloud_top > 0 else 0
        else:
            row["ichi_tenkan"] = row["ichi_kijun"] = cur_price
            row["ichi_senkou_a"] = row["ichi_senkou_b"] = cur_price
            row["ichi_above_cloud"] = False
            row["ichi_below_cloud"] = False
            row["ichi_in_cloud"] = True
            row["ichi_tk_bull"] = False
            row["ichi_cloud_dist_pct"] = 0

        # Momentum
        if ni >= 10:
            row["momentum_10"] = (float(c.iloc[-1]) / float(c.iloc[-10]) - 1) * 100
        else:
            row["momentum_10"] = 0
        if ni >= 20:
            row["momentum_20"] = (float(c.iloc[-1]) / float(c.iloc[-20]) - 1) * 100
        else:
            row["momentum_20"] = 0

        # Volatility
        if ni >= 20:
            rets = c.pct_change().iloc[-20:]
            row["volatility_20"] = float(rets.std() * 100 * (252 ** 0.5)) if len(rets) > 1 else 0
        else:
            row["volatility_20"] = 0

        # Volume ratio
        if ni >= 20:
            v5 = float(v.iloc[-5:].mean()) if ni >= 5 else float(v.iloc[-1])
            v20 = float(v.iloc[-20:].mean())
            row["volume_ratio"] = v5 / v20 if v20 > 0 else 1.0
        else:
            row["volume_ratio"] = 1.0

        # Pct change
        row["pct_chg"] = (float(c.iloc[-1]) / float(c.iloc[-2]) - 1) * 100 if ni >= 2 else 0

        # === 高级指标 (每5根bar算一次，减少计算量) ===
        if i % 3 == 0 or i == min_bars or i == n - 1:
            # Divergence
            div = DataBackend._detect_divergence(c, h, l, ni)
            row["divergence_score"] = div.get("divergence_score", 0)
            row["macd_top_div"] = div.get("macd_top_div", False)
            row["macd_bot_div"] = div.get("macd_bot_div", False)
            row["rsi_top_div"] = div.get("rsi_top_div", False)
            row["rsi_bot_div"] = div.get("rsi_bot_div", False)
            row["macd_div_magnitude"] = div.get("macd_div_magnitude", 0.0)
            row["rsi_div_magnitude"] = div.get("rsi_div_magnitude", 0.0)
            row["div_bars_ago"] = div.get("div_bars_ago", 999)

            # Volume-price signals
            vp = DataBackend._detect_volume_price_signals(c, v, ni)
            row["obv_trend"] = vp.get("obv_trend", "flat")
            row["volume_breakout"] = vp.get("volume_breakout", False)
            row["shrink_pullback"] = vp.get("shrink_pullback", False)
            row["climax_volume"] = vp.get("climax_volume", False)
            row["volume_anomaly"] = vp.get("volume_anomaly", False)
            row["volume_price_score"] = vp.get("volume_price_score", 0)

            # Support/Resistance
            sr = DataBackend._detect_support_resistance(c, h, l, v, ni)
            row["sr_score"] = sr.get("sr_score", 0)

            # Chart patterns
            cp = DataBackend._detect_chart_patterns(c, h, l, ni, volume=v)
            row["chart_pattern_score"] = cp.get("chart_pattern_score", 0)

            # Kline patterns
            kp = DataBackend._detect_advanced_kline_patterns(o, h, l, c, ni)
            bull_conf = max((p.get("confidence", 50) - 50 for p in kp if p.get("direction") == "bullish"), default=0)
            bear_conf = max((p.get("confidence", 50) - 50 for p in kp if p.get("direction") == "bearish"), default=0)
            row["kp_net"] = bull_conf - bear_conf

            # Chanlun
            cl = DataBackend._detect_chanlun_signals(c, h, l, ni)
            row["chanlun_score"] = cl.get("chanlun_score", 0)

            # Trendlines
            tl = DataBackend._construct_trendlines(c, h, l, v, ni)
            row["tl_down_broken"] = False
            row["tl_down_confirmed"] = False
            row["tl_up_broken"] = False
            row["tl_up_confirmed"] = False
            dtl = tl.get("down_trendline", {})
            if isinstance(dtl, dict):
                row["tl_down_broken"] = bool(dtl.get("broken"))
                bo = dtl.get("breakout", {})
                row["tl_down_confirmed"] = bool(bo.get("confirmed")) if isinstance(bo, dict) else False
            utl = tl.get("up_trendline", {})
            if isinstance(utl, dict):
                row["tl_up_broken"] = bool(utl.get("broken"))
                bo = utl.get("breakout", {})
                row["tl_up_confirmed"] = bool(bo.get("confirmed")) if isinstance(bo, dict) else False

            # Continuity stats
            cs = DataBackend._compute_continuity_stats(o, h, l, c, ni)
            row["consecutive_bull"] = cs.get("consecutive_bull", 0)
            row["consecutive_bear"] = cs.get("consecutive_bear", 0)
            row["body_trend"] = cs.get("body_trend", "")
            row["higher_highs"] = cs.get("higher_highs", 0)
            row["lower_lows"] = cs.get("lower_lows", 0)

            # Shadow ratios
            if ni >= 1:
                body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
                total = float(h.iloc[-1]) - float(l.iloc[-1])
                if total > 0:
                    row["upper_shadow_ratio"] = (float(h.iloc[-1]) - max(float(c.iloc[-1]), float(o.iloc[-1]))) / total * 100
                    row["lower_shadow_ratio"] = (min(float(c.iloc[-1]), float(o.iloc[-1])) - float(l.iloc[-1])) / total * 100
                else:
                    row["upper_shadow_ratio"] = 0
                    row["lower_shadow_ratio"] = 0

            # ── 周线/月线聚合K线 (用于多周期共振) ──
            # 周线: 每5根日线聚合一根，取最近3根
            row["wk_bulls"] = 0    # 最近3根周K阳线数
            row["wk_pattern"] = 0  # 周线形态得分: 阳包阴+5, 晨星+8, 锤子+4, 连阳+3
            row["wk_higher_lows"] = False  # 周线底部抬升
            if ni >= 15:
                wk = []
                for wi in range(3):
                    s_idx = ni - 5 * (wi + 1)
                    e_idx = ni - 5 * wi
                    if s_idx < 0:
                        break
                    wo = float(o.iloc[s_idx])
                    wh = float(h.iloc[s_idx:e_idx].max())
                    wl = float(l.iloc[s_idx:e_idx].min())
                    wc = float(c.iloc[e_idx - 1])
                    wk.append({"o": wo, "h": wh, "l": wl, "c": wc})
                wk.reverse()  # [oldest, ..., newest]
                if len(wk) >= 2:
                    row["wk_bulls"] = sum(1 for w in wk if w["c"] > w["o"])
                    # 阳包阴: 上根阴线本根阳线且实体包裹
                    if wk[-2]["c"] < wk[-2]["o"] and wk[-1]["c"] > wk[-1]["o"]:
                        if wk[-1]["c"] > wk[-2]["o"] and wk[-1]["o"] < wk[-2]["c"]:
                            row["wk_pattern"] += 5
                    # 连续两阳
                    if wk[-1]["c"] > wk[-1]["o"] and wk[-2]["c"] > wk[-2]["o"]:
                        row["wk_pattern"] += 3
                    # 锤子线: 下影线长, 实体小
                    newest = wk[-1]
                    body_w = abs(newest["c"] - newest["o"])
                    total_w = newest["h"] - newest["l"]
                    lower_shadow = min(newest["c"], newest["o"]) - newest["l"]
                    if total_w > 0 and lower_shadow / total_w > 0.5 and body_w / total_w < 0.3:
                        row["wk_pattern"] += 4
                    # 底部抬升
                    if len(wk) >= 3 and wk[-1]["l"] > wk[-2]["l"] and wk[-2]["l"] > wk[-3]["l"]:
                        row["wk_higher_lows"] = True

            # 月线: 每20根日线聚合一根，取最近2根
            row["mn_not_bearish"] = True   # 最近两根月线不看空
            row["mn_pattern"] = 0          # 月线形态得分
            if ni >= 40:
                mn = []
                for mi in range(2):
                    s_idx = ni - 20 * (mi + 1)
                    e_idx = ni - 20 * mi
                    if s_idx < 0:
                        break
                    mo = float(o.iloc[s_idx])
                    mh = float(h.iloc[s_idx:e_idx].max())
                    ml = float(l.iloc[s_idx:e_idx].min())
                    mc = float(c.iloc[e_idx - 1])
                    mn.append({"o": mo, "h": mh, "l": ml, "c": mc})
                mn.reverse()
                if len(mn) >= 2:
                    # 看空判断: 大阴线(跌>5%) 或 射击之星
                    for m in mn:
                        pct_m = (m["c"] / m["o"] - 1) * 100 if m["o"] > 0 else 0
                        total_m = m["h"] - m["l"]
                        upper_m = m["h"] - max(m["c"], m["o"])
                        if pct_m < -5:
                            row["mn_not_bearish"] = False  # 大阴线
                        if total_m > 0 and upper_m / total_m > 0.5 and pct_m < -1:
                            row["mn_not_bearish"] = False  # 射击之星
                    # 最近一根月线阳线加分
                    if mn[-1]["c"] > mn[-1]["o"]:
                        row["mn_pattern"] += 3
                    # 两根都是阳线
                    if mn[-1]["c"] > mn[-1]["o"] and mn[-2]["c"] > mn[-2]["o"]:
                        row["mn_pattern"] += 2
        else:
            # 复用上一行的高级指标
            if results:
                prev = results[-1]
                for k in ["divergence_score", "macd_top_div", "macd_bot_div", "rsi_top_div", "rsi_bot_div",
                          "macd_div_magnitude", "rsi_div_magnitude", "div_bars_ago",
                          "obv_trend", "volume_breakout", "shrink_pullback", "climax_volume", "volume_anomaly",
                          "volume_price_score", "sr_score", "chart_pattern_score", "kp_net", "chanlun_score",
                          "tl_down_broken", "tl_down_confirmed", "tl_up_broken", "tl_up_confirmed",
                          "consecutive_bull", "consecutive_bear", "body_trend", "higher_highs", "lower_lows",
                          "upper_shadow_ratio", "lower_shadow_ratio",
                          "wk_bulls", "wk_pattern", "wk_higher_lows",
                          "mn_not_bearish", "mn_pattern"]:
                    row[k] = prev.get(k, 0)

        results.append(row)

    return pd.DataFrame(results)


# ── Agent评分函数 ──────────────────────────────────────────────

def score_trend_momentum(r) -> float:
    ma_vals = r.get("ma_vals", {})
    periods = sorted(ma_vals.keys())
    ordered = sum(1 for i in range(len(periods)-1) if ma_vals[periods[i]] > ma_vals[periods[i+1]])
    reversed_ = sum(1 for i in range(len(periods)-1) if ma_vals[periods[i]] < ma_vals[periods[i+1]])
    n_pairs = max(1, len(periods)-1)
    ma_score = (ordered - reversed_) / n_pairs * 25.0  # was 20

    slope = r.get("trend_slope_pct", 0)
    slope_score = max(-20, min(20, slope * 100))  # was ±15, coeff 80

    tl_bonus = 0
    if r.get("tl_down_confirmed"): tl_bonus += 15   # was 12
    elif r.get("tl_down_broken"): tl_bonus += 7      # was 5
    if r.get("tl_up_confirmed"): tl_bonus -= 15      # was -12
    elif r.get("tl_up_broken"): tl_bonus -= 7        # was -5

    adx = r.get("adx")
    adx_adj = 0
    if adx is not None:
        if adx > 40: adx_adj = 10.0 if (ma_score + slope_score) > 0 else -10.0  # was ±8
        elif adx > 25: adx_adj = 5.0 if (ma_score + slope_score) > 0 else -5.0  # was ±4

    mom = r.get("momentum_20", 0)
    score = 50 + ma_score + slope_score + tl_bonus + 0.3 * mom + adx_adj
    # 反转: A股趋势动量为反向指标(强趋势后回调概率高)
    score = 100 - score
    return max(0, min(100, score))


def score_capital_liquidity(r) -> float:
    vr = r.get("volume_ratio", 1.0)
    pct = r.get("pct_chg", 0)
    obv = r.get("obv_trend", "flat")
    obv_score = 18 if obv == "up" else (-18 if obv == "down" else 0)  # was ±12
    vr_score = max(-22, min(22, (vr - 1) * 25))  # was ±15, coeff 18
    vp_score = 0
    if r.get("volume_breakout"): vp_score += 16  # was 12
    if r.get("shrink_pullback"): vp_score += 12  # was 8
    if r.get("climax_volume"): vp_score -= 12    # was -8
    if r.get("volume_anomaly"): vp_score += 6    # was 4
    return max(0, min(100, 50 + obv_score + vr_score + vp_score + 0.3 * pct))


def score_divergence(r) -> float:
    """背离评分 — 背离agent职责是检测背离，不是判多空。

    无背离(大多数情况): 中性 → 50
    底背离(看涨): 55 + 强度×新鲜度 → 55~90
    顶背离(看跌): 45 - 强度×新鲜度 → 10~45
    """
    ds = r.get("divergence_score", 0)

    if ds == 0:
        return 50.0

    # ── 有背离信号: 用强度+新鲜度做连续评分 ──
    macd_mag = r.get("macd_div_magnitude", 0.5)
    rsi_mag = r.get("rsi_div_magnitude", 0.5)
    bars_ago = r.get("div_bars_ago", 30)

    # 综合强度: 取两个指标中较大的，避免只有单指标时被平均稀释
    if macd_mag > 0 and rsi_mag > 0:
        strength = macd_mag * 0.6 + rsi_mag * 0.4  # 双重背离加权
    else:
        strength = max(macd_mag, rsi_mag)  # 单指标背离

    # 时间衰减: 0 bars ago → 1.0, 40 bars ago → 0.4, 80+ bars ago → 0.2
    recency = max(0.2, 1.0 - bars_ago / 60)

    effective = strength * recency  # 0~1

    if ds > 0:
        # 底背离(看涨): 55 + effective * 35 → 55~90
        score = 55 + effective * 35
    else:
        # 顶背离(看跌): 45 - effective * 35 → 10~45
        score = 45 - effective * 35

    return max(10, min(90, score))


def score_chanlun(r) -> float:
    cs = r.get("chanlun_score", 0)
    score = 50 + cs * 0.8  # was *0.5, 放宽区分力
    return max(5, min(95, score))


def score_pattern(r) -> float:
    kp_net = r.get("kp_net", 0)
    kp_score = max(-15, min(15, kp_net))
    cp_score = r.get("chart_pattern_score", 0) * 0.5

    # 顶底结构
    upper = r.get("upper_shadow_ratio", 0)
    lower = r.get("lower_shadow_ratio", 0)
    mom = r.get("momentum_10", 0)
    top_sig = 0
    bot_sig = 0
    if upper > 40 and mom < 0: top_sig += 8
    elif upper > 35: top_sig += 4
    if lower > 40 and mom > 0: bot_sig += 8
    elif lower > 35: bot_sig += 4
    structure = (bot_sig - top_sig) * 1.2

    cont_bonus = 0
    if r.get("consecutive_bull", 0) >= 3 and r.get("body_trend") == "escalating": cont_bonus += 3
    if r.get("consecutive_bear", 0) >= 3 and r.get("body_trend") == "escalating": cont_bonus -= 3
    if r.get("higher_highs", 0) >= 3: cont_bonus += 2
    if r.get("lower_lows", 0) >= 3: cont_bonus -= 2

    return max(10, min(90, 50 + kp_score + cp_score + structure + cont_bonus))


def score_sentiment_flow(r) -> float:
    # 无新闻数据，仅用量比+动量作为代理
    vr = r.get("volume_ratio", 1.0)
    mom = r.get("momentum_20", 0)
    pct = r.get("pct_chg", 0)
    mm_score = 0
    if vr > 1.8: mm_score = 8
    elif vr > 1.3: mm_score = 4
    elif vr < 0.5: mm_score = -6
    elif vr < 0.7: mm_score = -3
    return max(0, min(100, 50 + mm_score + 0.2 * mom + 0.15 * pct))


def score_volume_structure(r) -> float:
    vp = r.get("volume_price_score", 0)
    sr = r.get("sr_score", 0)
    return max(10, min(90, 50 + vp * 0.6 + sr * 0.4))


def score_resonance(r) -> float:
    """多周期共振评分 — 高级别趋势主导 + 空头一票否决。

    核心原则:
    - 月线/周线/日线 三级共振多头 → 80~90 (极强信号)
    - 月线多头 + 周线多头 → 65~80 (强多)
    - 月线多头 + 其他不明 → 55~65 (温和偏多, 单月线多头无妨)
    - 月线空头(无论日线如何) → 25~40 (警惕, 高级别一票否决)
    - 月线空头 + 日线"假"多头 → 20~35 (额外扣分)
    - 周线/月线 共振空头 → 10~25 (做空信号)
    - 中性 → 50
    """
    ma5 = r.get("ma5_val", 0)
    ma20 = r.get("ma20_val", 0)
    ma60 = r.get("ma60_val", 0)
    ma120 = r.get("ma120_val", 0)
    price = r.get("close", 0)
    if price <= 0:
        return 50.0

    ma20_pct = r.get("ma20_pct", 0)
    slope = r.get("trend_slope_pct", 0)
    mom20 = r.get("momentum_20", 0)

    # ── 月线趋势判定 (price vs MA60/MA120, MA20 vs MA60) ──
    # mn_state: +2=强多, +1=多, 0=中性, -1=空, -2=强空
    mn_state = 0
    if ma60 > 0 and ma120 > 0:
        ma60_pct = (price / ma60 - 1) * 100
        ma120_pct = (price / ma120 - 1) * 100
        # 双均线之上 + MA20 > MA60 = 强多
        if price > ma60 and price > ma120 and ma20 > ma60 and ma60_pct > 3:
            mn_state = 2
        elif price > ma60 and ma20 > ma60:
            mn_state = 1
        elif price < ma60 and price < ma120 and ma20 < ma60 and ma60_pct < -5:
            mn_state = -2  # 双均线下方 + 严重破位
        elif price < ma60 and ma20 < ma60:
            mn_state = -1
        # 否则保持中性
    elif ma60 > 0:
        if price > ma60 * 1.03 and ma20 > ma60:
            mn_state = 1
        elif price < ma60 * 0.95:
            mn_state = -1

    # 月K形态修正
    if not r.get("mn_not_bearish", True):
        mn_state = min(mn_state, -1)  # 月线大阴线 强制至少-1

    # ── 周线趋势判定 (price vs MA20, slope, MA5 vs MA20) ──
    wk_state = 0
    if ma5 > 0 and ma20 > 0:
        if price > ma20 and ma5 > ma20 and slope > 0.05 and ma20_pct > 3:
            wk_state = 2  # 强多
        elif price > ma20 and ma5 > ma20 and slope > 0:
            wk_state = 1
        elif price < ma20 and ma5 < ma20 and slope < -0.05:
            wk_state = -2
        elif price < ma20 and ma5 < ma20:
            wk_state = -1

    # 周K形态修正
    wk_bulls = r.get("wk_bulls", 0)
    if wk_bulls >= 3 and wk_state >= 0:
        wk_state = max(wk_state, 1)
    if wk_bulls == 0 and wk_state <= 0:
        wk_state = min(wk_state, -1)

    # ── 日线趋势判定 ──
    dy_state = 0
    if slope > 0.03 and mom20 > 5:
        dy_state = 2
    elif slope > 0 and mom20 > 0:
        dy_state = 1
    elif slope < -0.03 and mom20 < -5:
        dy_state = -2
    elif slope < 0 and mom20 < 0:
        dy_state = -1

    # ── 共振评分 (核心逻辑) ──
    score = 50.0

    # 三级共振 (最强信号)
    if mn_state >= 1 and wk_state >= 1 and dy_state >= 1:
        score = 75 + (mn_state + wk_state + dy_state - 3) * 4  # 75~87
    elif mn_state <= -1 and wk_state <= -1 and dy_state <= -1:
        score = 25 - (-mn_state - wk_state - dy_state - 3) * 4  # 13~25

    # 月线主导 (高级别)
    elif mn_state == 2:
        if wk_state >= 1:
            score = 70  # 月强多 + 周多 = 强多
        elif wk_state == 0:
            score = 60  # 月强多 但周线观望 = 偏多
        else:
            score = 50  # 月强多 但周线开始空 = 中性
    elif mn_state == 1:
        if wk_state >= 1:
            score = 65
        elif wk_state == 0:
            score = 55  # 月线多头, 周线不明 = 温和偏多 (用户原意: 也无妨)
        else:
            score = 45

    # 月线空头 一票否决
    elif mn_state == -2:
        # 即便日线多头, 月线深度破位 -> 极度警惕
        if dy_state >= 1:
            score = 22  # 假突破警告
        else:
            score = 18
    elif mn_state == -1:
        if dy_state >= 1:
            # 月线空头 但日线多头 = 警惕假反弹
            score = 32
        elif wk_state <= -1:
            score = 28  # 月+周 共振空头
        else:
            score = 38

    # 月线中性, 周线主导
    else:  # mn_state == 0
        if wk_state == 2:
            score = 65  # 周线强多 主导日线
        elif wk_state == 1:
            score = 58
        elif wk_state == -2:
            score = 32  # 周线空头主导
        elif wk_state == -1:
            score = 42
        elif dy_state >= 1:
            score = 53
        elif dy_state <= -1:
            score = 47

    return max(10, min(90, score))


def score_kline_vision_fallback(r) -> float:
    rsi = r.get("rsi") or 50
    slope = r.get("trend_slope_pct", 0)
    mom = r.get("momentum_20", 0)
    vr = r.get("volume_ratio", 1.0)
    vol = r.get("volatility_20", 0)
    rsi_score = 0
    if rsi > 75: rsi_score = -8
    elif rsi < 25: rsi_score = 8
    slope_score = max(-10, min(10, slope * 60))
    return max(0, min(100, 50 + slope_score + rsi_score + 0.3 * mom + (vr - 1) * 8 - 0.15 * vol))


def score_fundamental_pe(r) -> float:
    # 无PE数据回测中，返回50 (中性)
    return 50.0


def score_ichimoku(r) -> float:
    """一目均衡图评分 — 云上/云下/TK交叉/云距离综合判断。

    云上+TK多头 → 65~85 (强多)
    云中 → 40~60 (震荡)
    云下+TK空头 → 15~35 (强空)
    """
    above = r.get("ichi_above_cloud", False)
    below = r.get("ichi_below_cloud", False)
    in_cloud = r.get("ichi_in_cloud", True)
    tk_bull = r.get("ichi_tk_bull", False)
    cloud_dist = r.get("ichi_cloud_dist_pct", 0)

    # 基础位置分
    if above:
        base = 68   # was 62
    elif below:
        base = 32   # was 38
    else:
        base = 50

    # TK交叉方向
    tk_adj = 12 if tk_bull else -12  # was ±8

    # 距云距离: 越远离云越确认趋势, 但过远可能过度延伸
    dist_adj = max(-15, min(15, cloud_dist * 2.0))  # was ±10, coeff 1.5

    # 转换线与基准线的价格位置作为动量确认
    tenkan = r.get("ichi_tenkan", 0)
    kijun = r.get("ichi_kijun", 0)
    cur = r.get("close", 0)
    price_vs_kijun = 0
    if kijun > 0 and cur > 0:
        pct = (cur / kijun - 1) * 100
        price_vs_kijun = max(-12, min(12, pct * 3))  # was ±8, coeff 2

    score = base + tk_adj + dist_adj + price_vs_kijun
    # 反转: A股一目均衡图为反向指标(云上=过热回调, 云下=超跌反弹)
    score = 100 - score
    return max(10, min(90, score))


def score_atr_regime(r) -> float:
    """ATR波动率状态评分 — 低波蓄力看多，高波衰竭看空。

    ATR%低+趋势向上 → 60~80 (蓄力突破)
    ATR%高+趋势向下 → 20~40 (恐慌抛售)
    ATR%适中 → 45~55 (正常波动)
    """
    atr_pct = r.get("atr_pct", 0)
    slope = r.get("trend_slope_pct", 0)
    mom = r.get("momentum_20", 0)
    vr = r.get("volume_ratio", 1.0)

    # ATR%分位: A股日均ATR%约1.5-3%
    if atr_pct < 1.2:
        vol_score = 10      # 极低波动 → 蓄力待突破
    elif atr_pct < 2.0:
        vol_score = 5       # 低波动 → 偏稳
    elif atr_pct < 3.5:
        vol_score = 0       # 正常波动
    elif atr_pct < 5.0:
        vol_score = -5      # 高波动 → 不确定性大
    else:
        vol_score = -12     # 极高波动 → 恐慌/狂热

    # 波动率+趋势交互: 低波+上行=蓄力, 高波+下行=崩溃
    trend_dir = 1 if (slope > 0.02 and mom > 0) else (-1 if (slope < -0.02 and mom < 0) else 0)
    interaction = vol_score * trend_dir * 0.3

    # 量比确认: 缩量低波=真蓄力, 放量高波=真恐慌
    vol_confirm = 0
    if atr_pct < 2.0 and vr < 0.8:
        vol_confirm = 5   # 缩量低波蓄力
    elif atr_pct > 3.5 and vr > 1.5:
        vol_confirm = -5  # 放量高波恐慌

    score = 50 + vol_score + interaction + vol_confirm + 0.15 * mom
    return max(10, min(90, score))


def score_quant_alpha(r) -> float:
    """量化Alpha评分 — 融合6个核心因子的趋势质量综合评分。

    总分 = 30%相对强度 + 20%趋势斜率 + 15%波动收敛 + 15%近高位结构 + 10%流动性 + 10%量能确认
    评分范围 0-100，50=中性。
    """
    # ── 1. 相对强度 (30%) ── 用 ret20 做个股绝对动量代理
    # (RPS需跨股票对比，暂用绝对动量; 线上版可接入板块/指数对比)
    ret20 = r.get("momentum_20", 0)
    ret10 = r.get("momentum_10", 0)
    ret5 = r.get("momentum_5", 0)
    # "中高强度但不过热": ret20在5-20%最优, >30%过热, <0弱
    if ret20 > 30:
        rps_score = 55     # 过热，不是最优
    elif ret20 > 20:
        rps_score = 70     # 强但略过热
    elif ret20 > 10:
        rps_score = 85     # 最优区间
    elif ret20 > 5:
        rps_score = 75     # 中强
    elif ret20 > 0:
        rps_score = 60     # 弱多
    elif ret20 > -5:
        rps_score = 45     # 弱
    elif ret20 > -15:
        rps_score = 30     # 下跌
    else:
        rps_score = 15     # 深度下跌
    # 短期收益结构加成: 稳态上升(ret5>0 & ret10>0 & ret20>0)
    if ret5 > 0 and ret10 > 0 and ret20 > 0:
        rps_score += 5     # 三级联涨
    elif ret5 < 0 and ret10 > 0 and ret20 > 0:
        rps_score += 2     # 强趋势回踩

    # ── 2. 趋势斜率 (20%) ── MA20斜率
    slope = r.get("ma20_slope", 0)
    # slope > 0.1 强上行, 0.02~0.1 平稳抬升, <0 下行
    if slope > 0.15:
        slope_score = 85
    elif slope > 0.08:
        slope_score = 75
    elif slope > 0.03:
        slope_score = 65   # 平稳抬升最理想
    elif slope > 0:
        slope_score = 55
    elif slope > -0.03:
        slope_score = 45
    elif slope > -0.1:
        slope_score = 30
    else:
        slope_score = 15

    # ── 3. 波动收敛 (15%) ── ATR5/ATR20
    atr_conv = r.get("atr_converge", 1.0)
    # <0.8 强收敛(蓄势), 0.8-1.0 正常, >1.2 扩张(不稳定)
    if atr_conv < 0.6:
        vol_score = 85     # 极度收敛
    elif atr_conv < 0.8:
        vol_score = 75     # 蓄势完成
    elif atr_conv < 1.0:
        vol_score = 60     # 正常偏收敛
    elif atr_conv < 1.2:
        vol_score = 45     # 正常偏扩张
    elif atr_conv < 1.5:
        vol_score = 30     # 波动扩大
    else:
        vol_score = 15     # 剧烈波动

    # ── 4. 近高位结构 (15%) ── 当前价/60日高点
    near_h = r.get("near_high60", 1.0)
    # 0.95-1.0 近高位(结构强), 0.85-0.95 中位, <0.85 远离(弱)
    if near_h > 0.98:
        high_score = 80    # 接近新高
    elif near_h > 0.93:
        high_score = 75    # 强结构
    elif near_h > 0.85:
        high_score = 60    # 中位
    elif near_h > 0.75:
        high_score = 40    # 偏弱
    else:
        high_score = 20    # 深度调整

    # ── 5. 流动性 (10%) ── 20日均额
    avg_amt = r.get("avg_amount20", 0)
    # A股: >5亿优秀, 1-5亿正常, <5000万流动性差
    if avg_amt > 5e8:
        liq_score = 80
    elif avg_amt > 2e8:
        liq_score = 70
    elif avg_amt > 5e7:
        liq_score = 55
    elif avg_amt > 1e7:
        liq_score = 40
    else:
        liq_score = 25

    # ── 6. 量能确认 (10%) ── 当日量能比
    amt_ratio = r.get("amount_ratio", 1.0)
    # 1.0-1.5 温和放量(健康), >2.0 爆量(可能过热), <0.6 缩量
    if 1.0 <= amt_ratio < 1.5:
        amt_score = 75     # 温和放量最佳
    elif 1.5 <= amt_ratio < 2.0:
        amt_score = 65     # 明显放量
    elif amt_ratio >= 2.0:
        amt_score = 45     # 爆量,可能见顶
    elif amt_ratio > 0.7:
        amt_score = 55     # 正常
    elif amt_ratio > 0.5:
        amt_score = 40     # 缩量
    else:
        amt_score = 25     # 极度缩量

    # ── 均线多头结构加成 ──
    ma_vals = r.get("ma_vals", {})
    periods = sorted(ma_vals.keys())
    ordered = sum(1 for i in range(len(periods)-1) if ma_vals[periods[i]] > ma_vals[periods[i+1]])
    n_pairs = max(1, len(periods)-1)
    ma_bonus = 0
    if ordered == n_pairs:
        ma_bonus = 5       # 完美多头排列
    elif ordered >= n_pairs - 1:
        ma_bonus = 2       # 近似多头

    # ── 加权合成 ──
    total = (rps_score * 0.30 + slope_score * 0.20 + vol_score * 0.15 +
             high_score * 0.15 + liq_score * 0.10 + amt_score * 0.10 + ma_bonus)
    # 反转: A股低分段(超跌+弱势)后续收益远超高分段，均值回归效应
    total = 100 - total
    return max(10, min(90, total))


# ── 单因子评分 (用于逐因子IC评估) ──────────────────────────────

def _f_rps(r) -> float:
    """相对强度: ret20 映射到分数"""
    ret20 = r.get("momentum_20", 0)
    ret5 = r.get("momentum_5", 0)
    ret10 = r.get("momentum_10", 0)
    if ret20 > 30: s = 55
    elif ret20 > 20: s = 70
    elif ret20 > 10: s = 85
    elif ret20 > 5: s = 75
    elif ret20 > 0: s = 60
    elif ret20 > -5: s = 45
    elif ret20 > -15: s = 30
    else: s = 15
    if ret5 > 0 and ret10 > 0 and ret20 > 0: s += 5
    elif ret5 < 0 and ret10 > 0 and ret20 > 0: s += 2
    return s

def _f_slope(r) -> float:
    """MA20斜率"""
    slope = r.get("ma20_slope", 0)
    if slope > 0.15: return 85
    elif slope > 0.08: return 75
    elif slope > 0.03: return 65
    elif slope > 0: return 55
    elif slope > -0.03: return 45
    elif slope > -0.1: return 30
    else: return 15

def _f_atr_conv(r) -> float:
    """ATR5/ATR20 波动收敛"""
    c = r.get("atr_converge", 1.0)
    if c < 0.6: return 85
    elif c < 0.8: return 75
    elif c < 1.0: return 60
    elif c < 1.2: return 45
    elif c < 1.5: return 30
    else: return 15

def _f_near_high(r) -> float:
    """近60日高点距离"""
    h = r.get("near_high60", 1.0)
    if h > 0.98: return 80
    elif h > 0.93: return 75
    elif h > 0.85: return 60
    elif h > 0.75: return 40
    else: return 20

def _f_liquidity(r) -> float:
    """20日平均成交额"""
    a = r.get("avg_amount20", 0)
    if a > 5e8: return 80
    elif a > 2e8: return 70
    elif a > 5e7: return 55
    elif a > 1e7: return 40
    else: return 25

def _f_amt_ratio(r) -> float:
    """量能比"""
    ar = r.get("amount_ratio", 1.0)
    if 1.0 <= ar < 1.5: return 85    # was 75 温和放量最佳
    elif 1.5 <= ar < 2.0: return 72  # was 65
    elif ar >= 2.0: return 35        # was 45 爆量过热
    elif ar > 0.7: return 50         # was 55
    elif ar > 0.5: return 30         # was 40
    else: return 15                  # was 25 极度缩量


# ── Agent注册 ──────────────────────────────────────────────────

AGENTS = {
    "trend_momentum": ("趋势动量(反转)", score_trend_momentum),
    "capital_liquidity": ("资金流动性", score_capital_liquidity),
    "divergence": ("背离检测", score_divergence),
    "chanlun": ("缠论", score_chanlun),
    "pattern": ("K线形态", score_pattern),
    "sentiment_flow": ("情绪舆情(降级)", score_sentiment_flow),
    "volume_structure": ("量价结构", score_volume_structure),
    "resonance": ("多周期共振(日线)", score_resonance),
    "kline_vision": ("K线视觉(降级)", score_kline_vision_fallback),
    "ichimoku": ("一目均衡图", score_ichimoku),
    "atr_regime": ("ATR波动状态", score_atr_regime),
    "quant_alpha": ("量化Alpha(合)", score_quant_alpha),
    # ── 单因子拆分评估 ──
    "f_rps20": ("因子:相对强度", lambda r: max(10,min(90, _f_rps(r)))),
    "f_slope": ("因子:MA20斜率", lambda r: max(10,min(90, _f_slope(r)))),
    "f_atr_conv": ("因子:ATR收敛", lambda r: max(10,min(90, _f_atr_conv(r)))),
    "f_near_high": ("因子:近高位", lambda r: max(10,min(90, _f_near_high(r)))),
    "f_liquidity": ("因子:流动性", lambda r: max(10,min(90, _f_liquidity(r)))),
    "f_amt_ratio": ("因子:量能比", lambda r: max(10,min(90, _f_amt_ratio(r)))),
}


# ── 回测主逻辑 ──────────────────────────────────────────────────

def compute_all_scores(symbols: list[str]) -> dict[str, list]:
    """一次性计算所有股票的指标，然后对所有Agent评分。返回 {agent_key: [(score, r5, r10, r20), ...]}"""
    agent_data = {k: [] for k in AGENTS}
    total = len(symbols)

    for si, sym in enumerate(symbols):
        if (si + 1) % 20 == 0:
            print(f"  进度: {si+1}/{total}...", flush=True)
        try:
            df = load_tdx_daily(sym)
            if df is None or len(df) < 160:
                continue
            ind = rolling_indicators(df, min_bars=120)
            if ind.empty:
                continue

            close_arr = df["close"].values
            rows = [r.to_dict() for _, r in ind.iterrows()]

            for row in rows:
                idx = int(row["idx"])
                r5 = (close_arr[idx + 5] / close_arr[idx] - 1) * 100 if idx + 5 < len(close_arr) else np.nan
                r10 = (close_arr[idx + 10] / close_arr[idx] - 1) * 100 if idx + 10 < len(close_arr) else np.nan
                r20 = (close_arr[idx + 20] / close_arr[idx] - 1) * 100 if idx + 20 < len(close_arr) else np.nan

                for agent_key, (label, score_fn) in AGENTS.items():
                    score = score_fn(row)
                    agent_data[agent_key].append((score, r5, r10, r20))
        except Exception as e:
            print(f"  {sym} 失败: {e}", flush=True)

    return agent_data


def analyze_agent(pairs: list) -> dict:
    """分析单个Agent的回测结果。"""
    if not pairs:
        return {}

    scores = np.array([p[0] for p in pairs])
    r5 = np.array([p[1] for p in pairs])
    r10 = np.array([p[2] for p in pairs])
    r20 = np.array([p[3] for p in pairs])

    mask5 = ~np.isnan(r5)
    mask10 = ~np.isnan(r10)
    mask20 = ~np.isnan(r20)
    ic5 = np.corrcoef(scores[mask5], r5[mask5])[0, 1] if mask5.sum() > 30 else 0
    ic10 = np.corrcoef(scores[mask10], r10[mask10])[0, 1] if mask10.sum() > 30 else 0
    ic20 = np.corrcoef(scores[mask20], r20[mask20])[0, 1] if mask20.sum() > 30 else 0

    bins = [(0, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 75), (75, 85), (85, 100)]
    buckets = []
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        sub_r5 = r5[mask & mask5]
        sub_r10 = r10[mask & mask10]
        sub_r20 = r20[mask & mask20]
        buckets.append({
            "range": f"{lo}-{hi}",
            "count": int(mask.sum()),
            "avg_5d": float(np.mean(sub_r5)) if len(sub_r5) > 0 else None,
            "wr_5d": float((sub_r5 > 0).mean() * 100) if len(sub_r5) > 0 else None,
            "avg_10d": float(np.mean(sub_r10)) if len(sub_r10) > 0 else None,
            "wr_10d": float((sub_r10 > 0).mean() * 100) if len(sub_r10) > 0 else None,
            "avg_20d": float(np.mean(sub_r20)) if len(sub_r20) > 0 else None,
            "wr_20d": float((sub_r20 > 0).mean() * 100) if len(sub_r20) > 0 else None,
        })

    return {
        "total": len(pairs),
        "ic5": ic5, "ic10": ic10, "ic20": ic20,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "buckets": buckets,
    }


def print_agent_report(agent_key: str, label: str, result: dict):
    """打印单个Agent的回测结果。"""
    if not result:
        print(f"\n{'=' * 80}")
        print(f"{agent_key} ({label}) — 无数据")
        return

    print(f"\n{'=' * 80}")
    print(f"{agent_key} ({label})  样本={result['total']}  均值={result['mean']:.1f}  σ={result['std']:.1f}")
    print(f"IC(5d)={result['ic5']:+.4f}  IC(10d)={result['ic10']:+.4f}  IC(20d)={result['ic20']:+.4f}")
    print(f"{'=' * 80}")
    print(f"{'分数段':>10} | {'样本':>6} | {'5日均涨':>8} | {'5日胜率':>7} | {'10日均涨':>9} | {'10日胜率':>8} | {'20日均涨':>9} | {'20日胜率':>8}")
    print("-" * 90)
    for b in result["buckets"]:
        def _f(v): return f"{v:>+8.2f}%" if v is not None else "     N/A"
        def _w(v): return f"{v:>7.1f}%" if v is not None else "     N/A"
        print(f"{b['range']:>10} | {b['count']:>6} | {_f(b['avg_5d'])} | {_w(b['wr_5d'])} | {_f(b['avg_10d'])} | {_w(b['wr_10d'])} | {_f(b['avg_20d'])} | {_w(b['wr_20d'])}")


def main():
    # 获取所有可用股票
    history_dir = Path("output/history")
    if history_dir.exists():
        symbols = sorted([d.name for d in history_dir.iterdir() if d.is_dir() and len(d.name) == 6])
    else:
        symbols = []

    if not symbols:
        print("未找到股票数据，请确认 output/history/ 目录存在")
        return

    print(f"回测股票数: {len(symbols)}", flush=True)
    print(f"回测Agent数: {len(AGENTS)}", flush=True)
    print(flush=True)

    print("计算指标（一次性）...", flush=True)
    agent_data = compute_all_scores(symbols)

    all_results = {}
    for agent_key, (label, _) in AGENTS.items():
        result = analyze_agent(agent_data[agent_key])
        all_results[agent_key] = (label, result)
        if result:
            print(f"[{agent_key}] {result['total']}样本, IC(5d)={result['ic5']:+.4f}", flush=True)

    # 打印所有报告
    for agent_key, (label, result) in all_results.items():
        print_agent_report(agent_key, label, result)

    # IC汇总
    print(f"\n{'=' * 80}")
    print("IC汇总排名")
    print(f"{'=' * 80}")
    print(f"{'Agent':>22} {'中文':>14} | {'IC(5d)':>8} {'IC(10d)':>8} {'IC(20d)':>8} | {'均值':>6} {'σ':>6} {'样本':>8}")
    print("-" * 95)
    ranked = sorted(all_results.items(), key=lambda x: x[1][1].get("ic5", 0) if x[1][1] else -999, reverse=True)
    for agent_key, (label, result) in ranked:
        if not result: continue
        print(f"{agent_key:>22} {label:>14} | {result['ic5']:>+8.4f} {result['ic10']:>+8.4f} {result['ic20']:>+8.4f} | {result['mean']:>6.1f} {result['std']:>6.1f} {result['total']:>8}")


if __name__ == "__main__":
    main()
