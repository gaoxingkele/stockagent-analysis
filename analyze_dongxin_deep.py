"""688110.SH 东芯股份 — 深度分析: 缠论 + 波浪 + 风险预案."""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / ".env.cloubic", override=False)

import numpy as np
import pandas as pd
import talib
import tushare as ts

from stockagent_analysis.data_backend import DataBackend
from stockagent_analysis.tushare_enrich import enrich_with_tushare, compute_quant_score
from stockagent_analysis.sparse_layered_score import (
    extract_features_from_enrich, derive_context_from_enrich,
    derive_mf_state, compute_sparse_layered_score,
)

TS_CODE = "688110.SH"
NAME = "东芯股份"

pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

# 1) 拉 120 天日线
print(f"\n{'='*80}\n# {TS_CODE} {NAME} 深度分析\n{'='*80}")
end = datetime.now().strftime("%Y%m%d")
start = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
df = pro.daily(ts_code=TS_CODE, start_date=start, end_date=end)
df = df.sort_values("trade_date").reset_index(drop=True)
df["dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df.set_index("dt", inplace=True)
n = len(df)
print(f"\n日线数据: {n} 个交易日, 区间 {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"最新: 收盘 {df.iloc[-1]['close']:.2f} ({df.iloc[-1]['pct_chg']:+.2f}%)")

# 2) 关键价格点
high_60 = df["high"].iloc[-60:].max()
low_60 = df["low"].iloc[-60:].min()
high_120 = df["high"].iloc[-120:].max()
low_120 = df["low"].iloc[-120:].min()
high_all = df["high"].max()
low_all = df["low"].min()
curr = df.iloc[-1]["close"]
print(f"\n## 关键价格点")
print(f"  历史区间 ({n}日): {low_all:.2f} ~ {high_all:.2f}, 当前距高 {(curr/high_all-1)*100:+.1f}%")
print(f"  60 日区间: {low_60:.2f} ~ {high_60:.2f}")
print(f"  当前 {curr:.2f}, 距 60 日高 {(curr/high_60-1)*100:+.1f}%, 距 60 日低 {(curr/low_60-1)*100:+.1f}%")

# 3) 波浪结构 (用 ZigZag 5% 阈值找转折点)
def zigzag(prices, threshold=0.05):
    pivots = [(0, prices[0])]
    last_pivot = (0, prices[0], "init")
    direction = 0
    for i in range(1, len(prices)):
        p = prices[i]
        change = (p - last_pivot[1]) / last_pivot[1]
        if direction == 0:
            if abs(change) >= threshold:
                direction = 1 if change > 0 else -1
                pivots.append((i, p))
                last_pivot = (i, p, "up" if direction > 0 else "down")
        elif direction > 0:
            if p > last_pivot[1]:
                pivots[-1] = (i, p)
                last_pivot = (i, p, "up")
            elif (p / last_pivot[1] - 1) <= -threshold:
                pivots.append((i, p))
                last_pivot = (i, p, "down")
                direction = -1
        else:
            if p < last_pivot[1]:
                pivots[-1] = (i, p)
                last_pivot = (i, p, "down")
            elif (p / last_pivot[1] - 1) >= threshold:
                pivots.append((i, p))
                last_pivot = (i, p, "up")
                direction = 1
    return pivots

pivots = zigzag(df["close"].values, threshold=0.07)
print(f"\n## 波浪结构 (ZigZag 7% 阈值, {len(pivots)} 个转折点)")
for idx, price in pivots[-10:]:
    date = df.index[idx].strftime("%m-%d")
    print(f"  {date} {price:.2f}")

# 推断波浪阶段
if len(pivots) >= 3:
    p0, p1, p2 = pivots[-3:]
    if p1[1] > p0[1] and p2[1] > p1[1]:
        wave_state = "持续上涨 (推动浪 3-5)"
    elif p1[1] > p0[1] and p2[1] < p1[1]:
        wave_state = "上涨后回调 (调整浪 ABC 进行中)"
    elif p1[1] < p0[1] and p2[1] > p1[1]:
        wave_state = "下跌后反弹 (B 浪 / 1 浪起步)"
    else:
        wave_state = "持续下跌"
    last_idx = pivots[-1][0]
    bars_since = n - 1 - last_idx
    print(f"  >> 当前阶段: {wave_state}, 距最近转折 {bars_since} 个交易日")

# 4) 缠论分析 (用 DataBackend 静态方法)
print(f"\n## 缠论结构")
chan = DataBackend._detect_chanlun_signals(df["close"], df["high"], df["low"], n)
print(f"  分型: {len(chan['fractals'])} 个")
print(f"  笔: {len(chan['bi_list'])} 笔")
if chan['bi_list']:
    last_bi = chan['bi_list'][-1]
    bi_dir = "上涨" if last_bi['dir'] == 'up' else "下跌"
    print(f"  最近笔: {bi_dir} {last_bi['start_price']:.2f} → {last_bi['end_price']:.2f}")
print(f"  中枢: {len(chan['zhongshu'])} 个")
for zs in chan['zhongshu'][-3:]:
    print(f"    [{zs['low']:.2f} ~ {zs['high']:.2f}] ({zs['bi_count']} 笔)")
if chan['zhongshu']:
    last_zs = chan['zhongshu'][-1]
    in_zs = last_zs['low'] <= curr <= last_zs['high']
    pos_zs = "中枢内" if in_zs else (f"中枢上方 +{(curr/last_zs['high']-1)*100:.1f}%" if curr > last_zs['high'] else f"中枢下方 -{(1-curr/last_zs['low'])*100:.1f}%")
    print(f"  当前价位: {pos_zs}")
print(f"  缠论分: {chan['chanlun_score']:+d}")
print(f"  描述: {chan['desc']}")
if chan['buy_signals']:
    print(f"  买点信号:")
    for s in chan['buy_signals']:
        print(f"    [{s['type']}] {s['desc']}")
if chan['sell_signals']:
    print(f"  卖点信号:")
    for s in chan['sell_signals']:
        print(f"    [{s['type']}] {s['desc']}")

# 5) 技术指标
print(f"\n## 关键技术指标")
close = df["close"].values
high = df["high"].values
low = df["low"].values
volume = df["vol"].values

macd, macd_sig, macd_hist = talib.MACD(close, 12, 26, 9)
rsi6 = talib.RSI(close, 6)
rsi14 = talib.RSI(close, 14)
adx = talib.ADX(high, low, close, 14)
pdi = talib.PLUS_DI(high, low, close, 14)
mdi = talib.MINUS_DI(high, low, close, 14)
atr = talib.ATR(high, low, close, 14)
upper, mid, lower = talib.BBANDS(close, 20, 2, 2)
ma5 = talib.SMA(close, 5)
ma10 = talib.SMA(close, 10)
ma20 = talib.SMA(close, 20)
ma60 = talib.SMA(close, 60)

print(f"  收盘 {curr:.2f}")
print(f"  MA5={ma5[-1]:.2f}  MA10={ma10[-1]:.2f}  MA20={ma20[-1]:.2f}  MA60={ma60[-1]:.2f}")
print(f"  RSI6={rsi6[-1]:.1f}  RSI14={rsi14[-1]:.1f}")
print(f"  MACD: DIF={macd[-1]:.3f} DEA={macd_sig[-1]:.3f} HIST={macd_hist[-1]:+.3f}")
print(f"  ADX={adx[-1]:.1f} (+DI={pdi[-1]:.1f}, -DI={mdi[-1]:.1f})  趋势" + ("强" if adx[-1] > 25 else "弱"))
print(f"  ATR={atr[-1]:.2f}  ATR/close={atr[-1]/curr*100:.1f}%")
print(f"  布林: 上轨 {upper[-1]:.2f}  中轨 {mid[-1]:.2f}  下轨 {lower[-1]:.2f}")
print(f"  布林位置: {(curr-lower[-1])/(upper[-1]-lower[-1])*100:.0f}% (0=下轨, 100=上轨)")

# 趋势判定
trend = []
if curr > ma20[-1] > ma60[-1]:
    trend.append("MA 多头排列")
elif curr < ma20[-1] < ma60[-1]:
    trend.append("MA 空头排列")
if rsi14[-1] > 70:
    trend.append("RSI 超买")
elif rsi14[-1] < 30:
    trend.append("RSI 超卖")
if macd_hist[-1] > 0 and macd_hist[-1] > macd_hist[-2]:
    trend.append("MACD 红柱放大")
elif macd_hist[-1] < 0 and macd_hist[-1] < macd_hist[-2]:
    trend.append("MACD 绿柱放大")
print(f"  综合: {', '.join(trend) if trend else '中性'}")

# 6) sparse_layered + quant_score
print(f"\n## sparse_layered + quant_score (复用前面分析)")
enrich = enrich_with_tushare(TS_CODE)
if enrich:
    features = extract_features_from_enrich(enrich)
    context = derive_context_from_enrich(enrich, industry=enrich.get("industry"))
    mf_state = derive_mf_state(enrich)
    sl = compute_sparse_layered_score(
        features=features, context=context,
        regime={"trend": "slow_bull", "dispersion": "high_industry"},
        mf_state=mf_state,
    )
    qs = compute_quant_score(enrich)
    print(f"  sparse_layered: {sl['layered_score']:.1f} (active={sl['n_active']}, K={sl['conflict_K']:.2f}, conf={sl['confidence']})")
    print(f"  quant_score:    {qs['quant_score']:.1f} (Δ={qs['total_delta']:+.1f})")
    composite = 0.6 * sl['layered_score'] + 0.4 * qs['quant_score']
    print(f"  综合 (60% sparse + 40% quant): {composite:.1f}")

# 7) 风险预案 — 关键价位
print(f"\n## 关键支撑/阻力位")
last_5_low = df["low"].iloc[-5:].min()
last_10_low = df["low"].iloc[-10:].min()
last_20_low = df["low"].iloc[-20:].min()
support_strong = max(ma60[-1], last_20_low)   # 强支撑取较高者 (近) 为弱支撑
support_med = ma20[-1]
support_weak = ma10[-1]
resistance_recent = df["high"].iloc[-5:].max()
resistance_60 = high_60

print(f"  阻力位 (近 5 日高): {resistance_recent:.2f} (当前距 {(resistance_recent/curr-1)*100:+.1f}%)")
print(f"  阻力位 (60 日高):  {resistance_60:.2f} (距 {(resistance_60/curr-1)*100:+.1f}%)")
print(f"  支撑位 (MA10):     {ma10[-1]:.2f} ({(ma10[-1]/curr-1)*100:+.1f}%)")
print(f"  支撑位 (MA20):     {ma20[-1]:.2f} ({(ma20[-1]/curr-1)*100:+.1f}%)")
print(f"  支撑位 (MA60):     {ma60[-1]:.2f} ({(ma60[-1]/curr-1)*100:+.1f}%)")
print(f"  支撑位 (布林下轨): {lower[-1]:.2f} ({(lower[-1]/curr-1)*100:+.1f}%)")
print(f"  支撑位 (20 日低):  {last_20_low:.2f} ({(last_20_low/curr-1)*100:+.1f}%)")

if chan['zhongshu']:
    last_zs = chan['zhongshu'][-1]
    print(f"  缠论中枢上沿: {last_zs['high']:.2f} ({(last_zs['high']/curr-1)*100:+.1f}%)")
    print(f"  缠论中枢下沿: {last_zs['low']:.2f} ({(last_zs['low']/curr-1)*100:+.1f}%)")

# 8) 风险预案
print(f"\n## 风险预案 (基于 ATR 与关键位)")
atr_now = atr[-1]
print(f"  当日 ATR: {atr_now:.2f} (≈ {atr_now/curr*100:.1f}% 日波幅)")

# 假设当前进场位置
# - 1 ATR 止损 (激进) / 2 ATR 止损 (常规) / 3 ATR 止损 (保守)
print(f"\n  止损方案:")
print(f"    激进 1×ATR:  {curr - atr_now:.2f} ({-atr_now/curr*100:.1f}%)")
print(f"    常规 2×ATR:  {curr - 2*atr_now:.2f} ({-2*atr_now/curr*100:.1f}%)")
print(f"    保守 3×ATR:  {curr - 3*atr_now:.2f} ({-3*atr_now/curr*100:.1f}%)")
print(f"\n  目标位 (按 风险:收益 1:2):")
print(f"    激进位: {curr + 2*atr_now:.2f} (+{2*atr_now/curr*100:.1f}%)")
print(f"    常规位: {curr + 4*atr_now:.2f} (+{4*atr_now/curr*100:.1f}%)")
