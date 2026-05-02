"""688110.SH 东芯股份 周线深度分析."""
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

TS_CODE = "688110.SH"
NAME = "东芯股份"

pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

# 1) 拉日线 (3 年, 用于 resample 周线)
print(f"\n{'='*80}\n# {TS_CODE} {NAME} 周线深度分析\n{'='*80}")
end = datetime.now().strftime("%Y%m%d")
start = (datetime.now() - timedelta(days=1100)).strftime("%Y%m%d")

# 用 weekly 接口直接拉
wdf = pro.weekly(ts_code=TS_CODE, start_date=start, end_date=end)
wdf = wdf.sort_values("trade_date").reset_index(drop=True)
wdf["dt"] = pd.to_datetime(wdf["trade_date"], format="%Y%m%d")
wdf.set_index("dt", inplace=True)
n = len(wdf)
print(f"\n周线数据: {n} 周, 区间 {wdf.index[0].date()} ~ {wdf.index[-1].date()}")
last = wdf.iloc[-1]
print(f"最新一周 ({last.name.date()}): 开 {last['open']:.2f} 高 {last['high']:.2f} "
      f"低 {last['low']:.2f} 收 {last['close']:.2f} 涨 {last['pct_chg']:+.2f}%")

# 2) 关键价格区间
high_all = wdf["high"].max()
low_all = wdf["low"].min()
high_52 = wdf["high"].iloc[-52:].max()
low_52 = wdf["low"].iloc[-52:].min()
high_26 = wdf["high"].iloc[-26:].max()
low_26 = wdf["low"].iloc[-26:].min()
high_13 = wdf["high"].iloc[-13:].max()
low_13 = wdf["low"].iloc[-13:].min()
curr = float(last["close"])

print(f"\n## 关键周线区间")
print(f"  历史 ({n}周): {low_all:.2f} ~ {high_all:.2f}, 当前距高 {(curr/high_all-1)*100:+.1f}%")
print(f"  近 52 周: {low_52:.2f} ~ {high_52:.2f}")
print(f"  近 26 周: {low_26:.2f} ~ {high_26:.2f}, 距 26W 高 {(curr/high_26-1)*100:+.1f}%, 距 26W 低 {(curr/low_26-1)*100:+.1f}%")
print(f"  近 13 周: {low_13:.2f} ~ {high_13:.2f}")

# 3) 周线波浪 (ZigZag 10% 阈值更适合周线)
def zigzag(prices, threshold=0.10):
    pivots = [(0, prices[0])]
    last_pivot = (0, prices[0])
    direction = 0
    for i in range(1, len(prices)):
        p = prices[i]
        change = (p - last_pivot[1]) / last_pivot[1]
        if direction == 0:
            if abs(change) >= threshold:
                direction = 1 if change > 0 else -1
                pivots.append((i, p))
                last_pivot = (i, p)
        elif direction > 0:
            if p > last_pivot[1]:
                pivots[-1] = (i, p)
                last_pivot = (i, p)
            elif (p / last_pivot[1] - 1) <= -threshold:
                pivots.append((i, p))
                last_pivot = (i, p)
                direction = -1
        else:
            if p < last_pivot[1]:
                pivots[-1] = (i, p)
                last_pivot = (i, p)
            elif (p / last_pivot[1] - 1) >= threshold:
                pivots.append((i, p))
                last_pivot = (i, p)
                direction = 1
    return pivots

pivots = zigzag(wdf["close"].values, threshold=0.12)
print(f"\n## 周线波浪结构 (ZigZag 12% 阈值, {len(pivots)} 个转折)")
for idx, price in pivots[-12:]:
    date = wdf.index[idx].strftime("%Y-%m-%d")
    print(f"  {date}  {price:.2f}")

# 推断周线波浪
if len(pivots) >= 5:
    last_5 = pivots[-5:]
    p_curr = last_5[-1]
    bars_since = n - 1 - p_curr[0]

    # 整体趋势
    if last_5[-1][1] > last_5[0][1] * 1.2:
        big_trend = "**1 年级别上升趋势** (波段高点逐步抬高)"
    elif last_5[-1][1] < last_5[0][1] * 0.8:
        big_trend = "**1 年级别下降趋势**"
    else:
        big_trend = "1 年级别震荡区间"
    print(f"  >> 1 年级别: {big_trend}")
    print(f"  >> 距最近转折 {bars_since} 周")

# 4) 周线缠论
print(f"\n## 周线缠论结构")
chan = DataBackend._detect_chanlun_signals(wdf["close"], wdf["high"], wdf["low"], n)
print(f"  分型: {len(chan['fractals'])} 个")
print(f"  笔: {len(chan['bi_list'])} 笔")
if chan['bi_list']:
    last_bi = chan['bi_list'][-1]
    bi_dir = "上涨" if last_bi['dir'] == 'up' else "下跌"
    print(f"  最近笔: {bi_dir} {last_bi['start_price']:.2f} -> {last_bi['end_price']:.2f}")
print(f"  中枢: {len(chan['zhongshu'])} 个")
for zs in chan['zhongshu'][-3:]:
    print(f"    [{zs['low']:.2f} ~ {zs['high']:.2f}] ({zs['bi_count']} 笔)")
if chan['zhongshu']:
    last_zs = chan['zhongshu'][-1]
    in_zs = last_zs['low'] <= curr <= last_zs['high']
    if in_zs:
        pos = "中枢内"
    elif curr > last_zs['high']:
        pos = f"中枢上方 +{(curr/last_zs['high']-1)*100:.1f}%"
    else:
        pos = f"中枢下方 -{(1-curr/last_zs['low'])*100:.1f}%"
    print(f"  当前位置: {pos}")
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

# 5) 周线技术指标
print(f"\n## 周线技术指标")
close = wdf["close"].values
high = wdf["high"].values
low = wdf["low"].values
volume = wdf["vol"].values

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
print(f"  ADX={adx[-1]:.1f} (+DI={pdi[-1]:.1f}, -DI={mdi[-1]:.1f})")
print(f"  ATR={atr[-1]:.2f}  ATR/close={atr[-1]/curr*100:.1f}%")
print(f"  布林: 上 {upper[-1]:.2f}  中 {mid[-1]:.2f}  下 {lower[-1]:.2f}")
print(f"  布林位置: {(curr-lower[-1])/(upper[-1]-lower[-1])*100:.0f}% (0=下轨, 100=上轨)")

# 6) 周线 MACD 历史背离
print(f"\n## 周线 MACD 趋势 (近 8 周)")
print(f"  {'日期':<12} {'收盘':<10} {'DIF':<8} {'DEA':<8} {'HIST':<8} 信号")
recent_8 = wdf.tail(8).copy()
for i, (idx, row) in enumerate(recent_8.iterrows()):
    pos = n - 8 + i
    dif = macd[pos]
    dea = macd_sig[pos]
    hist = macd_hist[pos]
    if hist > 0 and macd_hist[pos-1] < 0:
        sig = ">> 金叉"
    elif hist < 0 and macd_hist[pos-1] > 0:
        sig = ">> 死叉"
    else:
        sig = ""
    print(f"  {idx.strftime('%m-%d')}     {row['close']:<10.2f} {dif:>+6.2f}  {dea:>+6.2f}  {hist:>+6.2f}  {sig}")

# 7) 周线趋势综合判定
print(f"\n## 周线趋势综合判定")
trend_score = 0
verdict = []

# MA 排列
ma_arr = [ma5[-1], ma10[-1], ma20[-1], ma60[-1]]
if curr > ma_arr[0] > ma_arr[1] > ma_arr[2] > ma_arr[3]:
    verdict.append("MA 完美多头排列 (+15)")
    trend_score += 15
elif ma_arr[0] > ma_arr[1] > ma_arr[2]:
    verdict.append("MA5/10/20 多头 (+8)")
    trend_score += 8
elif curr < ma_arr[0] < ma_arr[1] < ma_arr[2]:
    verdict.append("MA 空头排列 (-12)")
    trend_score -= 12

# RSI
if rsi14[-1] > 70:
    verdict.append(f"RSI={rsi14[-1]:.0f} 周线超买 (-8)")
    trend_score -= 8
elif rsi14[-1] < 30:
    verdict.append(f"RSI={rsi14[-1]:.0f} 周线超卖 (+8)")
    trend_score += 8
elif 50 < rsi14[-1] < 65:
    verdict.append(f"RSI={rsi14[-1]:.0f} 偏强健康 (+5)")
    trend_score += 5

# MACD
if macd_hist[-1] > 0 and macd[-1] > 0:
    verdict.append(f"MACD 双正 多头主导 (+10)")
    trend_score += 10
elif macd_hist[-1] > 0 and macd[-1] < 0:
    verdict.append(f"MACD 0 轴下方金叉 转强中 (+5)")
    trend_score += 5
elif macd_hist[-1] < 0 and macd[-1] > 0:
    verdict.append(f"MACD 高位死叉 见顶风险 (-8)")
    trend_score -= 8

# ADX 趋势强度
if adx[-1] > 25:
    if pdi[-1] > mdi[-1]:
        verdict.append(f"ADX={adx[-1]:.0f} 趋势强 多方主导 (+8)")
        trend_score += 8
    else:
        verdict.append(f"ADX={adx[-1]:.0f} 趋势强 空方主导 (-8)")
        trend_score -= 8
elif adx[-1] < 20:
    verdict.append(f"ADX={adx[-1]:.0f} 趋势弱 震荡市 (0)")

# 缠论
if chan['chanlun_score'] >= 15:
    verdict.append(f"周线缠论 {chan['chanlun_score']:+d} 多 (+10)")
    trend_score += 10
elif chan['chanlun_score'] <= -15:
    verdict.append(f"周线缠论 {chan['chanlun_score']:+d} 空 (-10)")
    trend_score -= 10

# 与中枢关系
if chan['zhongshu']:
    last_zs = chan['zhongshu'][-1]
    if curr > last_zs['high'] * 1.05:
        verdict.append("突破周线中枢 +5%以上 (+5)")
        trend_score += 5
    elif curr < last_zs['low'] * 0.95:
        verdict.append("跌破周线中枢 -5% 以下 (-5)")
        trend_score -= 5

print(f"  详情:")
for v in verdict:
    print(f"    {v}")
print(f"\n  >> 周线综合分: {trend_score:+d}")
if trend_score >= 25:
    overall = ">> 强势上行"
elif trend_score >= 10:
    overall = ">> 偏多"
elif trend_score >= -10:
    overall = ">> 中性震荡"
elif trend_score >= -25:
    overall = ">> 偏空"
else:
    overall = ">> 弱势下行"
print(f"  >> 周线判断: {overall}")

# 8) 周线 vs 日线的对比
print(f"\n## 周线 vs 日线 对比 (多周期共振检查)")
# 拉日线
ddf = pro.daily(ts_code=TS_CODE, start_date=start, end_date=end)
ddf = ddf.sort_values("trade_date").reset_index(drop=True)
ddf["dt"] = pd.to_datetime(ddf["trade_date"], format="%Y%m%d")
ddf.set_index("dt", inplace=True)

dclose = ddf["close"].values
dhigh = ddf["high"].values
dlow = ddf["low"].values
d_macd, d_macd_sig, d_macd_hist = talib.MACD(dclose, 12, 26, 9)
d_rsi14 = talib.RSI(dclose, 14)
d_adx = talib.ADX(dhigh, dlow, dclose, 14)

print(f"\n  {'指标':<15} {'日线':<15} {'周线':<15} {'共振':<10}")
print(f"  {'-'*60}")
print(f"  {'RSI14':<15} {d_rsi14[-1]:<15.1f} {rsi14[-1]:<15.1f} "
      f"{'同向' if (d_rsi14[-1]>50)==(rsi14[-1]>50) else '背离'}")
print(f"  {'MACD HIST':<15} {d_macd_hist[-1]:<+15.3f} {macd_hist[-1]:<+15.3f} "
      f"{'同向' if (d_macd_hist[-1]>0)==(macd_hist[-1]>0) else '背离'}")
print(f"  {'ADX':<15} {d_adx[-1]:<15.1f} {adx[-1]:<15.1f} "
      f"{'强日强周' if d_adx[-1]>25 and adx[-1]>25 else ('强日弱周' if d_adx[-1]>25 else '弱日弱周')}")

# 9) 关键周线位置
print(f"\n## 关键周线位置")
print(f"  阻力位 (周线高点):")
print(f"    历史最高:   {high_all:.2f} ({(high_all/curr-1)*100:+.1f}%)")
print(f"    52 周高:    {high_52:.2f} ({(high_52/curr-1)*100:+.1f}%)")
print(f"    26 周高:    {high_26:.2f} ({(high_26/curr-1)*100:+.1f}%)")

print(f"  支撑位:")
print(f"    周 MA10:    {ma10[-1]:.2f} ({(ma10[-1]/curr-1)*100:+.1f}%)")
print(f"    周 MA20:    {ma20[-1]:.2f} ({(ma20[-1]/curr-1)*100:+.1f}%)")
print(f"    周 MA60:    {ma60[-1]:.2f} ({(ma60[-1]/curr-1)*100:+.1f}%)")
print(f"    布林下轨:   {lower[-1]:.2f} ({(lower[-1]/curr-1)*100:+.1f}%)")
print(f"    52 周低:    {low_52:.2f} ({(low_52/curr-1)*100:+.1f}%)")

if chan['zhongshu']:
    last_zs = chan['zhongshu'][-1]
    print(f"    周线中枢上沿: {last_zs['high']:.2f} ({(last_zs['high']/curr-1)*100:+.1f}%)")
    print(f"    周线中枢下沿: {last_zs['low']:.2f} ({(last_zs['low']/curr-1)*100:+.1f}%)")
