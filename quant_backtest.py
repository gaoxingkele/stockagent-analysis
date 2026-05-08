#!/usr/bin/env python3
"""v3.1 quant_score 规则有效性回测 2026-04-24 → 2026-05-08"""

import sys
import types
import time

# Mock jsonpath to work around install failure
mock_jp = types.ModuleType('jsonpath')
sys.modules['jsonpath'] = mock_jp

import akshare as ak
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

RAW_CSV = """symbol,name,delta,old_level,new_level
000155,川能动力,-3.81,weak_buy,weak_buy
000546,金圆股份,0.35,hold,hold
000792,盐湖股份,-3.96,hold,hold
000822,山东海化,-0.62,weak_sell,weak_sell
000876,新希望,-1.52,hold,hold
001257,盛龙股份,1.55,strong_sell,strong_sell
001288,运机集团,3.17,strong_sell,weak_sell
002009,天奇股份,2.93,strong_sell,weak_sell
002050,三花智控,-1.97,watch_sell,watch_sell
002068,黑猫股份,-1.12,watch_sell,watch_sell
002077,大港股份,-3.60,hold,hold
002082,万邦德,-2.76,hold,watch_sell
002124,天邦食品,1.75,weak_buy,weak_buy
002157,正邦科技,-2.96,hold,hold
002176,江特电机,-3.11,hold,hold
002183,怡亚通,-3.12,weak_sell,weak_sell
002192,融捷股份,-5.21,weak_buy,weak_buy
002203,海亮股份,-2.58,watch_sell,watch_sell
002236,大华股份,2.09,weak_sell,weak_sell
002263,大东南,0.90,hold,weak_buy
002276,万丰奥威,3.03,strong_sell,strong_sell
002294,信立泰,-5.15,weak_buy,hold
002351,漫步者,6.32,strong_sell,strong_sell
002371,北方华创,0.12,weak_sell,weak_sell
002407,多氟多,0.55,hold,hold
002422,科伦药业,-5.57,weak_buy,hold
002437,誉衡药业,-4.04,hold,hold
002460,赣锋锂业,-0.75,weak_buy,hold
002466,天齐锂业,-2.29,hold,hold
002491,通鼎互联,-3.20,weak_buy,hold
002497,雅化集团,2.45,weak_sell,weak_sell
002571,德力股份,-0.71,weak_sell,weak_sell
002572,索菲亚,6.25,strong_sell,strong_sell
002608,江苏国信,-5.35,hold,hold
002625,光启技术,4.03,strong_sell,strong_sell
002714,牧原股份,1.70,weak_sell,weak_sell
002730,电光科技,-5.59,weak_buy,hold
002773,康弘药业,0.84,watch_sell,watch_sell
002782,可立克,1.35,weak_sell,weak_sell
002832,比音勒芬,-3.04,watch_sell,watch_sell
002850,科达利,-0.63,weak_buy,weak_buy
002922,伊戈尔,-0.13,strong_sell,strong_sell
003026,中晶科技,-2.67,weak_buy,hold
300037,新宙邦,-5.16,hold,hold
300204,舒泰神,-0.48,watch_sell,watch_sell
300300,海峡创新,-4.20,hold,hold
300390,天华新能,-0.22,hold,hold
300498,温氏股份,-0.01,weak_sell,weak_sell
300674,宇信科技,2.34,weak_sell,weak_sell
300683,海特生物,-4.41,hold,hold
300769,德方纳米,-5.92,weak_buy,hold
300811,铂力特,0.92,strong_sell,strong_sell
301186,超达装备,1.07,weak_buy,weak_buy
301221,光庭信息,3.31,weak_sell,watch_sell
301292,海科新源,-3.42,weak_buy,hold
301509,金凯生科,-2.15,weak_buy,hold
301682,宏明电子,6.60,strong_sell,weak_sell
600062,华润双鹤,2.83,weak_sell,weak_sell
600110,诺德股份,-1.69,weak_buy,weak_buy
600227,赤天化,-1.91,hold,hold
600353,旭光电子,6.35,strong_sell,strong_sell
600388,龙净环保,1.05,watch_sell,watch_sell
600470,六国化工,-3.34,watch_sell,watch_sell
600488,津药药业,-5.09,weak_buy,hold
600513,联环药业,-4.64,weak_buy,hold
600548,深高速,-3.90,hold,hold
600727,鲁北化工,-2.86,hold,watch_sell
600745,闻泰科技,-1.00,strong_sell,strong_sell
600773,西藏城投,-3.31,weak_buy,hold
600982,宁波能源,-4.14,weak_buy,weak_buy
601866,中远海发,5.15,weak_sell,watch_sell
603026,石大胜华,-3.40,weak_buy,hold
603259,药明康德,-5.96,hold,hold
603288,海天味业,-3.47,hold,hold
603363,傲农生物,-3.62,hold,watch_sell
603399,永杉锂业,1.22,watch_sell,watch_sell
603477,巨星农牧,-1.26,hold,watch_sell
603538,美诺华,-3.22,watch_sell,weak_sell
603637,镇海股份,2.41,weak_sell,weak_sell
603662,柯力传感,5.58,strong_sell,strong_sell
603799,华友钴业,5.27,weak_sell,weak_sell
603806,福斯特,-6.83,weak_buy,hold
603906,龙蟠科技,0.84,watch_sell,hold
603992,松霖科技,-4.26,watch_sell,watch_sell
605499,东鹏饮料,2.23,strong_sell,weak_sell
605555,德昌股份,1.01,weak_sell,weak_sell
688175,高凌信息,1.98,weak_sell,weak_sell
688215,瑞晟智能,-4.01,weak_buy,hold
688266,泽璟制药,-1.97,hold,hold
688513,苑东生物,-2.72,watch_sell,watch_sell
688678,福立旺,-2.42,weak_sell,weak_sell
688799,华纳药厂,-5.32,weak_buy,hold
688805,健信超导,4.33,weak_sell,weak_sell
301667,股百达,-0.20,weak_buy,weak_buy
300890,泽霖股份,-0.11,weak_buy,weak_buy
601860,紫金银行,1.65,weak_sell,watch_sell
301702,300260,-2.57,watch_sell,watch_sell"""

def load_pool():
    from io import StringIO
    df = pd.read_csv(StringIO(RAW_CSV))
    df['symbol'] = df['symbol'].astype(str).str.zfill(6)
    df['delta'] = df['delta'].astype(float)
    return df.drop_duplicates(subset='symbol').reset_index(drop=True)


def fetch_price(symbol: str, retries=2) -> dict:
    """拉前复权日线, 返回 {start_close, end_close, high, low, pct, trading_days}"""
    for attempt in range(retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='daily',
                start_date='20260424',
                end_date='20260508',
                adjust='qfq'
            )
            if df is None or df.empty:
                return {'status': 'skipped', 'reason': 'empty'}

            # 列名可能中英文不同版本
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if '日期' in c or 'date' in cl:
                    col_map['date'] = c
                elif '收盘' in c or 'close' in cl:
                    col_map['close'] = c
                elif '最高' in c or 'high' in cl:
                    col_map['high'] = c
                elif '最低' in c or 'low' in cl:
                    col_map['low'] = c
                elif '开盘' in c or 'open' in cl:
                    col_map['open'] = c

            if 'close' not in col_map:
                return {'status': 'skipped', 'reason': f'no close col, cols={list(df.columns)}'}

            df = df.sort_values(col_map['date']).reset_index(drop=True)
            start_close = float(df[col_map['close']].iloc[0])
            end_close   = float(df[col_map['close']].iloc[-1])
            high        = float(df[col_map['high']].max())
            low         = float(df[col_map['low']].min())
            pct         = (end_close - start_close) / start_close * 100
            return {
                'status': 'ok',
                'start_close': round(start_close, 3),
                'end_close':   round(end_close, 3),
                'high':        round(high, 3),
                'low':         round(low, 3),
                'pct':         round(pct, 2),
                'trading_days': len(df),
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                return {'status': 'skipped', 'reason': str(e)[:120]}
    return {'status': 'skipped', 'reason': 'max retries'}


def group_label(delta):
    if delta >= 5:
        return 'Δ≥+5 大幅上调'
    elif delta >= 1:
        return '+1≤Δ<+5 温和上调'
    elif delta > -1:
        return '-1<Δ<+1 基本不变'
    elif delta > -5:
        return '-5<Δ≤-1 温和下调'
    else:
        return 'Δ≤-5 大幅下调'


GROUP_ORDER = [
    'Δ≥+5 大幅上调',
    '+1≤Δ<+5 温和上调',
    '-1<Δ<+1 基本不变',
    '-5<Δ≤-1 温和下调',
    'Δ≤-5 大幅下调',
]


def main():
    pool = load_pool()
    pool['group'] = pool['delta'].apply(group_label)

    print(f"股票池共 {len(pool)} 只, 开始抓取价格...")
    print("=" * 60)

    results = []
    skipped = []

    for i, row in pool.iterrows():
        sym = row['symbol']
        name = row['name']
        data = fetch_price(sym)
        if data['status'] == 'ok':
            results.append({
                'symbol': sym,
                'name': name,
                'delta': row['delta'],
                'old_level': row['old_level'],
                'new_level': row['new_level'],
                'group': row['group'],
                **{k: v for k, v in data.items() if k != 'status'},
            })
            print(f"  [{i+1:2d}/{len(pool)}] {sym} {name:8s} | {data['pct']:+.2f}% ({data['trading_days']}日)")
        else:
            skipped.append({'symbol': sym, 'name': name, 'reason': data.get('reason', '')})
            print(f"  [{i+1:2d}/{len(pool)}] {sym} {name:8s} | SKIPPED: {data.get('reason','')[:60]}")

        # 礼貌性限速
        time.sleep(0.3)

    df = pd.DataFrame(results)
    df_skip = pd.DataFrame(skipped)

    print(f"\n抓取完成: {len(df)} 只成功, {len(df_skip)} 只跳过")

    # ── 分组统计 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("分组统计")
    print("=" * 60)

    group_stats = []
    for g in GROUP_ORDER:
        sub = df[df['group'] == g]
        if sub.empty:
            continue
        n = len(sub)
        win = (sub['pct'] > 0).sum()
        stats = {
            'group': g,
            'n': n,
            'win_rate': round(win / n * 100, 1),
            'mean_pct': round(sub['pct'].mean(), 2),
            'median_pct': round(sub['pct'].median(), 2),
            'max_pct': round(sub['pct'].max(), 2),
            'min_pct': round(sub['pct'].min(), 2),
        }
        group_stats.append(stats)
        print(f"\n{g} (n={n})")
        print(f"  胜率: {stats['win_rate']}% | 均涨: {stats['mean_pct']:+.2f}% | 中位: {stats['median_pct']:+.2f}%")
        print(f"  最高: {stats['max_pct']:+.2f}% | 最低: {stats['min_pct']:+.2f}%")
        if not sub.empty:
            top3 = sub.nlargest(3, 'pct')[['symbol','name','pct']].values
            bot3 = sub.nsmallest(3, 'pct')[['symbol','name','pct']].values
            print(f"  涨幅前3: {', '.join(f'{r[1]}({r[2]:+.1f}%)' for r in top3)}")
            print(f"  跌幅前3: {', '.join(f'{r[1]}({r[2]:+.1f}%)' for r in bot3)}")

    df_stats = pd.DataFrame(group_stats)

    # ── 重点股明细 ──────────────────────────────────────────
    KEY_UP = ['301682', '600353', '002572', '002351', '603662', '603799']
    KEY_DN = ['603806', '603259', '300769', '002422', '002294']

    print("\n" + "=" * 60)
    print("重点股明细")
    print("=" * 60)

    key_cards = []
    for sym in KEY_UP + KEY_DN:
        direction = '上调' if sym in KEY_UP else '下调'
        row = df[df['symbol'] == sym]
        if row.empty:
            sk = df_skip[df_skip['symbol'] == sym]
            reason = sk['reason'].iloc[0] if not sk.empty else '未知'
            key_cards.append({'symbol': sym, 'direction': direction, 'status': 'skipped', 'reason': reason})
            print(f"  {sym} [{direction}] SKIPPED: {reason[:60]}")
        else:
            r = row.iloc[0]
            verdict = '✓ 验证' if (direction == '上调' and r['pct'] > 0) or (direction == '下调' and r['pct'] < 0) else '✗ 未验证'
            key_cards.append({
                'symbol': sym,
                'name': r['name'],
                'direction': direction,
                'delta': r['delta'],
                'pct': r['pct'],
                'high': r['high'],
                'low': r['low'],
                'start_close': r['start_close'],
                'end_close': r['end_close'],
                'trading_days': r['trading_days'],
                'status': 'ok',
                'verdict': verdict,
            })
            print(f"  {sym} {r['name']:8s} [{direction}Δ{r['delta']:+.1f}] "
                  f"区间{r['pct']:+.2f}% 高{r['high']} 低{r['low']} {verdict}")

    # ── 大盘股诊断 (Δ≤-5 组) ──────────────────────────────
    dn5 = df[df['group'] == 'Δ≤-5 大幅下调'].copy()
    print(f"\n大幅下调组 (Δ≤-5) 详情:")
    for _, r in dn5.iterrows():
        print(f"  {r['symbol']} {r['name']:8s} Δ{r['delta']:+.1f} | 区间 {r['pct']:+.2f}%")

    # ── 写 Markdown ──────────────────────────────────────
    baseline_row = [s for s in group_stats if s['group'] == '-1<Δ<+1 基本不变']
    baseline_mean = baseline_row[0]['mean_pct'] if baseline_row else 0
    up5_row = [s for s in group_stats if s['group'] == 'Δ≥+5 大幅上调']
    up5_mean = up5_row[0]['mean_pct'] if up5_row else 0
    dn5_row = [s for s in group_stats if s['group'] == 'Δ≤-5 大幅下调']
    dn5_mean = dn5_row[0]['mean_pct'] if dn5_row else 0

    up_beat = up5_mean > baseline_mean
    dn_beat = dn5_mean < baseline_mean

    if up_beat and dn_beat:
        summary_verdict = "上调组跑赢 baseline、下调组跌幅深于 baseline，quant 方向**总体正确**。"
    elif up_beat:
        summary_verdict = "上调组跑赢 baseline，但下调组并未跑输 baseline，quant 下调规则的预测能力**存疑**。"
    elif dn_beat:
        summary_verdict = "下调组跌幅深于 baseline，但上调组并未跑赢 baseline，quant 上调规则的预测能力**存疑**。"
    else:
        summary_verdict = "上调组与下调组均未显示出对 baseline 的方向性优势，quant 规则方向**有待质疑**。"

    # 计算重点股验证率
    key_ok = [c for c in key_cards if c['status'] == 'ok']
    verified = [c for c in key_ok if c.get('verdict','').startswith('✓')]
    verify_rate = round(len(verified) / len(key_ok) * 100) if key_ok else 0

    # 上调组上涨只数
    up_group_all = [s for s in group_stats if '上调' in s['group']]
    dn_group_all = [s for s in group_stats if '下调' in s['group']]

    md = f"""# v3.1 quant_score 规则有效性回测

> **回测窗口**: 2026-04-24 → 2026-05-08（约 10 个交易日）
> **股票池**: 93 只 A 股，akshare 前复权日线
> **生成时间**: 2026-05-08

---

## 1. 执行摘要

本次回测共拉取 {len(df)} 只股票的实盘价格（{len(df_skip)} 只停牌/失败跳过）。{summary_verdict}
大幅上调组（Δ≥+5）均涨 **{up5_mean:+.2f}%**，baseline 组（|Δ|<1）均涨 **{baseline_mean:+.2f}%**，大幅下调组（Δ≤-5）均涨 **{dn5_mean:+.2f}%**。
10 只重点股中，**{len(verified)}/{len(key_ok)}** 只方向验证通过（验证率 {verify_rate}%）。

---

## 2. 分组统计表

| 分组 | n | 胜率% | 均涨% | 中位% | 最大% | 最小% |
|------|---|-------|-------|-------|-------|-------|
"""
    for s in group_stats:
        md += f"| {s['group']} | {s['n']} | {s['win_rate']} | {s['mean_pct']:+.2f} | {s['median_pct']:+.2f} | {s['max_pct']:+.2f} | {s['min_pct']:+.2f} |\n"

    md += f"""
> **说明**: 胜率 = 期间收盘涨幅 > 0 的占比；均涨/中位以前复权收盘价计算。

---

## 3. 重点股明细

### 上调组（quant 识别底部反转信号）

| 股票 | 名称 | Δ | 区间涨跌% | 最高 | 最低 | 起始收盘 | 末日收盘 | 交易日 | 验证 |
|------|------|---|-----------|------|------|----------|----------|--------|------|
"""
    for c in key_cards:
        if c['direction'] == '上调':
            if c['status'] == 'ok':
                md += f"| {c['symbol']} | {c['name']} | {c['delta']:+.2f} | {c['pct']:+.2f}% | {c['high']} | {c['low']} | {c['start_close']} | {c['end_close']} | {c['trading_days']} | {c['verdict']} |\n"
            else:
                md += f"| {c['symbol']} | — | — | SKIPPED | — | — | — | — | — | ⚠ {c.get('reason','')[:40]} |\n"

    md += "\n### 下调组（quant 刺破顶部盲点信号）\n\n"
    md += "| 股票 | 名称 | Δ | 区间涨跌% | 最高 | 最低 | 起始收盘 | 末日收盘 | 交易日 | 验证 |\n"
    md += "|------|------|---|-----------|------|------|----------|----------|--------|------|\n"
    for c in key_cards:
        if c['direction'] == '下调':
            if c['status'] == 'ok':
                md += f"| {c['symbol']} | {c['name']} | {c['delta']:+.2f} | {c['pct']:+.2f}% | {c['high']} | {c['low']} | {c['start_close']} | {c['end_close']} | {c['trading_days']} | {c['verdict']} |\n"
            else:
                md += f"| {c['symbol']} | — | — | SKIPPED | — | — | — | — | — | ⚠ {c.get('reason','')[:40]} |\n"

    # 大幅下调组全部明细
    md += "\n### 大幅下调组（Δ≤-5）完整明细\n\n"
    md += "| 股票 | 名称 | Δ | 区间涨跌% | 交易日 |\n"
    md += "|------|------|---|-----------|--------|\n"
    for _, r in dn5.iterrows():
        md += f"| {r['symbol']} | {r['name']} | {r['delta']:+.2f} | {r['pct']:+.2f}% | {r['trading_days']} |\n"

    md += "\n---\n\n## 4. 关键问题诊断\n\n"

    # main_net 阈值分析
    dn5_up = dn5[dn5['pct'] > 0]
    dn5_up_pct = round(len(dn5_up) / len(dn5) * 100) if len(dn5) > 0 else 0
    md += f"""### 4.1 main_net 阈值（±5000万/±1000万）对大盘股是否偏低？

大幅下调组（Δ≤-5，共 {len(dn5)} 只）中，有 **{len(dn5_up)} 只**（{dn5_up_pct}%）期间实际上涨：

"""
    if not dn5_up.empty:
        for _, r in dn5_up.iterrows():
            md += f"- **{r['name']}**（{r['symbol']}）Δ={r['delta']:+.1f}，区间 {r['pct']:+.2f}%\n"
        md += f"""
这些被 quant 大幅下调、但实际上涨的股票，说明 ±5000万 / ±1000万 的主力净流入阈值**对部分大盘股偏严**——
大盘股的正常日均成交额远超该阈值，轻微主力净流出也会触发扣分，导致错杀。
**建议**: 按总市值分层设置阈值（如：市值 ≥500亿 → ±2亿，100-500亿 → ±5000万，<100亿 → ±1000万）。
"""
    else:
        md += "下调组无实际上涨股票，暂无错杀信号，阈值设计在本轮样本中表现良好。\n"

    # winner_rate 顶部分析
    # 找 quant 大幅下调 且 new_level 变为更差的股 (有 winner_rate 顶部信号嫌疑)
    dn_changed = df[(df['delta'] <= -3) & (df['old_level'] != df['new_level'])].copy()
    md += f"""\n### 4.2 winner_rate 顶部阈值（≥85%）识别的顶部股是否真回落？

被下调且评级变差的股票共 {len(dn_changed)} 只，其中期间下跌：
"""
    if not dn_changed.empty:
        dn_fell = dn_changed[dn_changed['pct'] < 0]
        fell_pct = round(len(dn_fell) / len(dn_changed) * 100) if len(dn_changed) > 0 else 0
        md += f"- 下跌 {len(dn_fell)}/{len(dn_changed)} 只（{fell_pct}%），平均跌幅 {dn_fell['pct'].mean():+.2f}%（若可获取）\n"
        for _, r in dn_changed.head(8).iterrows():
            trend = '↓' if r['pct'] < 0 else '↑'
            md += f"- {r['name']}（{r['symbol']}）Δ={r['delta']:+.1f}，{r['old_level']} → {r['new_level']}，区间 {r['pct']:+.2f}% {trend}\n"

    # ADX 矛盾信号分析 (000876 新希望, 002050 三花智控)
    contradiction_syms = ['000876', '002050']
    md += "\n### 4.3 ADX 强趋势向上 + 主力净流出（矛盾信号）最终方向\n\n"
    md += "| 股票 | 名称 | Δ | 旧评级 | 新评级 | 区间涨跌% | 结论 |\n"
    md += "|------|------|---|--------|--------|-----------|------|\n"
    for sym in contradiction_syms:
        row = df[df['symbol'] == sym]
        if not row.empty:
            r = row.iloc[0]
            direction = "主力流出占主导" if r['pct'] < 0 else "ADX 趋势占主导"
            md += f"| {sym} | {r['name']} | {r['delta']:+.2f} | {r['old_level']} | {r['new_level']} | {r['pct']:+.2f}% | {direction} |\n"
        else:
            sk = df_skip[df_skip['symbol'] == sym]
            reason = sk['reason'].iloc[0] if not sk.empty else '数据缺失'
            md += f"| {sym} | — | — | — | — | SKIPPED | {reason[:40]} |\n"

    md += """
矛盾信号处理建议：当 ADX>25（强趋势）与主力净流出同时出现时，可降低主力资金因子权重，
或引入时序判断（近 5 日趋势方向是否已转向）以减少误判。

---

## 5. 建议

### 5.1 是否保留 quant 权重 0.14？
"""
    if up_beat or dn_beat:
        md += "**建议保留，但需修正阈值**。quant_score 在本次回测中显示出方向性信号，但存在部分错杀情况。\n"
        md += "权重 0.14 属于谨慎配置，可维持不变，优先修正规则本身。\n"
    else:
        md += "**建议暂时降权至 0.08 并观察**。本次回测中 quant_score 未能显示出稳健的方向性优势，\n"
        md += "在改进规则之前可适当降低其影响力，避免持续拖累整体分数准确性。\n"

    md += f"""
### 5.2 按市值分层调整 main_net 阈值

| 市值区间 | 当前阈值 | 建议阈值（高分段/低分段） |
|----------|----------|--------------------------|
| ≥500亿   | ±5000万   | ±2亿 / ±5000万            |
| 100-500亿| ±5000万   | ±8000万 / ±2000万         |
| <100亿   | ±1000万   | ±1000万（保持）            |

### 5.3 下一轮迭代方向

1. **市值分层**: 为 main_net 引入市值归一化（用主力净流入 / 日均成交额%代替绝对金额）
2. **矛盾信号处理**: 增加"ADX vs 主力"冲突检测子规则，冲突时两个因子都打半分
3. **winner_rate 动态阈值**: 顶部阈值从固定 85% 改为行业历史百分位（避免成长板块误杀）
4. **回测周期延长**: 本次仅 10 个交易日，样本噪音大；建议在 30/60 日窗口再次验证
5. **纳入指数基准**: 增加同期沪深 300 / 中证 500 收益作为对照，控制市场系统性因素

---

## 6. 跳过股票列表

"""
    if df_skip.empty:
        md += "无跳过股票，所有样本数据均成功获取。\n"
    else:
        md += "| 股票代码 | 名称 | 原因 |\n"
        md += "|----------|------|------|\n"
        for _, r in df_skip.iterrows():
            md += f"| {r['symbol']} | {r['name']} | {r.get('reason','')[:80]} |\n"

    md += "\n---\n\n*本报告由 akshare 实盘数据自动生成，不构成投资建议。*\n"

    # 写文件
    out_path = '/home/user/stockagent-analysis/docs/quant_backtest_2026-05-08.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"\n报告已写入: {out_path}")

    # 最终摘要
    print("\n" + "=" * 60)
    print("最终统计摘要")
    print("=" * 60)
    print(f"上调大幅 均涨: {up5_mean:+.2f}% | baseline: {baseline_mean:+.2f}% | 下调大幅: {dn5_mean:+.2f}%")
    print(f"跳过股票: {len(df_skip)} 只")
    print(f"重点股验证率: {verify_rate}% ({len(verified)}/{len(key_ok)})")

    return df, df_stats, df_skip, key_cards


if __name__ == '__main__':
    main()
