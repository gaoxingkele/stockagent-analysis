#!/usr/bin/env python3
"""V11 实战测试: 5 只半导体/精密制造股 (用户指定).

300567 精测电子
688037 芯源微
688577 浙海德曼
688409 富创精密
688720 艾森股份

输出: 每股 3 张 K 线 PNG + LLM 视觉分析 (5 域 + 缠论 + Elliott + 多场景)
"""
from __future__ import annotations
import os, struct, json, time, base64, re
from pathlib import Path
from datetime import datetime, date as dt_date
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.cloubic")
load_dotenv(".env")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from stockagent_analysis.chart_generator import generate_kline_chart

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "v11_test_5"
OUT_DIR.mkdir(exist_ok=True, parents=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

CLOUBIC_KEY = os.environ["CLOUBIC_API_KEY"]
CLOUBIC_BASE = "https://api.cloubic.com/v1"

STOCKS = [
    ("300567", "sz", "精测电子"),
    ("688037", "sh", "芯源微"),
    ("688577", "sh", "浙海德曼"),
    ("688409", "sh", "富创精密"),
    ("688720", "sh", "艾森股份"),
]

TODAY = datetime.now().strftime("%Y%m%d")


SYS_PROMPT_V11 = """你是 A 股专业 K 线视觉技术分析师, 严格基于图像本身做客观分析 (不依赖外部信息).

你将看到同一只股票的 3 张 K 线综合图 (按顺序: 月线 / 周线 / 日线), 每张图含蜡烛+MA5/10/20/60/120/250+布林带+趋势线+成交量+MACD+RSI+KDJ.

## 分析框架 (必须覆盖 5 大域)

### 1. 趋势分析 (Dow Theory)
- 主要趋势 (月线): 上升/下降/震荡, 强度
- 次要趋势 (周线): 是否与主要趋势同向?
- 短期趋势 (日线): 当前位置
- HH/HL pattern (Higher High/Higher Low) 是否完整?

### 2. 支撑/阻力 (3+ touches 才算 strong)
- 月线/周线 关键长期支撑阻力位 (具体价格)
- 日线 短期支撑阻力位
- 角色翻转 (broken support → resistance)

### 3. 移动均线
- 价格 vs MA20/60/250 关系 (above/below/between)
- 金叉/死叉 (Golden Cross / Death Cross)
- MA 排列 (bullish 多头排列 / bearish 空头排列)

### 4. 成交量
- 量价配合 (rising price + rising volume = healthy)
- 量价背离 (rising price + falling volume = weak)
- 异常放量/缩量

### 5. 形态 + 缠论 + Elliott Wave
- 反转形态: hammer/engulfing/双底/双顶/M 头/W 底
- 延续形态: bull flag/bear flag/triangle
- **缠论视角**: 当前是顶分型/底分型? 在中枢震荡? 突破中枢? 接近哪类买卖点 (一类背驰/二类回撤/三类突破)?
- **Elliott Wave**: 周线视角下当前在主升浪 (1/3/5 推动) 还是修正浪 (a/b/c)? 关键 Fibonacci 位 (38.2/50/61.8)

## 输出要求

输出 JSON 对象 (不要 markdown 包裹):
{
  "trend_main": "uptrend/downtrend/sideways",
  "trend_strength": "strong/moderate/weak",
  "ma_alignment": "bullish/bearish/neutral",
  "volume_confirms_price": true/false,
  "key_pattern": "50字内核心形态描述, 含缠论术语",
  "elliott_phase": "30字内 Elliott Wave 当前位置",
  "support_levels": [价位1, 价位2, ...],
  "resistance_levels": [价位1, 价位2, ...],
  "scenarios": [
    {"name": "Bull", "prob": 0.0-1.0, "target_pct_20d": 数值, "invalidation": "30字内"},
    {"name": "Base", "prob": 0.0-1.0, "target_pct_20d": 数值, "invalidation": "30字内"},
    {"name": "Bear", "prob": 0.0-1.0, "target_pct_20d": 数值, "invalidation": "30字内"}
  ]
}

scenarios 概率求和必须等于 1.0. target_pct_20d 是该场景下 20 个交易日预期涨跌幅 (% as number)."""


def read_tdx(market, code, end_date_inclusive):
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    data = p.read_bytes()
    n = len(data) // 32
    if n == 0: return None
    rows = []
    end_int = int(end_date_inclusive)
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        if di > end_int: continue
        try: d = dt_date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append({
            "ts": d.strftime("%Y-%m-%d"),
            "open": f[1]/100.0, "high": f[2]/100.0,
            "low": f[3]/100.0, "close": f[4]/100.0,
            "volume": float(f[6]),
        })
    return rows


def daily_to_periodic(rows, freq):
    if not rows: return []
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["ts"])
    df = df.set_index("dt")
    if freq == "W":
        agg = df.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    elif freq == "M":
        agg = df.resample("ME").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    else:
        return rows
    agg["ts"] = agg.index.strftime("%Y-%m-%d")
    return agg[["ts","open","high","low","close","volume"]].to_dict("records")


def gen_3tf(code, market, name):
    rows = read_tdx(market, code, TODAY)
    if not rows or len(rows) < 60:
        return None, None, None, None
    last_close = rows[-1]["close"]
    last_date = rows[-1]["ts"]
    paths = {}
    for tf, freq, take_n in [("month","M",120), ("week","W",60), ("day",None,60)]:
        target_path = OUT_DIR / f"{code}_{name}_{tf}.png"
        if freq:
            data = daily_to_periodic(rows, freq)
        else:
            data = rows[-take_n:]
        if not data or len(data) < 10:
            paths[tf] = None; continue
        data = data[-take_n:]
        df = pd.DataFrame(data)
        png = generate_kline_chart(df, tf, code, name, save_path=target_path)
        paths[tf] = target_path if png else None
    return paths.get("month"), paths.get("week"), paths.get("day"), (last_close, last_date)


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_robust(text):
    if not text: return None
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m: text = m.group(1)
    else:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m: text = m.group(0)
    try: return json.loads(text)
    except: return None


def call_vision(client, m_b64, w_b64, d_b64):
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{m_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{w_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{d_b64}"}},
        {"type": "text", "text": "三张图按顺序: 月线 (10 年)、周线 (60 周)、日线 (60 日). 按 5 域框架做分析, 输出完整 JSON."}
    ]
    resp = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": SYS_PROMPT_V11},
            {"role": "user", "content": user_content},
        ],
        max_tokens=800, temperature=0.0,
    )
    return resp.choices[0].message.content, resp.usage


def main():
    t0 = time.time()
    client = OpenAI(api_key=CLOUBIC_KEY, base_url=CLOUBIC_BASE)
    total_in = total_out = 0
    cost = 0.0

    for code, market, name in STOCKS:
        print(f"\n{'='*70}", flush=True)
        print(f"### {code} {name} ({market})", flush=True)
        print(f"{'='*70}", flush=True)
        m_p, w_p, d_p, info = gen_3tf(code, market, name)
        if not (m_p and w_p and d_p):
            print(f"  ❌ 图生成失败"); continue
        last_close, last_date = info
        print(f"  最新: {last_date}, close={last_close:.2f}", flush=True)
        print(f"  图: {OUT_DIR}/{code}_*.png", flush=True)

        m_b64 = encode_image_b64(m_p)
        w_b64 = encode_image_b64(w_p)
        d_b64 = encode_image_b64(d_p)
        text, usage = call_vision(client, m_b64, w_b64, d_b64)
        if usage:
            total_in += usage.prompt_tokens
            total_out += usage.completion_tokens
            cost += (usage.prompt_tokens * 3.0 + usage.completion_tokens * 15.0) / 1e6

        j = parse_json_robust(text)
        if j is None:
            print(f"  ⚠ 解析失败, raw: {(text or '')[:200]}", flush=True)
            continue

        # 格式化输出
        print(f"\n  【趋势】 主要={j.get('trend_main')}, 强度={j.get('trend_strength')}", flush=True)
        print(f"  【MA】 排列={j.get('ma_alignment')}, 量价配合={j.get('volume_confirms_price')}", flush=True)
        print(f"  【支撑】 {j.get('support_levels', [])}", flush=True)
        print(f"  【阻力】 {j.get('resistance_levels', [])}", flush=True)
        print(f"  【形态】 {j.get('key_pattern', '')}", flush=True)
        print(f"  【Elliott】 {j.get('elliott_phase', '')}", flush=True)
        print(f"\n  【3 场景】 ", flush=True)
        for s in j.get("scenarios", []):
            print(f"    {s.get('name','')}: {s.get('prob',0)*100:.0f}% → {s.get('target_pct_20d',0):+.1f}% (失效: {s.get('invalidation','')})",
                  flush=True)
        print(f"\n  累计成本: ${cost:.4f}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s, token in={total_in:,} out={total_out:,}, 总成本 ${cost:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
