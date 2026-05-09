#!/usr/bin/env python3
"""V11 + V7c 联合评估: 11 只北交所股 (用户指定 2026-05-08).

策略:
  - 数据源走 Tushare cache (output/tushare_cache/daily/), 不再读 TDX
  - 11 股 × 3 张 K 线图 (月/周/日) → Sonnet 4.6 视觉分析
  - V7c 命中的 4 只附评分对照 (其它 7 只 factor_lab 无收录, 只 LLM)
  - 输出综合报告 (LLM 多场景 + V7c buy/sell + 行业)
"""
from __future__ import annotations
import os, json, time, base64, re
from pathlib import Path
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env.cloubic")
load_dotenv(ROOT / ".env")

import sys
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.chart_generator import generate_kline_chart

OUT_DIR = ROOT / "output" / "v11_bj11"
OUT_DIR.mkdir(exist_ok=True, parents=True)
DAILY_CACHE = ROOT / "output" / "tushare_cache" / "daily"

CLOUBIC_KEY = os.environ["CLOUBIC_API_KEY"]
CLOUBIC_BASE = "https://api.cloubic.com/v1"

CODES = ["920981","920012","920181","920186","920125","920078",
         "920177","920046","920011","920111","920152"]
END_DATE = "20260508"


SYS_PROMPT = """你是 A 股专业 K 线视觉技术分析师, 严格基于图像本身做客观分析.

你将看到同一只股票的 3 张 K 线综合图 (按顺序: 月线 / 周线 / 日线), 每张图含蜡烛+MA5/10/20/60/120/250+布林带+成交量+MACD+RSI+KDJ.

## 分析框架 (必须覆盖 5 大域)

1. 趋势 (主/次/短): 上升/下降/震荡 + 强度 + HH/HL pattern
2. 支撑/阻力: 月线/周线长期 + 日线短期 (具体价格)
3. 移动均线: 价格 vs MA20/60/250 + 金叉死叉 + 多/空头排列
4. 成交量: 量价配合/背离, 异常放量/缩量
5. 形态 + 缠论 + Elliott Wave:
   - 反转 (双底/双顶/M 头/W 底) / 延续 (旗形/三角)
   - 缠论: 顶底分型? 中枢震荡/突破? 一/二/三类买卖点?
   - Elliott: 主升 1/3/5 还是修正 a/b/c, 关键 Fibonacci 位

## 输出 JSON (不要 markdown 包裹)

{
  "trend_main": "uptrend/downtrend/sideways",
  "trend_strength": "strong/moderate/weak",
  "ma_alignment": "bullish/bearish/neutral",
  "volume_confirms_price": true/false,
  "key_pattern": "50字内核心形态描述, 含缠论术语",
  "elliott_phase": "30字内 Elliott Wave 当前位置",
  "support_levels": [价位1, 价位2],
  "resistance_levels": [价位1, 价位2],
  "scenarios": [
    {"name": "Bull", "prob": 0-1, "target_pct_20d": 数值, "invalidation": "30字内"},
    {"name": "Base", "prob": 0-1, "target_pct_20d": 数值, "invalidation": "30字内"},
    {"name": "Bear", "prob": 0-1, "target_pct_20d": 数值, "invalidation": "30字内"}
  ]
}

scenarios 概率求和必须 = 1.0. target_pct_20d 是该场景下未来 20 交易日预期涨跌 % (数值)."""


def load_daily_for(ts_codes_full):
    """从 Tushare cache 加载这 N 股的全历史 daily."""
    files = sorted(DAILY_CACHE.glob("*.parquet"))
    parts = []
    for f in files:
        if f.stem > END_DATE: continue
        d = pd.read_parquet(f)
        d = d[d["ts_code"].isin(ts_codes_full)]
        if not d.empty: parts.append(d)
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    by_code = {}
    for ts, g in big.groupby("ts_code"):
        rows = []
        for _, r in g.iterrows():
            rows.append({
                "ts": f"{r['trade_date'][:4]}-{r['trade_date'][4:6]}-{r['trade_date'][6:8]}",
                "open": float(r["open"]), "high": float(r["high"]),
                "low":  float(r["low"]),  "close": float(r["close"]),
                "volume": float(r["vol"]),
            })
        by_code[ts] = rows
    return by_code


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


def gen_3tf(rows, code, name):
    if not rows or len(rows) < 60:
        return None, None, None, None
    last_close = rows[-1]["close"]
    last_date = rows[-1]["ts"]
    paths = {}
    for tf, freq, take_n in [("month","M",120), ("week","W",60), ("day",None,60)]:
        target_path = OUT_DIR / f"{code}_{name}_{tf}.png"
        data = daily_to_periodic(rows, freq) if freq else rows
        if not data or len(data) < 10:
            paths[tf] = None; continue
        data = data[-take_n:]
        df = pd.DataFrame(data)
        png = generate_kline_chart(df, tf, code, name, save_path=target_path)
        paths[tf] = target_path if png else None
    return paths.get("month"), paths.get("week"), paths.get("day"), (last_close, last_date)


def encode_b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode("utf-8")


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
        {"type": "text", "text": "三张图按顺序: 月线 / 周线 / 日线. 按 5 域框架做分析, 输出完整 JSON."}
    ]
    resp = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=800, temperature=0.0,
    )
    return resp.choices[0].message.content, resp.usage


def load_v7c_for(codes_full):
    """从 V7c 0507 推理结果里捞这些股."""
    p = ROOT / "output" / "v7c_full_inference" / "v7c_inference_20260507.csv"
    if not p.exists(): return {}
    df = pd.read_csv(p)
    sub = df[df["ts_code"].isin(codes_full)]
    return {r["ts_code"]: dict(r) for _, r in sub.iterrows()}


def main():
    t0 = time.time()
    codes_full = [f"{c}.BJ" for c in CODES]
    print(f"=== V11 + V7c 联合评估 11 BJ 股 ({END_DATE}) ===\n", flush=True)

    print("加载 daily cache ...", flush=True)
    daily_map = load_daily_for(codes_full)
    print(f"  命中 {len(daily_map)}/{len(codes_full)}", flush=True)

    v7c_map = load_v7c_for(codes_full)
    print(f"  V7c 命中 {len(v7c_map)} 股\n", flush=True)

    client = OpenAI(api_key=CLOUBIC_KEY, base_url=CLOUBIC_BASE)
    total_in = total_out = 0
    cost = 0.0
    results = []

    for ts_code in codes_full:
        rows = daily_map.get(ts_code)
        if not rows or len(rows) < 60:
            print(f"\n### {ts_code}: 数据不足 ({len(rows) if rows else 0} 行), 跳过", flush=True)
            continue
        code = ts_code.split(".")[0]
        v7c = v7c_map.get(ts_code)
        ind = (v7c or {}).get("industry", "?")
        name = f"BJ_{code}"

        print(f"\n{'='*70}", flush=True)
        print(f"### {ts_code} ({ind}) — {len(rows)} 日历史", flush=True)
        print(f"{'='*70}", flush=True)

        m_p, w_p, d_p, info = gen_3tf(rows, code, name)
        if not (m_p and w_p and d_p):
            print(f"  图生成失败"); continue
        last_close, last_date = info
        print(f"  最新: {last_date}, close={last_close:.2f}", flush=True)

        text, usage = call_vision(client, encode_b64(m_p), encode_b64(w_p), encode_b64(d_p))
        if usage:
            total_in += usage.prompt_tokens
            total_out += usage.completion_tokens
            cost += (usage.prompt_tokens * 3.0 + usage.completion_tokens * 15.0) / 1e6

        j = parse_json_robust(text)
        if j is None:
            print(f"  解析失败: {(text or '')[:200]}", flush=True)
            continue

        if v7c is not None:
            print(f"\n  【V7c】 buy={v7c.get('buy_score',0):.0f}  sell={v7c.get('sell_score',0):.0f}  "
                  f"象限={v7c.get('quadrant','')}  推荐={v7c.get('v7c_recommend',False)}  "
                  f"r10={v7c.get('r10_pred',0):+.2f}%  r20={v7c.get('r20_pred',0):+.2f}%", flush=True)
        else:
            print(f"\n  【V7c】 不在 factor_lab 训练 5072 股名单, 无评分", flush=True)

        print(f"  【趋势】 主要={j.get('trend_main')}, 强度={j.get('trend_strength')}", flush=True)
        print(f"  【MA】 排列={j.get('ma_alignment')}, 量价={j.get('volume_confirms_price')}", flush=True)
        print(f"  【支撑】 {j.get('support_levels', [])}  【阻力】 {j.get('resistance_levels', [])}", flush=True)
        print(f"  【形态】 {j.get('key_pattern', '')}", flush=True)
        print(f"  【Elliott】 {j.get('elliott_phase', '')}", flush=True)
        for s in j.get("scenarios", []):
            print(f"    {s.get('name',''):>4s}: {s.get('prob',0)*100:>3.0f}% → "
                  f"{s.get('target_pct_20d',0):+.1f}% (失效: {s.get('invalidation','')})", flush=True)

        results.append({"ts_code": ts_code, "industry": ind, "close": last_close,
                         "v7c": v7c, "llm": j})

    out_p = OUT_DIR / "result.json"
    out_p.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n{'='*70}", flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s, in={total_in:,} out={total_out:,}, 成本 ${cost:.4f}", flush=True)
    print(f"输出: {out_p}", flush=True)


if __name__ == "__main__":
    main()
