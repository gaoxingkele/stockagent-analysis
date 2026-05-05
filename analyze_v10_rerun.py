#!/usr/bin/env python3
"""V10 重跑: 复用已 saved PNG, 只重调 LLM + 鲁棒 JSON 解析.

V10 PoC 50 样本全 parse_error 因 Claude 返回 ```json...``` markdown wrap.
本脚本:
  1. 复用 output/v10/charts/{ts_code}_{date}_{day,week}.png (已 100 张)
  2. Sonnet 4.6 视觉调用 (跟原版相同 prompt)
  3. 鲁棒 JSON 解析 (剥离 markdown fence + 正则提取 JSON 对象)
  4. 输出 output/v10/poc_results_v2.csv
"""
from __future__ import annotations
import os, json, time, re, base64
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.cloubic")
load_dotenv(".env")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "v10"
CHARTS_DIR = OUT_DIR / "charts"
CLOUBIC_KEY = os.environ["CLOUBIC_API_KEY"]
CLOUBIC_BASE = "https://api.cloubic.com/v1"


SYS_PROMPT_IDEAL = """你是 A 股 K 线视觉技术分析师 (重要: 严格基于图像本身判断, 不依赖外部信息).

背景: V7c 量化系统已推荐这只股票 (高 buy + 低 sell, 4 象限"理想多" 段, OOS 实测 r20=+6.31%).
你的任务: 看日线 + 周线 K 线综合图, 判断 V7c 的"温和上涨"判断是否成立.

判断角度:
1. 日线/周线趋势是否一致看多?
2. 是否处于"健康上涨" (温和阳线 + 量能配合 + 均线支撑) 而非"派发拉高" (放量长上影/超买)?
3. MACD/RSI/KDJ 是否在合理区间, 还是已极度超买?
4. 周线视角下, 当前位置是趋势中段还是末端?

直接输出 JSON 对象 (不要 markdown 包裹, 不要 ```json fence):
{"verdict": "confirm" 或 "reject", "conviction": 0-100, "key_pattern": "30字内", "risk_warning": "30字内"}"""


SYS_PROMPT_CONTRADICTION = """你是 A 股 K 线视觉解谜专家 (重要: 严格基于图像本身判断).

背景: V7c 系统识别这只股为"矛盾段" (高 buy + 高 sell, OOS r20=+3.07% 但 dd<-15%=29% 风险高).
你的任务: 看日线 + 周线 K 线综合图, 判断这是"洗盘后突破"还是"顶部派发".

判断角度:
1. 周线视角: 长期趋势是上升通道中的洗盘, 还是顶部 M 头?
2. 日线视角: 异常波动是缩量洗盘 (蓄势)还是放量派发 (出货)?
3. 量价是否健康配合 (上涨放量, 调整缩量)?
4. 关键支撑位是否有效?

直接输出 JSON 对象 (不要 markdown 包裹, 不要 ```json fence):
{"verdict": "真涨" 或 "假涨" 或 "不确定", "conviction": 0-100, "key_pattern": "30字内", "risk_warning": "30字内"}"""


def parse_json_robust(text: str):
    """鲁棒 JSON 解析: 剥离 markdown fence + 正则提取 JSON 对象."""
    if not text: return None
    text = text.strip()
    # 1. 剥离 ```json...``` fence
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        # 2. 直接找 JSON 对象 (第一个 { 到最后一个 })
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        return json.loads(text)
    except:
        return None


def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_dual(client, daily_b64, weekly_b64, sys_prompt, max_retry=2):
    user_content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{daily_b64}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{weekly_b64}"}},
        {"type": "text",
         "text": "第一张是日线 (60 日), 第二张是周线 (60 周). 请综合判断, 直接输出 JSON 对象."}
    ]
    for r in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=400, temperature=0.0,
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if r < max_retry - 1:
                time.sleep(2); continue
            return None, None


def main():
    t0 = time.time()
    # 加载原 PoC 样本元数据 (含 r20 等真值)
    df_orig = pd.read_csv(OUT_DIR / "poc_results.csv")
    print(f"加载 {len(df_orig)} 样本", flush=True)

    client = OpenAI(api_key=CLOUBIC_KEY, base_url=CLOUBIC_BASE)
    total_in = total_out = 0
    cost = 0.0
    new_results = []

    for i, row in df_orig.iterrows():
        ts_code = row["ts_code"]; trade_date = str(row["trade_date"])
        task = row["task"]
        sys_prompt = SYS_PROMPT_IDEAL if task == "理想多" else SYS_PROMPT_CONTRADICTION

        daily_p = CHARTS_DIR / f"{ts_code}_{trade_date}_day.png"
        weekly_p = CHARTS_DIR / f"{ts_code}_{trade_date}_week.png"
        if not daily_p.exists() or not weekly_p.exists():
            new_results.append({**row.to_dict(), "verdict_v2": "no_image",
                                  "conviction_v2": 0, "key_pattern_v2": "PNG 不存在",
                                  "risk_warning_v2": ""})
            continue

        daily_b64 = load_image_b64(daily_p)
        weekly_b64 = load_image_b64(weekly_p)
        text, usage = call_vision_dual(client, daily_b64, weekly_b64, sys_prompt)
        if usage:
            total_in += usage.prompt_tokens
            total_out += usage.completion_tokens
            cost += (usage.prompt_tokens * 3.0 + usage.completion_tokens * 15.0) / 1e6

        j = parse_json_robust(text)
        if j is None:
            new_results.append({**row.to_dict(), "verdict_v2": "parse_error_v2",
                                  "conviction_v2": 0,
                                  "key_pattern_v2": (text or "")[:80],
                                  "risk_warning_v2": ""})
            print(f"  [{i+1}/{len(df_orig)}] {ts_code} parse fail: {(text or '')[:60]}",
                  flush=True)
        else:
            new_results.append({**row.to_dict(),
                                  "verdict_v2": str(j.get("verdict","unknown")),
                                  "conviction_v2": int(j.get("conviction",50)),
                                  "key_pattern_v2": str(j.get("key_pattern",""))[:80],
                                  "risk_warning_v2": str(j.get("risk_warning",""))[:80]})
            print(f"  [{i+1}/{len(df_orig)}] {ts_code} {trade_date} → {new_results[-1]['verdict_v2']:8s} conv={new_results[-1]['conviction_v2']:3d} ${cost:.3f}",
                  flush=True)

    res = pd.DataFrame(new_results)
    res.to_csv(OUT_DIR / "poc_results_v2.csv", index=False, encoding="utf-8-sig")

    # ── 评估 ──
    print(f"\n[{int(time.time()-t0)}s] === 评估 ===", flush=True)

    print(f"\n# Task A 理想多:", flush=True)
    a = res[res["task"] == "理想多"]
    print(a["verdict_v2"].value_counts().to_string(), flush=True)
    for v in ["confirm","reject","unknown","parse_error_v2","no_image"]:
        sub = a[a["verdict_v2"] == v]
        if len(sub) >= 3:
            r20 = sub["r20"].dropna()
            print(f"  {v:8s} (n={len(sub):2d}): r20={r20.mean():+.2f}% win={(r20>0).mean()*100:.0f}%",
                  flush=True)
    confirm = a[a["verdict_v2"] == "confirm"]
    reject = a[a["verdict_v2"] == "reject"]
    if len(confirm) >= 5 and len(reject) >= 5:
        t, p = stats.ttest_ind(confirm["r20"].dropna(), reject["r20"].dropna())
        print(f"  confirm vs reject diff={confirm['r20'].mean()-reject['r20'].mean():+.2f}pp, p={p:.4f}",
              flush=True)
    if len(a) > 5:
        hi = a[a["conviction_v2"] >= 70]
        lo = a[a["conviction_v2"] < 50]
        if len(hi) >= 3 and len(lo) >= 3:
            print(f"  conv>=70 (n={len(hi)}): r20={hi['r20'].mean():+.2f}% vs <50 (n={len(lo)}): r20={lo['r20'].mean():+.2f}%",
                  flush=True)

    print(f"\n# Task B 矛盾段:", flush=True)
    b = res[res["task"] == "矛盾段"]
    print(b["verdict_v2"].value_counts().to_string(), flush=True)
    for v in ["真涨","假涨","不确定","parse_error_v2"]:
        sub = b[b["verdict_v2"] == v]
        if len(sub) >= 3:
            r20 = sub["r20"].dropna()
            dd15 = (sub["max_dd_20"].dropna() <= -15).mean() * 100
            print(f"  {v:8s} (n={len(sub):2d}): r20={r20.mean():+.2f}% win={(r20>0).mean()*100:.0f}% dd<-15%={dd15:.0f}%",
                  flush=True)
    real_up = b[b["verdict_v2"] == "真涨"]
    fake_up = b[b["verdict_v2"] == "假涨"]
    if len(real_up) >= 3 and len(fake_up) >= 3:
        t, p = stats.ttest_ind(real_up["r20"].dropna(), fake_up["r20"].dropna())
        print(f"  真涨 vs 假涨 diff={real_up['r20'].mean()-fake_up['r20'].mean():+.2f}pp, p={p:.4f}",
              flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s, token in={total_in:,} out={total_out:,}, 成本 ${cost:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
