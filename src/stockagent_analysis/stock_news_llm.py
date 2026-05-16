"""个股新闻 LLM 情感+催化分析 (Sprint 3.1).

数据源: AKShare ak.stock_news_em(symbol) - 东方财富个股新闻
LLM: claude-sonnet-4-6 抽取情感 + 催化类型

输入: 单股最近 N 日新闻
输出: {sentiment, catalyst_type, key_events, hotness}

集成到 V12: 加 news_sentiment / catalyst_type / news_hotness 字段
触发 catalyst_type ∈ {业绩预增, 中标订单} → buy_score 加 5 分
"""
from __future__ import annotations
import os, json, re, time
from pathlib import Path
from typing import Optional
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
CLOUBIC_BASE = "https://api.cloubic.com/v1"
LLM_MODEL = "claude-sonnet-4-6"

SYS_PROMPT = """你是 A 股个股新闻分析师, 抽取个股短期催化和情感.

输入: 单只股票最近 7-14 日的新闻列表 (标题+摘要).

任务:
1. 整体情感分: -1.0 (大空) ~ +1.0 (大多)
2. 催化类型 (选一个最强):
   - "业绩预增" (业绩快报/年报/季报超预期)
   - "中标订单" (大额合同/客户突破)
   - "回购增持" (公司/股东回购或增持)
   - "股权激励" (员工持股/期权)
   - "并购重组" (M&A/资产注入)
   - "新业务" (进入新市场/新产品)
   - "政策受益" (政策红利明确受益)
   - "诉讼风险" (重大诉讼/被处罚)
   - "业绩下滑" (业绩低于预期)
   - "高管减持" (大股东/高管减持)
   - "无" (无明显催化)
3. 关键事件 (最多 3 个, 30 字内每个)
4. 热度 0-100 (新闻数量 × 重要性)

输出 JSON (不要 markdown 包裹):
{
  "sentiment_score": -1.0 ~ 1.0,
  "catalyst_type": "类型",
  "key_events": ["事件1", "事件2"],
  "hotness": 0-100,
  "reason": "30 字内综合判断"
}"""


class StockNewsAnalyzer:
    def __init__(self, cloubic_api_key: str):
        self.client = OpenAI(api_key=cloubic_api_key, base_url=CLOUBIC_BASE)

    def fetch_news(self, symbol: str, lookback_days: int = 30) -> list[dict]:
        """拉单股公告 (东方财富 API, symbol 是 6 位数字 不带后缀)."""
        import requests
        from datetime import datetime, timedelta
        url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
        params = {
            "sr": -1, "page_size": 30, "page_index": 1,
            "ann_type": "A", "client_source": "web",
            "stock_list": symbol,
            "f_node": 0, "s_node": 0,
        }
        try:
            r = requests.get(url, params=params, timeout=10,
                              headers={"User-Agent": "Mozilla/5.0"})
            d = r.json()
            if not d.get("success") or "data" not in d: return []
            items = d["data"].get("list", [])
        except Exception:
            return []

        cutoff = datetime.now() - timedelta(days=lookback_days)
        news = []
        for it in items:
            t = it.get("display_time", "")[:10]
            try:
                pub = datetime.strptime(t, "%Y-%m-%d")
                if pub < cutoff: continue
            except Exception:
                pass
            news.append({
                "title": str(it.get("title", ""))[:150],
                "time": t,
                "type": str(it.get("columns", [{}])[0].get("column_name", "")
                             if it.get("columns") else "")[:20],
            })
        return news[:25]

    def analyze(self, symbol: str, lookback_days: int = 14) -> dict:
        """完整流程: 拉新闻 → LLM → 输出."""
        news = self.fetch_news(symbol, lookback_days)
        if not news:
            return {"symbol": symbol, "sentiment_score": 0.0,
                     "catalyst_type": "无", "key_events": [],
                     "hotness": 0, "n_news": 0}

        news_text = "\n".join(
            f"[{n['time']}] ({n.get('type','')}) {n['title']}"
            for n in news[:25]
        )

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": f"股票代码: {symbol}\n最近 {lookback_days} 日新闻 ({len(news)} 条):\n\n{news_text}"},
                ],
                max_tokens=600, temperature=0.1,
            )
        except Exception as e:
            return {"symbol": symbol, "error": str(e)[:120], "n_news": len(news)}

        text = resp.choices[0].message.content
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m: text = m.group(1)
        else:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m: text = m.group(0)
        try:
            data = json.loads(text)
        except Exception as e:
            return {"symbol": symbol, "n_news": len(news), "raw": text[:200],
                     "parse_error": str(e)}

        return {
            "symbol": symbol,
            "n_news": len(news),
            "sentiment_score": float(data.get("sentiment_score", 0)),
            "catalyst_type": str(data.get("catalyst_type", "无"))[:20],
            "key_events": data.get("key_events", [])[:3],
            "hotness": float(data.get("hotness", 0)),
            "reason": str(data.get("reason", ""))[:60],
            "llm_cost_usd": round((resp.usage.prompt_tokens * 3.0 +
                                     resp.usage.completion_tokens * 15.0) / 1e6, 4),
        }


# 兼容 pandas
import pandas as pd
