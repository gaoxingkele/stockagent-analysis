# -*- coding: utf-8 -*-
"""v2 新闻/舆情增强模块。

在 akshare 新闻数据基础上:
1. 可选 Perplexity sonar 搜索增强
2. LLM 结构化提取: 情绪分 + 关键事件 + 风险预警
3. 输出独立的新闻分析结果, 供 SENTIMENT_FLOW Agent 消费
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON。"""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def search_perplexity(query: str, max_results: int = 10) -> list[dict[str, str]]:
    """通过 Perplexity sonar API 搜索新闻 (可选增强)。

    需要 PERPLEXITY_API_KEY 环境变量。返回空列表表示不可用。
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        import httpx

        proxy = os.getenv("LLM_PROXY", "").strip() or None
        model = os.getenv("PERPLEXITY_MODEL", "sonar")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": query}
            ],
            "max_tokens": 1000,
        }
        with httpx.Client(proxy=proxy, timeout=30) as client:
            resp = client.post(
                "https://api.perplexity.ai/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return [{"title": f"Perplexity搜索结果", "content": content[:1000], "source": "perplexity"}]
    except Exception as e:
        logger.debug("Perplexity search failed: %s", e)
    return []


def analyze_news_sentiment(
    router: Any,
    symbol: str,
    name: str,
    news_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """用LLM分析新闻列表, 提取结构化情绪信息。

    Returns:
        {
            "sentiment_score": -100~100,
            "key_events": ["事件1", "事件2"],
            "risk_alerts": ["风险1"],
            "positive_catalysts": ["利好1"],
            "summary": "一句话总结"
        }
    """
    if not news_items:
        return {
            "sentiment_score": 0,
            "key_events": [],
            "risk_alerts": [],
            "positive_catalysts": [],
            "summary": "无新闻数据",
        }

    # 构建新闻摘要 (控制token)
    news_block = ""
    for i, n in enumerate(news_items[:15]):
        title = str(n.get("title", ""))[:60]
        content = str(n.get("content", ""))[:100]
        time_ = str(n.get("time", ""))
        news_block += f"{i+1}. [{time_}] {title}\n   {content}\n"

    prompt = (
        f"分析以下{symbol} {name}的近期新闻, 给出结构化情绪评估:\n\n"
        f"{news_block}\n"
        f"请输出JSON:\n"
        f'{{"sentiment_score": -100到100的整数(正=利好, 负=利空, 0=中性),'
        f' "key_events": ["最重要的1-3个事件"],'
        f' "risk_alerts": ["风险预警, 0-2个"],'
        f' "positive_catalysts": ["利好催化, 0-2个"],'
        f' "summary": "一句话总结市场情绪"}}'
    )

    try:
        text = router._chat(prompt, multi_turn=True)
        result = _extract_json(text)
        if result and "sentiment_score" in result:
            # 确保类型正确
            result["sentiment_score"] = int(result.get("sentiment_score", 0))
            result["key_events"] = result.get("key_events", [])[:5]
            result["risk_alerts"] = result.get("risk_alerts", [])[:3]
            result["positive_catalysts"] = result.get("positive_catalysts", [])[:3]
            result["summary"] = str(result.get("summary", ""))[:200]
            logger.info("[新闻分析] %s %s: sentiment=%d events=%d",
                        symbol, name, result["sentiment_score"], len(result["key_events"]))
            return result
    except Exception as e:
        logger.warning("[新闻分析] LLM分析失败: %s", e)

    return {
        "sentiment_score": 0,
        "key_events": [],
        "risk_alerts": [],
        "positive_catalysts": [],
        "summary": "新闻分析失败",
    }


def enrich_news_data(
    router: Any | None,
    symbol: str,
    name: str,
    existing_news: list[dict[str, Any]],
    use_perplexity: bool = True,
) -> dict[str, Any]:
    """新闻数据增强: 可选Perplexity搜索 + LLM情绪分析。

    Returns:
        {
            "news_items": [...],     # 合并后的新闻列表
            "sentiment": {...},      # LLM结构化情绪分析
            "perplexity_used": bool,
        }
    """
    all_news = list(existing_news)
    perplexity_used = False

    # 可选: Perplexity 搜索增强
    if use_perplexity:
        query = f"{name} {symbol} 最新消息 利好 利空 公告"
        ppl_results = search_perplexity(query)
        if ppl_results:
            all_news.extend(ppl_results)
            perplexity_used = True
            logger.info("[新闻增强] Perplexity搜索返回 %d 条", len(ppl_results))

    # LLM 情绪分析
    sentiment = {}
    if router and all_news:
        sentiment = analyze_news_sentiment(router, symbol, name, all_news)
    elif all_news:
        # 无LLM时简单统计
        sentiment = {
            "sentiment_score": 0,
            "key_events": [n.get("title", "")[:50] for n in all_news[:3]],
            "risk_alerts": [],
            "positive_catalysts": [],
            "summary": f"共{len(all_news)}条新闻, 未做LLM分析",
        }

    return {
        "news_items": all_news,
        "sentiment": sentiment,
        "perplexity_used": perplexity_used,
    }
