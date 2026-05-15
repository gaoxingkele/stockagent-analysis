"""新闻舆情/政策面热度分析 (V11.2 替代视觉 LLM).

数据源:
  - Tushare cctv_news (央视新闻联播全文, 14 条/日, 政策金矿)
  - AKShare 财经新闻 (扩展, 暂未接入)

LLM 抽取:
  - top_themes: 当日政策主题 top 5 (含热度评分)
  - benefit_sectors: 每个主题对应受益板块 (申万二级)
  - sentiment: 整体市场情绪 ('bullish'/'neutral'/'bearish')

集成到 V12:
  - 输出 daily_policy_heat.json -> WEB 端展示
  - 每股加 policy_benefit 字段 (bool, 是否在今日政策受益板块)
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Optional
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env.cloubic"); load_dotenv(ROOT / ".env")

CLOUBIC_BASE = "https://api.cloubic.com/v1"
LLM_MODEL = "claude-sonnet-4-6"


SYS_PROMPT = """你是 A 股政策面分析师, 专门解读中国政府/中央会议/部委发文的政策导向, 并把政策映射到 A 股可投资板块。

输入: 一组当日的央视新闻联播全文 (政策风向标)。

任务:
1. 抽取本日最重要的 3-5 个政策主题 (优先级: 国务院常务会议 > 国家主席讲话 > 部委发文 > 地方政府)
2. 每个主题判断 A 股受益板块 (申万二级行业层级, 如"半导体"、"光伏设备"、"医疗器械"、"白酒"等)
3. 给每个主题打热度评分 (0-100, 越高越重要)
4. 给出整体市场情绪 (bullish/neutral/bearish)

输出 JSON (不要 markdown 包裹):
{
  "themes": [
    {
      "topic": "30字内主题描述",
      "heat_score": 0-100,
      "benefit_sectors": ["申万二级行业1", "申万二级行业2"],
      "key_phrases": ["关键政策措辞1", "关键政策措辞2"]
    }
  ],
  "market_sentiment": "bullish/neutral/bearish",
  "macro_signal": "30字内宏观信号总结"
}

注意:
- benefit_sectors 必须是申万二级行业 (如"半导体"而不是"科技")
- 没有明确板块映射的政策跳过
- 主题不要超过 5 个, 优先级前 3 个权重最大"""


class NewsSentimentAnalyzer:
    def __init__(self, cloubic_api_key: str):
        self.client = OpenAI(api_key=cloubic_api_key, base_url=CLOUBIC_BASE)

    def fetch_cctv_news(self, date: str) -> list[dict]:
        """date 格式 YYYYMMDD."""
        import tushare as ts
        ts.set_token(os.environ["TUSHARE_TOKEN"])
        pro = ts.pro_api()
        df = pro.cctv_news(date=date)
        if df is None or df.empty:
            return []
        return df.to_dict(orient="records")

    def analyze(self, date: str, save_path: Optional[Path] = None) -> dict:
        """完整流程: 拉 cctv -> LLM -> 输出 dict."""
        news = self.fetch_cctv_news(date)
        if not news:
            return {"date": date, "themes": [], "error": "no_news"}

        # 拼接新闻全文
        news_text = "\n\n---\n\n".join(
            f"【{i+1}. {n.get('title','')}】\n{n.get('content','')[:1500]}"
            for i, n in enumerate(news)
        )
        # LLM 调用
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": f"日期: {date}\n\n央视新闻联播全文:\n\n{news_text}"},
            ],
            max_tokens=2000, temperature=0.1,
        )
        text = resp.choices[0].message.content
        # 解析 JSON
        import re
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m: text = m.group(1)
        else:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m: text = m.group(0)
        try:
            data = json.loads(text)
        except Exception as e:
            return {"date": date, "themes": [], "raw_text": text[:500], "parse_error": str(e)}

        result = {
            "date": date,
            "n_news": len(news),
            "themes": data.get("themes", []),
            "market_sentiment": data.get("market_sentiment", "neutral"),
            "macro_signal": data.get("macro_signal", ""),
            "llm_usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "cost_usd": round((resp.usage.prompt_tokens * 3.0 +
                                    resp.usage.completion_tokens * 15.0) / 1e6, 4),
            },
        }
        # 派生: 受益板块去重 + 加权
        sector_heat: dict[str, float] = {}
        for t in data.get("themes", []):
            heat = float(t.get("heat_score", 0))
            for s in t.get("benefit_sectors", []):
                sector_heat[s] = sector_heat.get(s, 0) + heat
        result["sector_heat"] = sorted(sector_heat.items(), key=lambda x: -x[1])

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
