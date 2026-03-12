# StockAgent 功能扩展计划（第一期）

> 目标：增加消息通知和新闻聚合能力，提升系统的实用性和信息覆盖度
> 原则：最小可用实现，不引入重量级依赖，配置驱动可开关

---

## 执行检查清单

- [ ] Feature 1: 消息通知系统（P7）
- [ ] Feature 2: 新闻聚合增强（P8）

---

## Feature 1: 消息通知系统

**现状**：分析完成后结果仅写入本地文件（`final_decision.json` + PDF），用户需手动查看。无法在分析完成时及时获知结果，尤其不利于批量分析后的快速浏览。

**修改文件**：新建 `src/stockagent_analysis/notification.py`、修改 `src/stockagent_analysis/orchestrator.py`、`configs/project.json`

### 1.1 新建 notification.py

最小实现：企业微信 Webhook + 飞书 Webhook，后续可扩展 Telegram/邮件。

```python
"""分析完成后消息通知（可选）。"""

import json
import logging
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def send_notification(config: dict, result: dict) -> bool:
    """根据配置发送通知。返回是否成功。"""
    noti_cfg = config.get("notification", {})
    if not noti_cfg.get("enabled", False):
        return False

    message = _format_message(result, noti_cfg)
    success = False

    # 企业微信
    wechat_url = noti_cfg.get("wechat_webhook")
    if wechat_url:
        success |= _send_wechat(wechat_url, message)

    # 飞书
    feishu_url = noti_cfg.get("feishu_webhook")
    if feishu_url:
        success |= _send_feishu(feishu_url, message)

    # Telegram
    tg_cfg = noti_cfg.get("telegram", {})
    if tg_cfg.get("bot_token") and tg_cfg.get("chat_id"):
        success |= _send_telegram(tg_cfg["bot_token"], tg_cfg["chat_id"], message)

    return success


def _format_message(result: dict, noti_cfg: dict) -> str:
    """格式化通知消息。"""
    symbol = result.get("symbol", "")
    name = result.get("name", "")
    score = result.get("final_score", 0)
    decision = result.get("final_decision", "")
    decision_level = result.get("decision_level", "")

    # 决策emoji
    emoji_map = {
        "strong_buy": "🔴🔴",
        "weak_buy": "🔴",
        "hold": "⚪",
        "weak_sell": "🟢",
        "strong_sell": "🟢🟢",
    }
    emoji = emoji_map.get(decision_level, "")

    # 狙击点位（如果有）
    sp = result.get("sniper_points", {})
    sp_text = ""
    if sp:
        sp_text = (
            f"\n📌 狙击点位:"
            f"\n  买入: {sp.get('ideal_buy', 'N/A')}"
            f"\n  止损: {sp.get('stop_loss', 'N/A')}"
            f"\n  目标: {sp.get('take_profit_1', 'N/A')}"
        )

    # 持仓建议（如果有）
    pa = result.get("position_advice", {})
    pa_text = ""
    if pa:
        pa_text = (
            f"\n💼 空仓: {pa.get('no_position', '')}"
            f"\n📦 持仓: {pa.get('has_position', '')}"
        )

    # 警告
    warnings = result.get("warnings", "")
    warn_text = f"\n⚠️ {warnings}" if warnings else ""

    msg = (
        f"{emoji} {symbol} {name}\n"
        f"评分: {score:.1f} | 决策: {decision}"
        f"{sp_text}{pa_text}{warn_text}"
    )

    # 过滤阈值：仅推送满足条件的信号
    min_score = noti_cfg.get("min_score_to_notify", 0)
    if score < min_score:
        return ""  # 空消息不发送

    return msg


def _send_wechat(webhook_url: str, message: str) -> bool:
    """发送企业微信 Webhook 消息。"""
    if not message:
        return False
    try:
        payload = json.dumps({
            "msgtype": "text",
            "text": {"content": message}
        }).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning("企业微信通知失败: %s", e)
        return False


def _send_feishu(webhook_url: str, message: str) -> bool:
    """发送飞书 Webhook 消息。"""
    if not message:
        return False
    try:
        payload = json.dumps({
            "msg_type": "text",
            "content": {"text": message}
        }).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning("飞书通知失败: %s", e)
        return False


def _send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    """发送 Telegram Bot 消息。"""
    if not message:
        return False
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning("Telegram通知失败: %s", e)
        return False
```

### 1.2 集成到 orchestrator.py

在 `_run_pipeline` 末尾（PDF 生成、信号记录之后）调用通知：

```python
from .notification import send_notification

# 在 _run_pipeline 末尾
try:
    send_notification(config, final_decision_dict)
except Exception as e:
    logger.warning("通知发送失败（不影响主流程）: %s", e)
```

### 1.3 project.json 配置

新增 `notification` 配置块：

```json
{
  "notification": {
    "enabled": false,
    "min_score_to_notify": 0,
    "wechat_webhook": "",
    "feishu_webhook": "",
    "telegram": {
      "bot_token": "",
      "chat_id": ""
    }
  }
}
```

**配置说明**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | bool | 总开关，默认关闭 |
| `min_score_to_notify` | int | 最低通知分数（如设为 65 则只推送 >=65 分的信号） |
| `wechat_webhook` | str | 企业微信群机器人 Webhook URL |
| `feishu_webhook` | str | 飞书群机器人 Webhook URL |
| `telegram.bot_token` | str | Telegram Bot Token |
| `telegram.chat_id` | str | Telegram Chat ID |

### 1.4 批量通知汇总

批量分析完成后（多只股票），发送汇总消息而非逐条推送：

```python
def send_batch_summary(config: dict, results: list[dict]) -> bool:
    """批量分析完成后发送汇总通知。"""
    noti_cfg = config.get("notification", {})
    if not noti_cfg.get("enabled", False):
        return False

    # 按评分降序排列
    sorted_results = sorted(results, key=lambda r: r.get("final_score", 0), reverse=True)

    lines = ["📊 批量分析完成\n"]
    for r in sorted_results:
        emoji_map = {"strong_buy": "🔴🔴", "weak_buy": "🔴", "hold": "⚪",
                     "weak_sell": "🟢", "strong_sell": "🟢🟢"}
        emoji = emoji_map.get(r.get("decision_level", ""), "")
        lines.append(
            f"{emoji} {r.get('symbol', '')} {r.get('name', '')} "
            f"| {r.get('final_score', 0):.1f}分 | {r.get('final_decision', '')}"
        )

    lines.append(f"\n共 {len(results)} 只，"
                 f"买入 {sum(1 for r in results if 'buy' in r.get('final_decision', ''))} 只")

    message = "\n".join(lines)
    success = False
    if noti_cfg.get("wechat_webhook"):
        success |= _send_wechat(noti_cfg["wechat_webhook"], message)
    if noti_cfg.get("feishu_webhook"):
        success |= _send_feishu(noti_cfg["feishu_webhook"], message)
    tg = noti_cfg.get("telegram", {})
    if tg.get("bot_token") and tg.get("chat_id"):
        success |= _send_telegram(tg["bot_token"], tg["chat_id"], message)
    return success
```

### 验证

1. 配置 `notification.enabled: true` + 有效的 webhook URL
2. 运行一支股票分析
3. 检查 webhook 收到消息，格式正确
4. `notification.enabled: false` 时不发送任何请求
5. webhook URL 无效时不报错（graceful fallback + warning 日志）
6. 批量分析后收到汇总消息

---

## Feature 2: 新闻聚合增强

**现状**：`data_backend.py` 中仅有简单的 `news_sentiment_score` 数值（来自 AKShare 新闻情绪），缺乏多源搜索、结构化风险/利好提取能力。新闻信息对 SENTIMENT、NLP_SENTIMENT、SECTOR_POLICY 等维度影响较大。

**修改文件**：新建 `src/stockagent_analysis/search_service.py`、修改 `src/stockagent_analysis/data_backend.py`

### 2.1 新建 search_service.py

多源搜索服务，支持 Tavily/SerpAPI/Perplexity，按可用性自动切换：

```python
"""多源新闻搜索与结构化提取。"""

import json
import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class SearchService:
    """新闻搜索服务 — 多源自动切换。"""

    def __init__(self):
        self._tavily_key = os.getenv("TAVILY_API_KEY", "")
        self._serp_key = os.getenv("SERPAPI_KEY", "")
        self._perplexity_key = os.getenv("PERPLEXITY_API_KEY", "")

    def search_stock_news(self, symbol: str, name: str,
                          max_results: int = 10) -> list[dict]:
        """搜索股票相关新闻，返回结构化结果列表。

        每条结果: {"title", "snippet", "url", "date", "source"}
        """
        query = f"{name} {symbol} 股票 最新消息"

        # 按优先级尝试各搜索源
        if self._tavily_key:
            results = self._search_tavily(query, max_results)
            if results:
                return results

        if self._serp_key:
            results = self._search_serpapi(query, max_results)
            if results:
                return results

        # 无API key时返回空（不报错）
        logger.info("无可用搜索API key，跳过新闻搜索增强")
        return []

    def extract_risk_and_catalyst(self, news_items: list[dict],
                                   name: str) -> dict:
        """从新闻列表中提取风险因素和利好因素。

        简单关键词匹配 + 分类统计（不依赖LLM）。
        """
        risk_keywords = [
            "减持", "质押", "亏损", "下滑", "处罚", "违规", "退市",
            "诉讼", "调查", "暴跌", "爆雷", "商誉减值", "业绩预亏",
            "停牌", "ST", "高管辞职", "大股东减持",
        ]
        catalyst_keywords = [
            "增持", "回购", "业绩预增", "中标", "合同", "政策利好",
            "涨停", "突破", "新高", "龙头", "国产替代", "订单",
            "扩产", "战略合作", "并购", "分红", "送转",
        ]

        risks = []
        catalysts = []
        for item in news_items:
            text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
            for kw in risk_keywords:
                if kw in text:
                    risks.append({"keyword": kw, "title": item.get("title", ""),
                                  "date": item.get("date", "")})
                    break
            for kw in catalyst_keywords:
                if kw in text:
                    catalysts.append({"keyword": kw, "title": item.get("title", ""),
                                      "date": item.get("date", "")})
                    break

        return {
            "risk_count": len(risks),
            "catalyst_count": len(catalysts),
            "risks": risks[:5],        # 最多保留5条
            "catalysts": catalysts[:5],
            "sentiment_bias": "positive" if len(catalysts) > len(risks) + 2
                              else "negative" if len(risks) > len(catalysts) + 2
                              else "neutral",
        }

    def _search_tavily(self, query: str, max_results: int) -> list[dict]:
        """Tavily Search API。"""
        try:
            payload = json.dumps({
                "api_key": self._tavily_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_domains": [
                    "eastmoney.com", "10jqka.com.cn", "sina.com.cn",
                    "163.com", "cls.cn", "caixin.com",
                ],
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("content", "")[:200],
                    "url": r.get("url", ""),
                    "date": r.get("published_date", ""),
                    "source": "tavily",
                }
                for r in data.get("results", [])
            ]
        except Exception as e:
            logger.warning("Tavily搜索失败: %s", e)
            return []

    def _search_serpapi(self, query: str, max_results: int) -> list[dict]:
        """SerpAPI Google Search。"""
        try:
            from urllib.parse import urlencode
            params = urlencode({
                "q": query,
                "api_key": self._serp_key,
                "engine": "google",
                "gl": "cn",
                "hl": "zh-cn",
                "num": max_results,
                "tbm": "nws",  # 新闻搜索
            })
            url = f"https://serpapi.com/search?{params}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", "")[:200],
                    "url": r.get("link", ""),
                    "date": r.get("date", ""),
                    "source": "serpapi",
                }
                for r in data.get("news_results", [])
            ]
        except Exception as e:
            logger.warning("SerpAPI搜索失败: %s", e)
            return []
```

### 2.2 集成到 data_backend.py

在 `collect_and_save_context()` 中，现有新闻采集之后追加搜索增强：

```python
from .search_service import SearchService

# 在 collect_and_save_context() 中，news_sentiment 计算之后
try:
    search_svc = SearchService()
    news_items = search_svc.search_stock_news(symbol, name, max_results=10)
    if news_items:
        features["enhanced_news"] = {
            "items": news_items[:10],
            "count": len(news_items),
        }
        # 提取风险/利好
        risk_catalyst = search_svc.extract_risk_and_catalyst(news_items, name)
        features["news_risk_catalyst"] = risk_catalyst

        # 更新 news_sentiment（叠加搜索结果的情绪）
        bias = risk_catalyst.get("sentiment_bias", "neutral")
        if bias == "positive":
            features["news_sentiment"] = min(10, features.get("news_sentiment", 0) + 3)
        elif bias == "negative":
            features["news_sentiment"] = max(-10, features.get("news_sentiment", 0) - 3)
except Exception as e:
    logger.warning("新闻搜索增强失败（不影响主流程）: %s", e)
```

### 2.3 注入 Agent 数据上下文

在 `agents.py` 的 `_build_data_context()` 中：

```python
# 新闻风险/利好
rc = ctx.get("features", {}).get("news_risk_catalyst", {})
if rc:
    risk_titles = [r["title"] for r in rc.get("risks", [])[:3]]
    cat_titles = [c["title"] for c in rc.get("catalysts", [])[:3]]
    if risk_titles:
        parts.append("⚠️ 风险新闻: " + " | ".join(risk_titles))
    if cat_titles:
        parts.append("✅ 利好新闻: " + " | ".join(cat_titles))
    parts.append(f"新闻情绪: 风险{rc.get('risk_count', 0)}条 vs 利好{rc.get('catalyst_count', 0)}条")
```

### 2.4 环境变量配置

无需修改 `project.json`。搜索服务通过环境变量自动发现：

```bash
# .env 中可选配置（有key则启用，无则跳过）
TAVILY_API_KEY=tvly-xxxxxx      # Tavily Search (优先)
SERPAPI_KEY=xxxxxx               # SerpAPI Google Search (备选)
```

### 验证

1. 配置 `TAVILY_API_KEY` 或 `SERPAPI_KEY` 环境变量
2. 运行一支股票分析
3. 检查 `analysis_context.json` 中出现 `enhanced_news` 和 `news_risk_catalyst` 字段
4. `news_risk_catalyst` 包含 `risks`/`catalysts` 列表和 `sentiment_bias`
5. Agent LLM 上下文中出现风险/利好新闻标题
6. 无 API key 时不报错，优雅跳过
7. `news_sentiment` 数值因搜索结果调整

---

## 断点续做说明

| Feature | 依赖 | 可独立执行 |
|---------|------|-----------|
| Feature 1 (消息通知) | 无（可选依赖 Step 2/3 的 sniper_points/position_advice） | 是 |
| Feature 2 (新闻增强) | 无 | 是 |

**续做方式**：检查上方清单中的 `[x]` 标记，从第一个未完成的 Feature 继续。

---

## 不做的事（避免过度工程）

1. **不做9通道全覆盖** — 最小实现企业微信+飞书+Telegram，够用即可
2. **不做 LLM 新闻摘要** — 关键词匹配足够，LLM摘要增加调用成本和延迟
3. **不做新闻实时监控** — 系统定位是"分析时采集"，非"持续监控"
4. **不做自定义搜索域** — 硬编码财经网站域名列表，不暴露配置复杂度
5. **不做新闻去重/聚类** — 10条新闻量级无需复杂处理
