# stockagent-analysis 改进计划

> 生成日期: 2026-03-16
> 依据: docs/technical-analysis-and-improvement.md
> 参考项目: daily_stock_analysis, TradingAgents-CN

---

## 基础改进

### 1. Agent精简与重构 (29 → 10~12)

**问题:** 14个基础维度Agent的_simple_policy公式高度同质化(都是`50 + a*pct + b*mom + c*vol + d*vr + e*news`系数变体), SECTOR_POLICY与INDUSTRY公式完全相同, BETA/MACRO/QUANT差异极小。29个"独立"Agent本质是对同一信号的冗余投票。

**改进内容:**

1.1 合并同质化Agent:
- TREND(0.18) + BETA(0.04) + TRENDLINE(0.06) → **TREND_MOMENTUM** (0.18)
- TECH(0.10) + QUANT(0.06) → **TECH_QUANT** (0.12)
- CAPITAL_FLOW(0.15) + FLOW_DETAIL(0.05) + LIQ(0.06) → **CAPITAL_LIQUIDITY** (0.15)
- SECTOR_POLICY(0.06) + INDUSTRY(0.06) + MACRO(0.04) → **SECTOR_MACRO** (0.08)
- SENTIMENT(0.06) + NLP_SENTIMENT(0.04) + MM_BEHAVIOR(0.06) → **SENTIMENT_FLOW** (0.10)
- KLINE_PATTERN(0.10) + CHART_PATTERN(0.08) + TOP/BOTTOM(0.08) → **PATTERN** (0.10)
- 4个KLINE视觉Agent(0.19) → **KLINE_VISION** (0.10) 内部处理多周期
- DIVERGENCE(0.10) — 保留
- CHANLUN(0.10) — 保留
- VOLUME_PRICE(0.08) + SUPPORT_RESISTANCE(0.08) → **VOLUME_STRUCTURE** (0.08)
- TIMEFRAME_RESONANCE(0.08) — 保留, 作为跨周期元判断
- FUNDAMENTAL(0.12) — 保留
- DERIV_MARGIN(0.12) — 保留或并入CAPITAL_LIQUIDITY

1.2 为每个合并后Agent设计真正差异化的评分逻辑, 引入:
- MA排列状态+ADX替代简单`trend`变量
- 历史分位数概念(如换手率vs近60日分位)替代绝对值
- PEG估值(PE/增速)替代单一PE分级
- 条件分支+非线性逻辑替代纯线性公式

1.3 更新configs/agents/目录, 删除合并后废弃的JSON配置

**收益:** LLM调用从29次降到10~12次/Provider, 成本降65%; 消除虚假多样性

---

### 2. 评分公式差异化

**问题:** 合并后仍需确保每个Agent的量化评分逻辑真正独立, 而非换个名字的线性组合。

**改进内容:**

2.1 **TREND_MOMENTUM** — 多指标趋势综合:
- MA排列打分(全多头/空头/交叉各状态离散分)
- ADX趋势强度(>25强趋势, <20无趋势)
- 趋势线斜率+突破确认
- 乖离率惩罚(保留现有封顶逻辑)

2.2 **CAPITAL_LIQUIDITY** — 量价确认体系:
- OBV趋势 vs 价格趋势一致性(量价齐升/背离)
- 换手率历史分位数(vs近60日)
- 量比连续趋势(连续3日放量/缩量)
- 大单净流入(如有数据源)

2.3 **FUNDAMENTAL** — 多维估值模型:
- PEG估值(PE / 净利增速), 替代单一PE分级
- 行业PE中位数对比(相对估值)
- 简化F-Score(盈利+杠杆+效率3维)
- 自由现金流趋势

2.4 **SENTIMENT_FLOW** — 情绪独立化:
- 拆分news_c为多个子信号(政策/行业/个股/舆情)
- 量比异动作为主力行为独立信号
- 龙虎榜/大宗交易数据(如有)

2.5 **PATTERN** — 统一形态评分:
- K线组合形态 + 图表形态 + 顶底结构合并为单一形态评分
- 多周期加权(日0.50 + 周0.35 + 月0.15)
- 顶部结构自动反转(无需外部INVERT_DIMS)

**收益:** 各Agent评分真正独立, 加权平均有意义

---

### 3. 结构化辩论机制

**问题:** 当前辩论默认禁用, 29个Agent简单加权平均缺乏对抗推理。TradingAgents-CN的"Bull/Bear多轮辩论→Manager仲裁→风险三方辩论"是其核心竞争力。

**改进内容:**

3.1 将Agent按职能分为3个团队, 各团队内部先汇总:
- 技术面团队: TREND_MOMENTUM + TECH_QUANT + DIVERGENCE + CHANLUN + PATTERN + VOLUME_STRUCTURE + RESONANCE
- 资金面团队: CAPITAL_LIQUIDITY + SENTIMENT_FLOW + KLINE_VISION
- 基本面团队: FUNDAMENTAL + SECTOR_MACRO + DERIV_MARGIN

3.2 团队汇报阶段(3次LLM调用):
- 每个团队汇总内部Agent的本地评分+数据上下文
- LLM生成团队报告: "从{X}面看, 该股..."

3.3 投资辩论阶段(4次LLM调用):
- 看多方: 基于3份团队报告中的利多因素立论
- 看空方: 针对看多论点反驳, 提出利空证据
- 看多方反驳
- 看空方反驳

3.4 仲裁阶段(1次LLM调用):
- 综合辩论双方论点 + 原始数据报告
- 输出: 投资计划(买入/持有/卖出) + 目标价 + 置信度 + 核心理由

3.5 风险评估阶段(3次LLM调用):
- 激进派: 基于投资计划给出激进建议
- 保守派: 基于投资计划给出保守建议
- 风险经理: 仲裁, 输出最终决策 + 风险评分

3.6 用原生Python实现(不引入LangGraph), 保持技术栈一致

**收益:** 决策从"加权平均"升级为"对抗推理", 质量质变; 总LLM调用约42次(vs当前87次)

---

### 4. LLM快慢分层

**问题:** 当前所有LLM调用用同一模型, 数据整理和深度推理不分层。TradingAgents-CN用quick_think(gpt-4o-mini)做收集, deep_think(o4-mini)做决策。

**改进内容:**

4.1 在project.json中新增配置:
```json
"llm": {
    "quick_model": "deepseek-chat",
    "deep_model": "grok-4-1-fast-reasoning"
}
```

4.2 调用分层:
- **快速模型**: Agent的enrich_and_score(数据整理+初步评分)、团队报告汇总
- **深度模型**: 辩论环节(Bull/Bear)、仲裁决策、风险评估、场景生成

4.3 LLMRouter增加model_tier参数, 按tier选择模型

**收益:** 深度推理用强模型保证质量, 常规调用用快模型降低成本和延迟

---

### 5. 新闻/舆情数据增强

**问题:** 5个维度共享同一个预计算`news_c`变量, 信息量极低。daily_stock_analysis用4个搜索引擎, TradingAgents-CN有专门的News Analyst。

**改进内容:**

5.1 新增NewsSearchService, 集成1~2个搜索引擎:
- 优先: Perplexity sonar(已有API key)
- 备选: 财经网站爬取(东方财富/同花顺公告)

5.2 新闻采集流程:
- 在data_backend.collect_and_save_context()中新增新闻采集步骤
- 搜索 "{股票名} {代码} 最新消息" → 结构化提取
- 输出: 新闻列表 + 情绪分 + 关键事件 + 风险预警

5.3 SENTIMENT_FLOW Agent独立消费新闻数据, 不再依赖全局news_c

5.4 新闻数据写入analysis_context.json的`news`字段, 供辩论环节引用

**收益:** 信息维度真正独立, 新闻驱动的Agent有独立数据源

---

## 扩展改进 (未经指示不执行)

### E1. 推送通道

多渠道通知推送, 将分析结果自动发送到用户终端。

- E1.1 企业微信Webhook推送(Markdown格式)
- E1.2 飞书Webhook推送(Rich Card格式)
- E1.3 Telegram Bot推送
- E1.4 Email推送(HTML格式)
- E1.5 Web Dashboard(FastAPI + Vue前端)
- E1.6 股票分组→不同接收人路由

参考实现: daily_stock_analysis/src/notification.py

---

### E2. 回测闭环与动态权重

历史预测验证 + 基于表现的Agent权重自动调整。

- E2.1 BacktestEngine: 对比N日后实际走势 vs 预测(方向准确率/目标价命中/止损触发)
- E2.2 Agent级别历史准确率评估(win_rate, avg_return, sharpe)
- E2.3 动态权重: 胜率>60%的Agent权重×1.3, <40%的×0.7, 归一化
- E2.4 记忆系统: 持久化历史预测+结果, 供反思学习
- E2.5 定期报告: Agent准确率排行, 识别失效维度

参考实现: daily_stock_analysis/src/core/backtest_engine.py, TradingAgents-CN/agents/utils/memory.py
