# stockagent-analysis 技术分析与改进方案

> 生成日期: 2026-03-16
> 参考项目: daily_stock_analysis, TradingAgents-CN

---

## 一、当前系统架构概览

### 1.1 整体流程

```
CLI入口 (run.py)
  → main.py (参数解析)
    → orchestrator.run_analysis()
      ├── Phase 1: 数据采集 (data_backend.py)
      │   ├── TDX本地 → Tushare → AKShare (三级降级)
      │   ├── 多周期K线计算 (日/周/月, 周月由日线resample)
      │   ├── 技术指标计算 (MA/MACD/RSI/KDJ/布林带)
      │   ├── 高级形态检测 (背离/量价/缠论/图形/支撑阻力)
      │   └── K线图表生成 (chart_generator.py, 4个周期PNG)
      │
      ├── Phase 2: Agent本地分析 (agents.py)
      │   └── 29个Agent各自 analyze_local() → _simple_policy() 产出本地评分
      │
      ├── Phase 3: 多Provider并行LLM研判 (parallel_runner.py)
      │   ├── 每个Provider: ThreadPoolExecutor(max_workers=6)
      │   ├── 每个Agent: enrich_and_score() 单次LLM调用
      │   ├── 视觉Agent: chat_with_image() + score 两步调用
      │   └── 重试链: 主Provider → 重试1次 → 候选Provider×4
      │
      ├── Phase 4: 辩论 (可选, 多模型模式下一般禁用)
      │
      └── Phase 5: 加权汇总 → 最终决策 → PDF报告
```

### 1.2 核心模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| **CLI入口** | `run.py` → `main.py` | 参数解析, 调用orchestrator |
| **编排器** | `orchestrator.py` | 全流程编排: 数据→Agent→LLM→汇总→报告 |
| **Agent体系** | `agents.py` | 29个Agent类, _simple_policy评分公式, 本地分析 |
| **并行执行** | `parallel_runner.py` | Provider并行, Agent并发, 重试/降级 |
| **LLM客户端** | `llm_client.py` | enrich_and_score合并调用, 视觉研判, 权重配置 |
| **LLM路由** | `core/router.py` | 11个Provider抽象, 代理路由, 视觉检测 |
| **执行工具** | `core/runner.py` | _timed_call超时控制, 断点续传, 进度追踪 |
| **进度渲染** | `core/progress.py` | 终端表格实时刷新, CJK宽度处理 |
| **数据后端** | `data_backend.py` | 多源数据采集, 指标计算, 形态检测 |
| **图表生成** | `chart_generator.py` | 4周期K线图PNG, 叠加指标和形态标注 |
| **PDF报告** | `report_pdf.py` | 投资者级别PDF输出, 9维度评分表等 |

---

## 二、智能体(Agent)逻辑详解

### 2.1 Agent分类与权重

系统共29个Agent, 分为以下类别:

#### 基础维度Agent (14个, 基类AnalystAgent)

| Agent | dim_code | 权重 | 核心功能 |
|-------|----------|------|----------|
| trend_agent | TREND | 0.18 | 趋势强度+动量+乖离率封顶 |
| tech_agent | TECH | 0.10 | 技术指标综合(动量+趋势+量比+波动) |
| liq_agent | LIQ | 0.06 | 流动性评估(量比+换手+波动) |
| capital_flow_agent | CAPITAL_FLOW | 0.15 | 资金流向(涨跌+量比+新闻+动量) |
| sector_policy_agent | SECTOR_POLICY | 0.06 | 行业政策(新闻情绪+动量) |
| beta_agent | BETA | 0.04 | 贝塔系数(趋势+涨跌-波动) |
| sentiment_agent | SENTIMENT | 0.06 | 市场情绪(新闻+涨跌) |
| fundamental_agent | FUNDAMENTAL | 0.12 | 基本面(PE/PB/ROE/负债率/成长性+筹码) |
| quant_agent | QUANT | 0.06 | 量化因子(动量-波动+量比) |
| macro_agent | MACRO | 0.04 | 宏观环境(动量-波动+新闻) |
| industry_agent | INDUSTRY | 0.06 | 行业分析(新闻+动量) |
| flow_detail_agent | FLOW_DETAIL | 0.05 | 资金细节(量比+涨跌) |
| mm_behavior_agent | MM_BEHAVIOR | 0.06 | 主力行为(量比+新闻-波动) |
| nlp_sentiment_agent | NLP_SENTIMENT | 0.04 | NLP情绪(新闻×1.2+动量) |

#### 融资融券Agent (1个)

| Agent | dim_code | 权重 | 核心功能 |
|-------|----------|------|----------|
| deriv_margin_agent | DERIV_MARGIN | 0.12 | 融资融券+沪深港通+涨跌-波动 |

#### K线视觉Agent (4个, KlineVisionAgent)

| Agent | dim_code | 权重 | 核心功能 |
|-------|----------|------|----------|
| kline_1h_agent | KLINE_1H | 0.04 | 1小时K线图像识别 |
| kline_day_agent | KLINE_DAY | 0.06 | 日K线图像识别 |
| kline_week_agent | KLINE_WEEK | 0.05 | 周K线图像识别 |
| kline_month_agent | KLINE_MONTH | 0.04 | 月K线图像识别 |

#### 技术形态Agent (7个, 各自专属子类)

| Agent | dim_code | 权重 | agent_type | 核心功能 |
|-------|----------|------|------------|----------|
| kline_pattern_agent | KLINE_PATTERN | 0.10 | kline_pattern | 15+种K线组合形态 |
| divergence_agent | DIVERGENCE | 0.10 | divergence | MACD/RSI顶底背离 |
| volume_price_agent | VOLUME_PRICE | 0.08 | volume_price | 量价信号(放量突破/缩量回踩等) |
| support_resistance_agent | SUPPORT_RESISTANCE | 0.08 | support_resistance | 支撑阻力位检测 |
| chanlun_agent | CHANLUN | 0.10 | chanlun | 缠论分型→笔→中枢→买卖点 |
| chart_pattern_agent | CHART_PATTERN | 0.08 | chart_pattern | 三角形/箱体/旗形/杯柄/圆弧 |
| timeframe_resonance_agent | TIMEFRAME_RESONANCE | 0.08 | timeframe_resonance | 日/周/月多周期共振检测 |

#### 趋势线与相对强弱Agent (2个)

| Agent | dim_code | 权重 | 核心功能 |
|-------|----------|------|----------|
| trendline_agent | TRENDLINE | 0.06 | trendline | 趋势线斜率+突破信号 |
| relative_strength_agent | RELATIVE_STRENGTH | 0.05 | relative_strength | vs市场/行业/龙头/ETF超额收益 |

#### 顶底结构Agent (2个)

| Agent | dim_code | 权重 | 核心功能 |
|-------|----------|------|----------|
| top_structure_agent | TOP_STRUCTURE | 0.04 | analyst | 顶部信号(长上影+负动量) |
| bottom_structure_agent | BOTTOM_STRUCTURE | 0.04 | analyst | 底部信号(长下影+正动量) |

> **注意**: TOP_STRUCTURE 属于反向维度(`_INVERT_DIMS`), 高分=卖出警示, 最终加权时 `100 - score`。

### 2.2 _simple_policy 评分公式体系

所有评分以 **50分为中性基准**, 通过因子加减偏移:

#### 输入变量统一定义

```python
pct    = 涨跌幅%           # 当日或近期涨跌
mom    = 20日动量%          # 20日价格变化率
vol    = 20日波动%          # 20日历史波动率
dd     = 60日最大回撤%      # 近60日最大回撤
vr     = 量比(5日/20日均)   # 短期成交量/中期成交量
trend  = 趋势强度           # MA排列/斜率等综合趋势
news   = 新闻情绪(-100~100) # NLP情绪得分
pe     = 市盈率TTM          # 估值
pb     = 市净率             # 估值
turn   = 换手率%            # 流动性
```

#### 核心维度公式

**TREND (趋势, 权重0.18 — 最高权重)**
```
base = 50 + 1.0*mom + 0.8*trend + 0.3*pct + 0.12*dd
乖离惩罚:
  MA5乖离 > 8% → penalty = -25, 且 TREND 封顶35分
  MA5乖离 5~8% → penalty = -15
  MA5乖离 3~5% → penalty = -5
TREND = base + penalty
```
> 设计意图: 趋势是王, 但高乖离=追涨风险, 必须硬性压制。

**CAPITAL_FLOW (资金流向, 权重0.15)**
```
CAPITAL_FLOW = 50 + 0.5*pct + (vr-1)*16 + news*0.5 + 0.35*mom
```
> 量比偏移×16是最大的单因子, 放量=资金进场的直接证据。

**FUNDAMENTAL (基本面, 权重0.12)**
```
PE因子:  <15→+12, 15-25→+5, 40-60→-4, 60-100→-8, >100→-10, 亏损→-10
PB因子:  <1→+8, 1-2→+4, 8-15→-3, 15-50→-5, >50→-8
成长性:  营收>30%且净利>30%→+10, 营收>15%且净利>15%→+5, 差→-8
负债率:  >80%→-6, 60-80%→-3, <30%→+3
ROE:     >20%→+8, 10-20%→+4, <0→-6
筹码:    集中度因子(从kline_indicators)

FUNDAMENTAL = 50 + PE + PB + 成长 + 负债 + ROE + 0.15*mom + 筹码
```

**DIVERGENCE (背离, 权重0.10)**
```
多周期加权: 日0.50 + 周0.40 + 月0.10
每周期: MACD/RSI顶底背离检测 → divergence_score
DIVERGENCE = 50 + weighted_divergence_score * 0.8
```

**CHANLUN (缠论, 权重0.10)**
```
检测流程: K线包含处理 → 顶底分型 → 笔构造(间隔≥4) → 中枢(≥3笔重叠) → 三类买卖点
多周期加权: 日0.50 + 周0.40 + 月0.10
CHANLUN = 50 + weighted_chanlun_score
```

**TIMEFRAME_RESONANCE (多周期共振, 权重0.08)**
```
3周期都看涨 → 80
2周期看涨   → 68
3周期都看跌 → 20
2周期看跌   → 32
多空分歧    → 45
```
> 这是唯一不用连续公式的维度, 直接枚举离散状态。

#### 评分后处理

```python
# 1. 乖离率硬封顶
if MA5_bias > 8%:
    TREND = min(TREND, 35)

# 2. 所有维度归一化 [0, 100]
score = clamp(raw_score, 0, 100)

# 3. 置信度 = 数据质量驱动
confidence = clamp(0.45 + 0.45 * data_quality, 0.1, 0.98)

# 4. 投票
vote = "buy" if score >= 70 else "sell" if score < 50 else "hold"
```

### 2.3 LLM研判与评分合并流程

```python
# enrich_and_score() — 1次LLM调用完成研判+评分
prompt = f"""
你是中国股市{role}分析员。
股票: {symbol} {name}
本地结论: {base_reason}
【本地已获取数据】{data_context}
【评分标准校准】
  80-100: 强烈看多，上涨概率>75%
  60-79:  偏多，上涨概率55-75%
  40-59:  中性/不确定
  20-39:  偏空
  0-19:   强烈看空
请输出JSON: {{"analysis":"...", "score":68, "risk":"..."}}
"""
# 解析: JSON优先, 正则回退
```

### 2.4 最终决策汇总

```
对每个Provider p:
  对每个Agent a:
    llm_weight = base_weight(0.35) × consensus_factor
    local_weight = 1.0 - llm_weight
    agent_final_score = local_score × local_weight + llm_score × llm_weight
    if dim_code in INVERT_DIMS: agent_final_score = 100 - agent_final_score
    provider_total += agent_final_score × agent_weight

final_score = avg(所有Provider的provider_total)

# 动态阈值 (按市场状态)
牛市: buy≥65, sell<45
熊市: buy≥78, sell<55
震荡: buy≥70, sell<50

# 五级决策
≥85  → 强烈买入
70~85 → 弱买入
50~70 → 观望
40~50 → 弱卖出
<40   → 强烈卖出

# 乖离率过滤
if MA5乖离>5% and 决策为买入:
  → 降级为"观望" + 乖离率警告
```

### 2.5 consensus_factor (共识因子)

```python
# 跨Provider对同一Agent的评分一致性
scores = [provider_scores[p][agent_id] for p in providers]
sigma = std(scores)

# 高一致性(σ小) → 放大LLM权重; 低一致性(σ大) → 压缩LLM权重
consensus_factor = clamp(1.0 - (sigma - 3) / 15, 0.5, 1.2)
llm_weight = clamp(0.35 × consensus_factor, 0.15, 0.50)
```

> 设计意图: 多个LLM意见一致时, 更信任LLM; 分歧大时, 更信任本地量化评分。

---

## 三、参考项目对比分析

### 3.1 daily_stock_analysis

| 维度 | daily_stock_analysis | 本项目 (stockagent-analysis) |
|------|---------------------|----------------------------|
| **架构** | 单一全能LLM分析器 + 可选ReAct Agent | 29个专业Agent并行分析 |
| **LLM角色** | LLM做全部分析(主力) | LLM做研判增强(辅助), 本地量化为主 |
| **数据源** | 6个价格源 + 新闻搜索(Tavily/SerpAPI) | TDX本地 + Tushare + AKShare |
| **策略定义** | YAML配置(11个内置策略) | Python类硬编码(_simple_policy) |
| **输出** | JSON Dashboard → 多渠道推送(8+) | PDF报告 + JSON |
| **Web UI** | FastAPI + Vue3 完整前端 | 无 |
| **通知推送** | 微信/飞书/Telegram/Email/Discord | 无 |
| **回测** | AI回放引擎(历史验证) | 无 |
| **市场体系** | 进攻/均衡/防守三体系 | 牛/熊/震荡动态阈值 |
| **趋势纪律** | MA5>MA10>MA20硬性过滤, 乖离>5%禁入 | 乖离>8%封顶35分 |
| **Bot支持** | Discord/Telegram/飞书/钉钉 | 无 |

**值得借鉴的特性:**
1. **YAML策略配置** — 策略参数化, 无需改代码即可新增策略
2. **多渠道推送** — 分析结果自动推送到用户常用平台
3. **AI回测引擎** — 对历史预测做准确率评估, 形成反馈闭环
4. **ReAct Agent** — 多步推理+工具调用, 而非一次性prompt
5. **决策仪表盘格式** — 一句话结论+狙击点位+行动清单, 比PDF更轻量
6. **股票分组推送** — 不同股票组→不同接收人, 适合团队使用

### 3.2 TradingAgents-CN

| 维度 | TradingAgents-CN | 本项目 (stockagent-analysis) |
|------|-----------------|----------------------------|
| **架构** | LangGraph状态机驱动, 4个团队串行 | ThreadPoolExecutor并行, 扁平Agent |
| **Agent协作** | 结构化辩论(多轮Bull vs Bear + 风险三方辩论) | 简单辩论(可选, 一般禁用) |
| **LLM角色** | LLM做全部决策(纯LLM驱动) | LLM做研判增强(本地量化+LLM辅助) |
| **评分体系** | Buy/Hold/Sell + Confidence(0-1) + Risk(0-1) | 0-100连续评分 + 五级决策 |
| **风险管理** | 三层辩论(激进/保守/中立) + 历史记忆 | 乖离率过滤 + 市场状态阈值调整 |
| **数据源** | Tushare/AkShare/BaoStock + Finnhub + Alpha Vantage | TDX + Tushare + AKShare |
| **新闻** | Google News + Reddit + 中文财经 | 新闻情绪分(预计算) |
| **记忆系统** | MongoDB持久化, 反思+学习 | 无 |
| **LLM分层** | quick_thinking(快) + deep_thinking(慢)双模型 | 单一模型, 多Provider并行 |
| **缓存** | MongoDB + Redis多级缓存 | 文件级断点续传 |
| **技术栈** | LangGraph + LangChain + MongoDB + Redis | 原生Python + ThreadPoolExecutor |

**值得借鉴的特性:**
1. **结构化辩论机制** — Bull/Bear多轮辩论→Research Manager仲裁→投资计划→风险三方辩论→最终决策, 比简单加权更有逻辑
2. **风险三方辩论** — 激进/保守/中立三个视角, 避免单一偏见
3. **LLM分层** — 快速LLM做数据收集, 深度LLM做决策合成, 兼顾成本和质量
4. **记忆与反思** — reflect_and_remember()从历史P&L学习, 避免重复错误
5. **状态图编排** — LangGraph的条件路由比ThreadPool更灵活
6. **目标价必须给出** — Trader prompt强制要求target_price, 而非模糊的买入/卖出

---

## 四、问题诊断与改进方向

### 4.1 当前系统核心问题

#### P0: LLM利用效率低, 本地评分同质化严重

**问题描述:**
- 29个Agent中, 14个基础维度Agent的_simple_policy公式高度相似 — 都是`50 + a*pct + b*mom + c*vol + d*vr + e*news`的线性组合, 只是系数不同
- 例如: SECTOR_POLICY和INDUSTRY公式完全相同(`50 + news*0.8 + 0.3*mom`), 但分配了不同的权重
- BETA/MACRO/QUANT三个维度的公式差异极小, 本质上是噪声
- **后果**: 29个"独立"Agent实际上在量化层面高度相关, 加权平均后相当于对同一信号做了冗余投票

**对比:**
- TradingAgents-CN: 每个Agent有完全不同的数据源和分析角度(技术/基本面/新闻/社交), 真正的多维度
- daily_stock_analysis: 单一全能LLM, 但prompt中明确要求从数据/情报/作战三个维度独立分析

#### P1: 辩论机制形同虚设

**问题描述:**
- 多模型模式下辩论默认禁用(`debate_rounds=0`)
- 即使启用, 也只是最高分vs最低分两个Agent对话, 无Judge仲裁机制
- **对比**: TradingAgents-CN的辩论是核心设计 — Bull/Bear多轮→Manager仲裁→Trader→Risk三方辩论→最终决策, 形成真正的对抗性推理

#### P2: 无反馈闭环

**问题描述:**
- 系统产出分析报告后, 无法验证预测准确性
- 无回测机制, 无法评估哪些Agent/维度在历史上表现好
- 无记忆系统, 每次分析都是"从零开始"
- **对比**:
  - daily_stock_analysis: AI回测引擎, 对历史预测做准确率评估
  - TradingAgents-CN: reflect_and_remember(), 从P&L学习调整策略

#### P3: 新闻/舆情数据薄弱

**问题描述:**
- `news_c`(新闻情绪分)是预计算值, 来源单一
- 无实时新闻搜索能力
- 5个维度(SENTIMENT, NLP_SENTIMENT, SECTOR_POLICY, INDUSTRY, MM_BEHAVIOR)都依赖同一个`news_c`变量
- **对比**:
  - daily_stock_analysis: Tavily/SerpAPI/Bocha/Brave四引擎新闻搜索
  - TradingAgents-CN: Google News + Reddit + 中文财经, 专门的Social Media Analyst

#### P4: 缺少实用输出通道

**问题描述:**
- 仅输出PDF和JSON, 无推送能力
- 无Web UI, 无Bot交互
- 无法满足"每日定时分析→推送到手机"的典型使用场景
- **对比**: daily_stock_analysis支持8+渠道推送, 有完整Web UI

### 4.2 次要问题

| 编号 | 问题 | 影响 |
|------|------|------|
| P5 | 权重硬编码在JSON配置中, 无法根据市场状态动态调整 | 牛市/熊市下各维度重要性应不同 |
| P6 | 视觉Agent降级为文本时, 使用的是TECH/TREND的公式, 信息损失大 | 无视觉Provider时评分质量下降 |
| P7 | 缠论实现简化(笔间隔≥4, 中枢≥3笔), 缺少线段和多级别递归 | 缠论信号准确度有限 |
| P8 | 无实时盘中分析能力, 仅支持盘后 | 无法做盘中预警 |
| P9 | 评分校准缺乏锚定 — LLM被要求打0-100分但无ground truth | 不同LLM的分数分布可能差异大 |

---

## 五、改进方案

### 方案A: Agent精简与重构 — "少而精"

**目标:** 将29个Agent精简为10~12个真正独立的分析维度

#### A.1 合并同质化Agent

```
当前 → 合并后

TREND(0.18) + BETA(0.04) + TRENDLINE(0.06)
  → TREND_MOMENTUM (0.20) — 趋势+动量+趋势线一体化

TECH(0.10) + QUANT(0.06)
  → TECH_QUANT (0.12) — 技术指标+量化因子合并

SECTOR_POLICY(0.06) + INDUSTRY(0.06) + MACRO(0.04)
  → SECTOR_MACRO (0.10) — 行业+宏观+政策合并

SENTIMENT(0.06) + NLP_SENTIMENT(0.04) + MM_BEHAVIOR(0.06)
  → SENTIMENT_FLOW (0.10) — 情绪+主力行为合并

CAPITAL_FLOW(0.15) + FLOW_DETAIL(0.05) + LIQ(0.06)
  → CAPITAL_LIQUIDITY (0.15) — 资金+流动性合并

KLINE_PATTERN(0.10) + CHART_PATTERN(0.08)
  → PATTERN (0.12) — K线形态+图形形态合并

4个KLINE视觉Agent(0.19)
  → KLINE_VISION (0.12) — 合并为1个Agent, 内部处理多周期

TOP/BOTTOM(0.08)
  → 保留, 但融入PATTERN或DIVERGENCE

TIMEFRAME_RESONANCE(0.08)
  → 保留, 作为跨周期元判断
```

**精简后Agent列表 (10个):**

| 新Agent | 原组成 | 权重 | 核心逻辑 |
|---------|--------|------|----------|
| TREND_MOMENTUM | TREND+BETA+TRENDLINE | 0.18 | 趋势+动量+趋势线突破 |
| TECH_QUANT | TECH+QUANT | 0.10 | 技术指标综合+量化因子 |
| CAPITAL_LIQUIDITY | CAPITAL_FLOW+FLOW_DETAIL+LIQ | 0.15 | 资金流向+流动性 |
| FUNDAMENTAL | 保持 | 0.12 | 基本面(PE/PB/ROE/成长/负债) |
| DIVERGENCE | 保持 | 0.10 | MACD/RSI背离 |
| CHANLUN | 保持 | 0.10 | 缠论分型→笔→中枢→买卖点 |
| PATTERN | KLINE_PATTERN+CHART_PATTERN+TOP/BOTTOM | 0.10 | 全形态合并 |
| VOLUME_PRICE | 保持+SUPPORT_RESISTANCE | 0.08 | 量价关系+支撑阻力 |
| SENTIMENT_FLOW | SENTIMENT+NLP+MM_BEHAVIOR+SECTOR_MACRO | 0.10 | 情绪+舆情+行业 |
| RESONANCE | TIMEFRAME_RESONANCE+KLINE_VISION | 0.10 | 多周期共振+视觉验证 |

**收益:**
- LLM调用从29次降到10次/Provider, 成本降65%
- 每个Agent的数据上下文更丰富(合并后), LLM研判质量更高
- 消除虚假多样性, 加权平均更有意义

#### A.2 评分公式差异化

为每个合并后Agent设计真正不同的评分逻辑:

```python
# TREND_MOMENTUM — 多指标趋势综合
def _trend_momentum_policy(snap, ctx):
    # 1. MA排列打分 (MA5>MA10>MA20>MA60=全多头+20, 反之-20)
    # 2. ADX趋势强度
    # 3. 趋势线斜率
    # 4. 乖离率惩罚
    # 区别: 不再用简单线性, 而是条件分支+非线性

# CAPITAL_LIQUIDITY — 量价确认体系
def _capital_liquidity_policy(snap, ctx):
    # 1. OBV趋势vs价格趋势一致性
    # 2. 大单净流入(如有数据)
    # 3. 换手率分位数(vs历史)
    # 4. 量比趋势(连续放量/缩量)
    # 区别: 引入历史分位数概念, 而非绝对值

# FUNDAMENTAL — 多维估值模型
def _fundamental_policy(snap, ctx):
    # 1. PEG估值(PE/净利增速)
    # 2. 行业PE分位数
    # 3. 财务健康度综合分(F-Score)
    # 4. 自由现金流趋势
    # 区别: 引入PEG和行业对比
```

---

### 方案B: 引入结构化辩论 — 借鉴TradingAgents-CN

**目标:** 将简单加权平均升级为辩论驱动的决策流程

#### B.1 辩论架构设计

```
Phase 1: 数据分析 (并行)
  ├── 技术面团队: TREND + TECH + DIVERGENCE + CHANLUN + PATTERN
  ├── 资金面团队: CAPITAL + VOLUME_PRICE + SENTIMENT
  └── 基本面团队: FUNDAMENTAL + SECTOR

Phase 2: 团队汇报 (并行, 3次LLM调用)
  ├── 技术面报告: "技术面看, 该股..."
  ├── 资金面报告: "资金面看, 该股..."
  └── 基本面报告: "基本面看, 该股..."

Phase 3: 投资辩论 (串行, 2~3轮)
  ├── Round 1: 看多方(基于3份报告中的利多因素)
  ├── Round 1: 看空方(基于3份报告中的利空因素)
  ├── Round 2: 看多反驳
  ├── Round 2: 看空反驳
  └── 仲裁者: 综合辩论, 给出投资计划

Phase 4: 风险评估 (串行, 1轮)
  ├── 激进派: 基于投资计划, 给出激进建议
  ├── 保守派: 基于投资计划, 给出保守建议
  └── 风险经理: 仲裁, 给出最终决策

输出: final_decision + target_price + confidence + risk_score + reasoning
```

#### B.2 实现方式

```python
# 不引入LangGraph, 用原生Python实现
class DebateOrchestrator:
    def run_debate(self, tech_report, capital_report, fundamental_report):
        # Round 1
        bull_arg = self.llm.chat(f"作为看多分析师, 基于以下三份报告, 论证买入理由:\n{reports}")
        bear_arg = self.llm.chat(f"作为看空分析师, 针对以下看多论点反驳:\n{bull_arg}\n原始报告:\n{reports}")

        # Round 2
        bull_rebuttal = self.llm.chat(f"看多方反驳:\n看空论点: {bear_arg}")
        bear_rebuttal = self.llm.chat(f"看空方反驳:\n看多论点: {bull_rebuttal}")

        # 仲裁
        judge_decision = self.deep_llm.chat(f"""
        作为投资委员会主席, 综合以下辩论:
        看多方: {bull_arg} → {bull_rebuttal}
        看空方: {bear_arg} → {bear_rebuttal}
        原始数据报告: {reports}
        给出: 投资计划(买入/持有/卖出) + 目标价 + 置信度 + 理由
        """)
        return judge_decision
```

**LLM调用增量:** 约6~8次额外调用(辩论+仲裁), 但这些调用的质量远高于当前29个相似prompt

---

### 方案C: 反馈闭环 — 回测与记忆

#### C.1 简易回测引擎

```python
class BacktestEngine:
    def evaluate_past_prediction(self, symbol, prediction_date, prediction):
        """对比N天后实际走势vs预测"""
        actual = self.get_price_change(symbol, prediction_date, days_forward=20)

        # 方向准确率
        direction_correct = (prediction.decision == "buy" and actual.pct > 0) or \
                           (prediction.decision == "sell" and actual.pct < 0)

        # 目标价命中率
        target_hit = actual.high >= prediction.target_price if prediction.decision == "buy" \
                    else actual.low <= prediction.target_price

        # 止损触发
        stop_triggered = actual.low <= prediction.stop_loss if prediction.decision == "buy" \
                        else actual.high >= prediction.stop_loss

        return BacktestResult(direction_correct, target_hit, stop_triggered, actual.pct)

    def evaluate_agent_accuracy(self, agent_id, lookback_days=90):
        """评估单个Agent的历史准确率"""
        predictions = self.db.get_agent_predictions(agent_id, lookback_days)
        results = [self.evaluate_past_prediction(p.symbol, p.date, p) for p in predictions]
        return AgentAccuracy(
            win_rate=sum(r.direction_correct for r in results) / len(results),
            avg_return=mean(r.actual_pct for r in results),
            sharpe=...,
        )
```

#### C.2 动态权重调整

```python
def adaptive_weights(agents, lookback_days=60):
    """基于历史表现动态调整Agent权重"""
    base_weights = {a.agent_id: a.weight for a in agents}
    accuracies = {a.agent_id: backtest.evaluate_agent_accuracy(a.agent_id, lookback_days)
                  for a in agents}

    # 表现好的Agent权重上调, 差的下调
    adjusted = {}
    for agent_id, base_w in base_weights.items():
        acc = accuracies[agent_id]
        # 胜率高于60%的Agent权重×1.3, 低于40%的×0.7
        if acc.win_rate > 0.6:
            adjusted[agent_id] = base_w * 1.3
        elif acc.win_rate < 0.4:
            adjusted[agent_id] = base_w * 0.7
        else:
            adjusted[agent_id] = base_w

    # 归一化
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}
```

---

### 方案D: 新闻/舆情增强

#### D.1 实时新闻搜索集成

```python
class NewsSearchService:
    """多引擎新闻搜索, 优先级降级"""
    engines = [
        TavilySearch(),      # 优先: 结构化搜索
        BochaSearch(),       # 备选: 中文优化
        PerplexitySearch(),  # 备选: AI增强搜索
    ]

    def search_stock_news(self, symbol, name, days=3):
        query = f"{name} {symbol} 最新消息 利好 利空"
        for engine in self.engines:
            try:
                results = engine.search(query, max_results=10)
                return self._extract_sentiment(results)
            except:
                continue
        return default_sentiment

    def _extract_sentiment(self, results):
        """LLM提取新闻情绪"""
        prompt = f"分析以下新闻, 给出情绪分(-100~100)和关键事件:\n{results}"
        return self.llm.chat(prompt)
```

#### D.2 新闻Agent独立化

将新闻从`news_c`全局变量, 改为独立的NewsAgent:
- 有自己的数据采集流程(搜索引擎→结构化提取)
- 产出: 情绪分 + 关键事件列表 + 风险预警
- 其他Agent可引用NewsAgent的输出, 但不再共享同一个`news_c`

---

### 方案E: 输出通道扩展

#### E.1 优先级排序

| 优先级 | 通道 | 实现复杂度 | 用户价值 |
|--------|------|-----------|---------|
| P0 | 微信Webhook | 低(requests.post) | 最高 — 中国用户核心渠道 |
| P1 | 飞书Webhook | 低 | 高 — 团队协作 |
| P2 | Telegram Bot | 中 | 中 — 国际用户 |
| P3 | Email | 中 | 中 — 正式报告 |
| P4 | Web Dashboard | 高(需前端) | 高 — 可视化 |

#### E.2 简单推送实现

```python
class NotificationService:
    def push_wechat(self, webhook_url, analysis_result):
        """企业微信Webhook推送"""
        markdown = self._format_markdown(analysis_result)
        requests.post(webhook_url, json={
            "msgtype": "markdown",
            "markdown": {"content": markdown}
        })

    def _format_markdown(self, result):
        emoji = "🔴" if result.decision == "buy" else "🟢" if result.decision == "sell" else "⚪"
        return f"""
{emoji} **{result.name}({result.symbol})** | 评分 {result.score:.1f} | {result.decision_level}
> 目标价: {result.target_price} | 止损: {result.stop_loss}
> 核心逻辑: {result.one_sentence}
"""
```

---

### 方案F: LLM分层优化 — 借鉴TradingAgents-CN

#### F.1 快慢双模型

```python
# 配置
"llm": {
    "quick_model": "deepseek-chat",        # 快速模型: 数据收集、格式化
    "deep_model": "grok-4-1-fast-reasoning", # 深度模型: 辩论、仲裁、最终决策
}

# 使用场景
# 快速模型: 29个Agent的enrich_and_score → 改为10个Agent的数据整理
# 深度模型: 辩论环节(Bull/Bear) + 仲裁 + 风险评估 + 场景生成
```

#### F.2 成本对比

```
当前: 29 Agent × 3 Provider × 1次调用 = 87次LLM调用

方案A(精简): 10 Agent × 3 Provider × 1次调用 = 30次LLM调用

方案A+B(精简+辩论):
  10 Agent × 3 Provider × 1次(quick) = 30次快速调用
  + 3 团队报告(quick) = 3次快速调用
  + 辩论4轮(deep) = 4次深度调用
  + 风险评估3方(deep) = 3次深度调用
  + 仲裁2次(deep) = 2次深度调用
  = 33次快速 + 9次深度 = 42次总调用(vs当前87次)
  成本下降约50%, 决策质量显著提升
```

---

## 六、实施优先级建议

| 优先级 | 方案 | 预期收益 | 实施难度 | 建议阶段 |
|--------|------|---------|---------|---------|
| **P0** | A: Agent精简与重构 | 降本65%, 消除冗余 | 中 | 立即 |
| **P1** | B: 结构化辩论 | 决策质量质变 | 中 | 第二阶段 |
| **P1** | F: LLM分层 | 成本优化+质量提升 | 低 | 与B同步 |
| **P2** | D: 新闻增强 | 信息维度扩展 | 中 | 第三阶段 |
| **P2** | E: 推送通道 | 用户体验 | 低 | 第三阶段 |
| **P3** | C: 回测闭环 | 长期迭代能力 | 高 | 第四阶段 |

---

## 七、附录: 关键代码位置索引

| 文件 | 关键函数/类 | 行号 | 说明 |
|------|-----------|------|------|
| `agents.py` | `_simple_policy()` | 291-724 | 所有维度评分公式 |
| `agents.py` | `create_agent()` | 1503-1532 | Agent工厂 |
| `agents.py` | `KlineVisionAgent` | 826-880 | 视觉Agent |
| `agents.py` | `DivergenceAgent` | 1040-1075 | 背离检测Agent |
| `agents.py` | `ChanlunAgent` | 1150-1200 | 缠论Agent |
| `orchestrator.py` | `run_analysis()` | 200-697 | 主编排流程 |
| `orchestrator.py` | `_calc_agent_llm_weight()` | 472-485 | 共识因子计算 |
| `parallel_runner.py` | `_enrich_and_score_with_fallback()` | 91-182 | 重试链 |
| `parallel_runner.py` | `_provider_worker()` | 43-457 | Provider并行执行 |
| `llm_client.py` | `enrich_and_score()` | 187-269 | 合并LLM调用 |
| `llm_client.py` | `config_weights()` | 272-277 | 配置权重归一化 |
| `data_backend.py` | `_detect_chanlun_signals()` | — | 缠论检测 |
| `data_backend.py` | `_detect_divergence()` | — | 背离检测 |
| `data_backend.py` | `_detect_advanced_kline_patterns()` | — | K线形态检测 |
| `core/router.py` | `LLMRouter` | 128-150 | 多Provider抽象 |
| `core/runner.py` | `_timed_call()` | — | 超时控制 |
| `configs/project.json` | — | — | 全局参数 |
| `configs/agents/*.json` | — | — | 29个Agent配置 |
