# 更新历史

## 2026-03-22 — v2 里程碑

### Cloubic 桥接 + 模型降级链 + 辩论鲁棒性 + 北交所数据源

- **Cloubic 统一调度平台**：claude/gemini/openai/glm/qwen/deepseek 通过 Cloubic 国内直连，无需代理
- **模型降级链**：每个 Provider 配置最优→次优→次次优模型链（Cloubic 逗号分隔 + 直连 FALLBACK_MODEL）
- **全量模型升级**：glm-5, kimi-k2.5, grok-4.20-reasoning, MiniMax-M2.7, qwen3.5-plus, sonar-pro, gpt-5.4, cc/claude-opus-4-6
- **辩论 JSON 鲁棒性**：`_chat_for_json()` 统一重试框架，multi-agent 仲裁，盈亏比硬约束 (target-price >= 2*risk)
- **Cloubic provider 过滤修复**：`_has_api_key()` 支持 ANTHROPIC_API_KEY 映射 + CLOUBIC_API_KEY 认可
- **北交所数据源**：TDX .day/.lc5 二进制直接解析（bj 市场 mootdx 不支持）
- **狙击点位校验**：`_fix_sniper_points()` 盈亏比>=2 + 止损<=10% 后处理
- **DeepSeek reasoner 兼容**：content 为空时取 reasoning_content

### 数据获取梯次化增强

- **5 级 K 线降级**：TDX本地 → Tushare日线 → AKShare日线 → Tushare周线 → AKShare周线
- **PE 修正**：TDX 快照无 pe_ttm 时从基本面接口补充
- **TDX 信号量**：vipdoc 目录不存在时静默降级
- **盘中虚拟 K 线**：1h K 线合成当日虚拟日线

### 新闻/舆情数据增强

- Perplexity sonar-pro 实时新闻搜索
- 情感评分融入 sentiment_flow_agent
- 新闻摘要写入分析上下文

### 结构化辩论机制 + LLM 分层配置

- **四阶段辩论**：团队汇报 → 多轮辩论(bull vs bear) → 仲裁决策 → 风险评估
- **辩论评分融合**：加权评分 × 60% + 辩论评分 × 40%
- **Grok Multi-Agent API**：`/v1/responses` 端点，4-16 路推理线程并行分析（`debate_multi_agent` 开关）
- **LLM 分层**：debate_provider 独立配置，辩论与评分使用不同模型

### Agent 精简 29→12 + 差异化评分公式

- 合并冗余维度：TREND+TRENDLINE→趋势动量综合, CAPITAL_FLOW+LIQ→资金流动性, TECH+SUPPORT_RESISTANCE→技术指标量化
- 12 个精简 Agent 配置：`configs/agents_v2/`
- 差异化评分公式：`_simple_policy()` 中每个 Agent 独立公式

## 2026-03-13

- **feat: 逐 Agent 动态 LLM 融合权重** — 按可公式化程度分三档（A=0.20/B=0.35/C=0.45），根据跨 Provider 评分标准差动态调整
- **feat: 评分历史记录** — 每次分析自动存储快照到 `output/history/{symbol}/{date}.json`，支持逐日对比和趋势追踪
- **feat: 突破检测 v2** — 真实趋势线构造 + W 底/M 顶颈线突破 + 三角形边界突破 + 统一确认机制
- **feat: 盘中虚拟日线** — 1h K 线合成当日虚拟日线，支持盘中研判
- **feat: Gemini 三级模型降级** — 429 限速时自动降级到备用模型
- **perf: 短线权重优化** — 多周期加权调整为日 0.50/周 0.40/月 0.10

## 2026-03-09

- **feat: 相对强弱多层对标增强**（vs 行业/龙头 TOP3/行业 ETF）
- **feat: K 线形态识别增强**，新增 12 种形态 + 位置权重 + 连续性统计
- **feat: 国产视觉回退链**，无视觉能力 Provider 自动借用国产视觉 API
- **perf: 网络可靠性优化**，运行耗时从 300s 降至 107s
- **perf: 研判+评分合并为 1 次 LLM 调用**，Provider 内 Agent 并发执行

## 2026-03-05

- **feat: 7 个新技术 Agent** — DIVERGENCE, VOLUME_PRICE, SUPPORT_RESISTANCE, CHANLUN, CHART_PATTERN, TIMEFRAME_RESONANCE, TRENDLINE
- **feat: KLINE_PATTERN Agent** — 15+ 经典 K 线形态检测（单根→五根 5 类层次）
- **feat: DERIV_MARGIN Agent** — 融资融券分析
- **feat: Vision K 线分析** — 4 个视觉 Agent 分析 1h/日/周/月 K 线图表
- **feat: 通达信本地数据源** — mootdx 读取 TDX 日/周/月 K 线

## 2026-03-01

- 初始版本：29 智能体 + 多 Provider 并行 + PDF 报告
