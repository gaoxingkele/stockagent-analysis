# LLM Agent 架构重构开发计划

**作者：** gaoxingkele
**创建日期：** 2026-04-18
**预计工期：** 4-5 周
**参考项目：** `D:\aicoding\gtihub\TradingAgents`
**重构前基线 tag：** `v2.9.0-prereform-baseline`

---

## 背景与动机

### 现状问题

当前系统采用"每个量化因子 × 5 个 LLM 独立打分"的架构：

- 12 个 Agent × 5 个 LLM ≈ 60 次 LLM 调用/股
- LLM 对结构化因子（缠论分型、MACD 背离、Donchian 阶段）判断经常出错（memory 中已记录）
- 典型漂移案例：新希望分析中 gemini 给 ichimoku 打 8 分、chanlun 打 0 分，其他 4 模型给 56-82 分，严重拖低均分
- 架构让 LLM 做它不擅长的事（对确定性因子打分），却没有发挥它擅长的能力（推理、角色扮演、跨维度综合、辩论）
- Token 成本高，且每个 LLM 调用都是"独立判断"，缺乏辩论和反思机制
- 没有交易员视角、没有风控视角、没有长期记忆

### 核心设计理念转变

**旧：** LLM 做"打分员"——给每个因子独立打分
**新：** LLM 做"角色扮演者"——扮演分析师/交易员/风控员，参与辩论

**旧：** 每个因子独立，并行打分后加权求和
**新：** 流水线状态机，上游产出报告给下游，辩论+反思

**旧：** LLM 参与结构化因子定分，结果漂移影响最终分
**新：** 结构化因子由 Python 权威计算，LLM 只做综合判断和角色视角解读

---

## TradingAgents 架构要点（参考）

### 5 层流水线

```
[Analyst 层]
  market_analyst / social_media_analyst / news_analyst / fundamentals_analyst
  → 产出 4 份客观研究报告
          ↓
[Researcher 辩论层]
  bull_researcher ↔ bear_researcher （多轮辩论）
  research_manager 仲裁 → 产出 investment_plan
          ↓
[Trader 层]
  trader → 基于 plan 输出具体交易决策（FINAL TRANSACTION PROPOSAL）
          ↓
[Risk Management 辩论层]
  aggressive_debator ↔ conservative_debator ↔ neutral_debator
  portfolio_manager 拍板 → 产出最终风险敞口决策
          ↓
[Memory/反思层]
  每个关键角色（bull/bear/trader/judge/portfolio_mgr）都有向量库
  从历史决策中召回相似情境，学习教训
```

### 关键设计

1. **快慢两种 LLM 分工**
   - `quick_thinking_llm`：快速推理，用于分析师、辩手
   - `deep_thinking_llm`：深度推理，用于 Judge、Portfolio Manager 等拍板角色

2. **流水线状态机（LangGraph）**
   上游产出报告存入 `state`，下游从 `state` 读取。状态在整个流水线流转。

3. **辩论是核心机制**
   - 投资方向层：Bull vs Bear，3 轮辩论后 Judge 仲裁
   - 风控层：Aggressive vs Conservative vs Neutral，Portfolio Manager 拍板

4. **长期记忆**
   每个角色有独立向量库，基于当前情境（报告综合）召回 n 条相似历史决策，学习经验教训

5. **角色分工明确**
   每个 LLM 节点只做一件事（分析某个维度 / 为某方辩护 / 做最终决策），prompt 简短聚焦

---

## 新架构设计

### 整体流水线（6 Phase）

```
┌──────────────────────────────────────────────────────────────┐
│  Phase 0: 量化事实层 (纯 Python, 无 LLM)                      │
│  ──────────────────────────────────────                        │
│  输出 6 份客观事实报告 (markdown 格式)，作为所有 LLM 输入：    │
│   • 技术面报告  (MA/MACD/RSI/KDJ/布林/多周期)                  │
│   • K线结构报告 (缠论分型中枢/Donchian 8阶段/Ichimoku)         │
│   • 波浪结构报告(ZigZag识别当前波浪位/目标位/斐波那契)         │
│   • 资金/筹码报告(量比/OBV/换手/主力/北向/筹码分布)            │
│   • 基本面报告  (PE/PB/ROE/同业对比/财报)                      │
│   • 舆情环境报告(新闻/融资融券/市场感知/板块热度)              │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Phase 1: 专业视角分析师 (4 个 LLM 角色，并行)                 │
│  ──────────────────────────────────────                        │
│   • K走势结构分析师  → 重点读 P0 技术+K线结构报告               │
│   • 波浪理论分析师   → 重点读 P0 波浪+K线结构报告               │
│   • 短线做T分析师    → 重点读 P0 技术+资金报告                  │
│   • 马丁策略交易员   → 重点读 P0 技术+资金+舆情报告             │
│  每个角色输出：研判 + 评分(0-100) + 关键位 + 风险点             │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Phase 2: 多空辩论 (2 个 LLM 角色，3 轮辩论)                   │
│  ──────────────────────────────────────                        │
│   • Bull Analyst    → 构建多方论据，反驳空方                    │
│   • Bear Analyst    → 构建空方论据，反驳多方                    │
│   • Research Judge  → 仲裁 → 产出 investment_plan               │
│     (方向+置信度+核心理由+关键位)                               │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Phase 3: 首席交易员 (1 个 LLM 角色)                           │
│  ──────────────────────────────────────                        │
│   • Head Trader → 读 P0+P1+P2 全部输出，产出交易方案：          │
│     - 入场价（理想买/次选买/追涨）                              │
│     - 目标价（目标1/目标2）                                     │
│     - 止损价                                                    │
│     - 建议初始仓位                                              │
│     - 持有周期                                                  │
│     - FINAL TRANSACTION PROPOSAL: BUY / HOLD / SELL            │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Phase 4: 风控三辩论 (3 LLM + 1 决策)                          │
│  ──────────────────────────────────────                        │
│   • Aggressive   → 挑战 "太保守，错失良机"                      │
│   • Conservative → 挑战 "太激进，风险未识别"                    │
│   • Neutral      → 权衡双方                                     │
│   • Portfolio Manager → 最终拍板：                              │
│     - 仓位上限（占账户 %）                                      │
│     - 止损纪律（严格 / 移动 / 分批）                            │
│     - 是否加仓/减仓条件                                         │
│     - 黑天鹅应对                                                │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  Phase 5: 综合评分 + 报告生成                                  │
│  ──────────────────────────────────────                        │
│   final_score = α·量化合成分 + β·多空 Judge 分 + γ·风控评级     │
│   最终 PDF 报告 = Phase 0-4 所有产出汇总                       │
└──────────────────────────────────────────────────────────────┘
```

### 角色配置表

| 层 | 角色 | 职责 | 输入 | 输出 | LLM 类型 |
|----|------|------|------|------|---------|
| P0 | 6 个数据 Agent | 量化计算 | 行情/基本面原始数据 | 6 份 markdown 报告 | 无（纯 Python）|
| P1 | K线结构分析师 | 技术派视角 | P0 全部 | 结构评分+关键位+研判 | 快速推理 |
| P1 | 波浪理论分析师 | 艾略特波浪视角 | P0 全部 | 波浪位+目标位+研判 | 深度推理 |
| P1 | 短线做T分析师 | 日内波段视角 | P0 全部 | T+0 机会点+日内策略 | 快速推理 |
| P1 | 马丁策略交易员 | 网格/逆势加仓视角 | P0 全部 | 加仓阈值+极端止损 | 快速推理 |
| P2 | Bull Analyst | 看多辩手 | P0 + P1 | 多方论据 | 快速+记忆 |
| P2 | Bear Analyst | 看空辩手 | P0 + P1 | 空方论据 | 快速+记忆 |
| P2 | Research Judge | 多空仲裁 | Bull/Bear 辩论历史 | 方向+置信度+关键位 | **深度推理+记忆** |
| P3 | Head Trader | 交易方案制定 | P0+P1+P2 | 入场/目标/止损/仓位 | **深度推理+记忆** |
| P4 | Aggressive | 激进风控辩手 | P3 方案 | 激进立场 | 快速推理 |
| P4 | Conservative | 保守风控辩手 | P3 方案 | 保守立场 | 快速推理 |
| P4 | Neutral | 中性风控辩手 | P3 方案 | 权衡立场 | 快速推理 |
| P4 | Portfolio Manager | 风险拍板 | 风控辩论历史 | 仓位+纪律+红线 | **深度推理+记忆** |

**合计：11 个 LLM 角色，平均 15-20 次调用/股**（vs 现有 60+ 次，降 ~70%）

### LLM 分工

| 分类 | 模型建议 | 用途 |
|------|---------|------|
| quick_thinking_llm | grok-fast / doubao / claude-haiku | 分析师/辩手（需要快速多轮）|
| deep_thinking_llm | claude-opus / gpt-5 / gemini-pro | Judge / Trader / Portfolio Mgr（关键决策）|
| vision_llm | gemini / claude-vision | K 线图像识别（保留现有）|

### 向量记忆库设计

每个需要学习的角色独立一个向量库：

```
memory/
├── bull_memory/           # 看多分析师历史论据
├── bear_memory/           # 看空分析师历史论据
├── research_judge_memory/ # Judge 历史仲裁记录
├── trader_memory/         # Trader 历史交易方案
└── portfolio_mgr_memory/  # 风控历史决策
```

每条记录结构：
```json
{
  "timestamp": "2026-04-18",
  "symbol": "000876",
  "situation": "当时综合报告摘要（用于向量检索）",
  "decision": "当时决策内容",
  "outcome": "20日后回测结果（价格涨跌幅）",
  "lesson": "教训总结（如果决策错误）"
}
```

召回方式：当前情境 embedding → top-N 相似历史决策 → 作为 prompt 一部分

实现选型：
- 向量库：ChromaDB（简单、本地、无外部依赖）或复用现有 `memory.py`
- Embedding：OpenAI text-embedding-3-small / BGE-M3（本地）

---

## 现有系统对比

| 维度 | 现有系统 | 新架构 | 提升 |
|------|---------|--------|------|
| LLM 作用 | 给每个因子打分 | 扮演角色、辩论、推理 | 质变 |
| LLM 调用数 | 60-70 次/股 | 15-20 次/股 | -70% |
| 辩论机制 | 单轮 Bull-Bear（简单） | 两层辩论（投资方向+风控），各 3 轮 | +质量 |
| 记忆机制 | 无 | 每角色向量库召回历史 | 新增 |
| 结构化因子处理 | LLM 参与打分（易漂移） | 纯本地量化（权威） | 消除漂移 |
| 交易方案产出 | 规则后处理生成 | Trader + PM 协作 | 智能化 |
| 风控 | 缺失 | 三方辩论独立层 | 新增 |
| Token 成本 | 基准 | ~30% | -70% |
| 单股耗时 | 5-10 分钟（并行为主） | 10-15 分钟（流水线） | +5 分钟 |

---

## 实施路径（分阶段）

### 阶段 0：准备工作（0.5 周）

- [x] 打基线 tag：`v2.9.0-prereform-baseline`
- [ ] 新建重构分支：`feat/llm-role-refactor`
- [ ] 搭建新目录结构：
  ```
  src/stockagent_analysis/
  ├── agents_v3/          # 新架构模块根
  │   ├── phase0_data/    # 量化事实层
  │   ├── phase1_experts/ # 专业视角分析师
  │   ├── phase2_debate/  # 多空辩论
  │   ├── phase3_trader/  # 首席交易员
  │   ├── phase4_risk/    # 风控辩论
  │   └── orchestrator_v3.py # 新流水线
  ├── memory_v3/          # 向量记忆库
  └── prompts_v3/         # 新角色 prompt 库
  ```

### 阶段 1：Phase 0 量化事实层（1 周）

**目标：** 把现有 12 个 agent 的本地量化部分抽出，产出 6 份 markdown 报告，完全脱离 LLM

- [ ] 定义 6 份报告的 schema（markdown 结构）
- [ ] 实现 `phase0_data/technical_report.py`（技术面）
- [ ] 实现 `phase0_data/structure_report.py`（K线结构：缠论/Donchian/Ichimoku）
- [ ] 实现 `phase0_data/wave_report.py`（波浪理论：ZigZag + 斐波那契）
- [ ] 实现 `phase0_data/capital_report.py`（资金/筹码）
- [ ] 实现 `phase0_data/fundamental_report.py`（基本面）
- [ ] 实现 `phase0_data/sentiment_report.py`（舆情环境）
- [ ] 单元测试：每份报告在真实数据上生成，人工 review

### 阶段 2：Phase 1 专业视角分析师（1 周）

**目标：** 4 个 LLM 角色，各自读 P0 全部报告，输出视角研判

- [ ] 写 4 份 system prompt（K走势/波浪/短线做T/马丁）
- [ ] 实现 `phase1_experts/base_expert.py`（共享基类）
- [ ] 实现 4 个角色类
- [ ] 并行执行框架（复用现有 `parallel_runner.py`）
- [ ] 输出结构化：`{role, analysis, score, key_levels, risks}`

### 阶段 3：Phase 2 多空辩论（1 周）

**目标：** 3 轮 Bull vs Bear 辩论 + Judge 仲裁

- [ ] 实现 `phase2_debate/bull_analyst.py`
- [ ] 实现 `phase2_debate/bear_analyst.py`
- [ ] 实现 `phase2_debate/research_judge.py`
- [ ] 辩论状态机（3 轮循环，记录 bull_history / bear_history）
- [ ] Judge 输出 `investment_plan`：方向+置信度+核心理由+关键位
- [ ] 接入 bull_memory / bear_memory / judge_memory

### 阶段 4：Phase 3 首席交易员（0.5 周）

**目标：** 基于投资方向产出具体交易方案

- [ ] 实现 `phase3_trader/head_trader.py`
- [ ] 交易方案 schema：入场/目标/止损/仓位/周期
- [ ] 接入 trader_memory
- [ ] 输出以 `FINAL TRANSACTION PROPOSAL: BUY / HOLD / SELL` 结尾

### 阶段 5：Phase 4 风控三辩论（1 周）

**目标：** 激进/保守/中性辩论 + Portfolio Manager 拍板

- [ ] 实现 `phase4_risk/aggressive.py`
- [ ] 实现 `phase4_risk/conservative.py`
- [ ] 实现 `phase4_risk/neutral.py`
- [ ] 实现 `phase4_risk/portfolio_manager.py`
- [ ] 风控辩论状态机
- [ ] PM 输出：仓位上限 / 止损纪律 / 加减仓条件 / 黑天鹅应对
- [ ] 接入 portfolio_mgr_memory

### 阶段 6：Phase 5 综合评分 + 报告（0.5 周）

- [ ] 实现 `orchestrator_v3.py`（新流水线串联所有 Phase）
- [ ] 综合评分公式：`final_score = α·量化 + β·Judge + γ·PM`（α/β/γ 待回测确定）
- [ ] 升级 PDF 报告生成（新增风控章节、辩论摘要章节）
- [ ] 保留 JSON 结构化输出（final_decision.json 新 schema）

### 阶段 7：对比回测与上线（0.5-1 周）

- [ ] 在养猪股 7 只 + 建材股 18 只上跑新架构 vs 旧架构
- [ ] 对比指标：IC、胜率、夏普、单股耗时、token 消耗
- [ ] 人工 review 10 只股票的 PDF 报告，评估决策质量
- [ ] 调整评分融合系数 α/β/γ
- [ ] 文档更新：CLAUDE.md / README.md
- [ ] 合并到 main 分支

---

## 关键技术决策

### 1. 是否引入 LangGraph？

**选项 A：自写状态机（推荐）**
- 优点：无新依赖，可控性强，和现有 orchestrator 风格一致
- 缺点：需要手动管理状态流转

**选项 B：引入 LangGraph**
- 优点：TradingAgents 原生使用，节点/边定义清晰，调试工具友好
- 缺点：新依赖（langgraph, langchain-openai），学习成本，和现有 `llm_client.py` 风格不一致

**决策：** 初期用选项 A（自写），验证架构可行性后可考虑迁移 LangGraph。

### 2. 向量库选型

**选项 A：ChromaDB**（推荐）
- 本地持久化，无外部服务
- Python 原生，API 简单

**选项 B：FAISS**
- 性能更强但无持久化开箱支持
- 需要自己写序列化

**选项 C：复用现有 memory.py**
- 如果当前已有简单向量召回能力，先复用

**决策：** 先看现有 `memory.py` 的能力，不够再引 ChromaDB。

### 3. Embedding 模型

- 在线：OpenAI text-embedding-3-small（便宜、准确）
- 本地：BGE-M3 / bge-large-zh-v1.5（中文优化）

**决策：** 首选 BGE-M3 本地（避免 API 依赖），回测确认效果后再定。

### 4. 如何保持向后兼容

- 新架构独立目录 `agents_v3/`，不动 `agents.py`
- 新流水线独立入口 `orchestrator_v3.py`
- 旧系统通过 `--version v2` 或 `--version v3` 切换（默认 v3）
- 对比回测阶段保留双系统
- 稳定后废弃旧系统（但保留 tag 可回溯）

---

## 权衡与风险

### 收益

1. **LLM 用在擅长的事上** — 推理、角色扮演、辩论、跨维度综合
2. **消除结构化因子 LLM 漂移** — 本地量化权威，LLM 不插手因子定分
3. **风控层独立** — 符合真实机构流程
4. **Token 成本降 70%**
5. **辩论链路更接近真实研投流程**
6. **记忆层让系统越用越准** — 可从历史决策中学习

### 代价

1. **重构量大** — 4-5 周全职开发
2. **流水线比并行慢** — 单股耗时从 5-10 分钟增至 10-15 分钟（阶段间依赖无法并行）
3. **关键单点** — Judge / Trader / PM 判错影响大 → 必须用高质量深度推理模型
4. **评分分布重新校准** — 历史回测数据需重新跑一遍

### 风险与缓解

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| Judge/Trader/PM 决策质量不达标 | 整体输出退化 | 用 Claude Opus 级别模型，加充分 prompt 示例 |
| 辩论陷入循环/跑偏 | 浪费 token 无结论 | 限制辩论轮数（3 轮），Judge 强制输出 |
| 记忆召回不准 | 学不到有用历史 | 精心设计 situation embedding，人工审核召回样本 |
| 现有回测策略失效 | 历史数据无法直接比较 | 新增 v3 专用回测，α/β/γ 基于新数据校准 |
| 单股耗时过长 | 批量分析慢 | 分析师层（P1）并行，辩论层限制轮数 |

---

## 验收标准

### 功能验收

- [ ] 6 份 Phase 0 报告在 100 只股票上稳定生成（无失败、无 token 超限）
- [ ] 全流水线端到端跑通 10 只股票，产出 PDF 报告
- [ ] 所有 11 个 LLM 角色 prompt 稳定返回期望结构
- [ ] 记忆库正确召回历史决策（人工抽查 20 条召回样本）

### 质量验收

- [ ] 对比 7 只养猪股 + 18 只建材股的 v2 vs v3 结果
- [ ] IC(20d) 不低于 v2（+0.048）
- [ ] 胜率（65+分段）不低于 v2（74%）
- [ ] 10 份 PDF 报告经人工盲评，v3 质量评分 > v2

### 工程验收

- [ ] 单股耗时 < 15 分钟
- [ ] Token 消耗降低 > 50%
- [ ] 错误处理：任一 LLM 失败不影响整体流水线（降级到量化评分）
- [ ] 所有 Phase 输出结构化，可 JSON 化

---

## 回退方案

- 基线 tag：`v2.9.0-prereform-baseline`（指向 commit `05a6294` + 未提交改动记录在此文档）
- 重构分支：`feat/llm-role-refactor`（独立分支，不动 main）
- 如新架构未达预期，可：
  1. `git checkout v2.9.0-prereform-baseline` 回到基线
  2. 旧代码 `agents.py` + `orchestrator.py` 未被删除，直接 `--version v2` 运行
  3. 向量记忆库作为独立 feature 可保留（给 v2 也能用）

---

## 附录 A：现有未提交改动说明

tag 打在 HEAD commit `05a6294` 上。以下改动当时在工作区未提交（重构开始前应先提交或 stash）：

```
M backtest_stock_pool.txt
M configs/agents_v2/capital_liquidity_agent.json
M configs/agents_v2/ichimoku_agent.json
M src/stockagent_analysis/market_context.py
M src/stockagent_analysis/orchestrator.py

?? backtest_composite_compare.py
?? backtest_composite_v2.py
?? backtest_composite_v7.py
?? backtest_debug_80_85.py
?? build_stock_pool.py
?? docs/composite-backtest-report-baseline.md
?? run_batch_*.sh
?? logs/
```

---

## 附录 B：角色 Prompt 初稿（示例）

### K 走势结构分析师

```
你是一位专业的 A 股 K 线结构分析师。你的分析方法论包括：
1. 缠论：识别分型、笔、段、中枢
2. Donchian 通道 8 阶段状态机
3. Ichimoku 一目均衡图
4. 经典 K 线形态（头肩/三角/旗形/楔形）

你将收到 6 份客观量化报告，请从 K 线结构角度综合研判：
- 当前处于什么结构阶段？
- 结构是否完整？
- 关键结构位置在哪里？
- 结构视角下的买卖点？

输出 JSON:
{
  "structure_phase": "...",
  "key_levels": {"support": [...], "resistance": [...]},
  "buy_signal": "...",
  "sell_signal": "...",
  "score": 0-100,
  "analysis": "2-3 句精炼结论",
  "risk": "主要风险"
}
```

### 波浪理论分析师

```
你是一位艾略特波浪理论分析师。你的核心方法论：
1. 5 推动波 + 3 调整波（ABC）
2. 斐波那契回撤位/扩展位
3. 波浪等级（日/周/月）

任务：基于 6 份报告，判断：
- 当前处于哪一浪？
- 下一浪的目标位？
- 关键斐波那契位？
- 波浪失效的条件？

输出 JSON:
{
  "current_wave": "...",
  "next_target": "...",
  "fibonacci_levels": {...},
  "invalidation_condition": "...",
  "score": 0-100,
  "analysis": "...",
  "risk": "..."
}
```

### 短线做 T 分析师

```
你是一位 A 股 T+0 短线交易员。你专注于当日/3 日内的波段机会。
关注：
1. 日内量价配合
2. 涨跌停板动能
3. 龙虎榜资金意图
4. 板块联动强度

任务：判断当前是否存在做 T 机会：
- T+0 高抛低吸的节奏点？
- 量能支撑的日内阻力位？
- 做 T 的止损位？

输出 JSON:
{
  "T_opportunity": "yes/no/wait",
  "high_sell": "...",
  "low_buy": "...",
  "stop_loss": "...",
  "score": 0-100,
  "analysis": "...",
  "risk": "..."
}
```

### 马丁策略交易员

```
你是一位马丁格尔/网格策略交易员。你的核心原则：
1. 逆势加仓（跌越多加越多）
2. 资金分层（总仓位分 3-5 档）
3. 极端情况规划（最大回撤承受）
4. 反弹止盈规则

任务：给出马丁网格方案：
- 首仓入场价+仓位
- 第 2/3/4 档加仓位+仓位
- 整体止盈位
- 极端止损（资金红线）

输出 JSON:
{
  "grid": [
    {"level": 1, "price": ..., "ratio": 0.2},
    {"level": 2, "price": ..., "ratio": 0.2},
    ...
  ],
  "take_profit": "...",
  "hard_stop": "...",
  "score": 0-100,
  "analysis": "...",
  "risk": "..."
}
```

### Bull Analyst（看多辩手）

```
你是一位看多分析师。任务：基于提供的所有报告和分析师研判，构建强有力的看多论据。

要点：
- 成长潜力（增长空间、产品/品牌优势）
- 正面催化（财报/行业拐点/资金流入）
- 反驳空方（针对空方每一点论据进行针对性反驳）
- 以对话风格辩论，不是罗列数据

历史记忆召回：
{past_memories}

当前辩论上下文：
{history}
空方上一轮论据：
{current_bear_response}

请给出你的看多论据（不输出 JSON，输出辩论文本）：
```

### Research Judge（多空仲裁）

```
你是一位资深投资研究主管。刚刚 Bull 和 Bear 进行了 3 轮辩论。任务：
1. 评估双方论据强度
2. 判断哪一方论据更充分
3. 产出最终 investment_plan

投资计划结构：
- direction: BUY / HOLD / SELL
- confidence: 0-1
- key_reason: 最重要的 3 条理由
- key_levels: 关键买卖位
- time_horizon: 短线/中线/长线

历史类似情境：
{past_memories}

Bull 辩论历史：
{bull_history}
Bear 辩论历史：
{bear_history}

请输出 JSON investment_plan：
```

### Head Trader（首席交易员）

```
你是一位资深 A 股交易员。基于研究团队给你的 investment_plan，产出具体可执行的交易方案。

需要考虑：
- 当前价格位置（现价 vs 关键位）
- 入场时机（追涨/回踩/确认）
- 止损纪律
- 仓位管理
- 持有周期

历史交易记忆：
{past_memories}

investment_plan:
{investment_plan}

分析师观点：
{expert_analyses}

输出 JSON:
{
  "entry_plans": [
    {"strategy": "追涨", "entry": ..., "target": ..., "stop": ..., "rr": ...},
    {"strategy": "回踩", ...},
    {"strategy": "确认", ...}
  ],
  "position_ratio": "30%-50%",
  "time_horizon": "短线/中线",
  "final_decision": "BUY/HOLD/SELL"
}

结尾必须以 FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** 结束。
```

### Portfolio Manager（风控主管）

```
你是风控主管，刚刚激进/保守/中性三位风控分析师进行了辩论。任务：
1. 综合三方论据
2. 确定最终仓位上限
3. 制定止损纪律
4. 规划黑天鹅应对

历史风控记忆：
{past_memories}

Trader 方案：
{trader_plan}
三方辩论历史：
{risk_debate_history}

输出 JSON:
{
  "max_position_ratio": 0.5,
  "stop_loss_discipline": "严格止损 X% / 移动止损 / 分批止损",
  "add_position_condition": "...",
  "reduce_position_condition": "...",
  "black_swan_response": "...",
  "final_risk_rating": "低/中/高"
}
```

---

## 变更记录

| 日期 | 变更 | 作者 |
|------|------|------|
| 2026-04-18 | 初版（架构设计 + 实施路径 + prompt 初稿） | gaoxingkele + Claude |
