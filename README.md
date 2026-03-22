# stockagent-analysis

A 股个股多智能体分析决策系统。12 个专业智能体并行研判，4+ 大模型交叉验证，结构化辩论仲裁，输出投资者级 PDF 报告。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI (run.py)                           │
│         python run.py analyze --symbol 600388 ...           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Orchestrator 流水线                        │
│  数据采集 → 本地分析 → 并行分析 → 合并评分 → 结构化辩论 → 报告 │
└──────┬───────────┬──────────┬──────────┬────────────────────┘
       │           │          │          │
  DataBackend   12 Agents  LLMRouter    Debate
  (数据层)     (智能体层)  (多模型路由)  (辩论仲裁)
```

### 核心设计理念

- **本地公式 + LLM 动态融合评分**：每个智能体根据"可公式化程度"分配不同的 LLM 权重，量化信号以公式为主、语义判断给 LLM 更多话语权
- **多模型交叉验证**：默认 4 个 Provider 并行运行（minmax/doubao/claude/openai），各自独立评分，取加权平均
- **Cloubic 桥接 + 直连混合路由**：海外模型通过 Cloubic 国内直连（无需代理），国内模型直连 API
- **模型降级链**：每个 Provider 配置最优→次优→次次优模型链，自动降级
- **结构化辩论**：多空团队辩论 → 仲裁 → 风险评估，辩论评分与加权评分 4:6 融合
- **候选备选池**：默认 Provider 失败时自动切换到候选 Provider 重试

## 功能特性

### 12 智能体体系（v2）

| # | 智能体 | 说明 | 权重 |
|---|--------|------|------|
| 1 | 趋势动量综合 | 均线系统 + 动量 + 涨幅 + 趋势线 | 15% |
| 2 | 资金流动性 | 资金流向 + 换手率 + 成交额 | 12% |
| 3 | 基本面估值 | PE/PB/ROE + 营收增速 + 行业对比 | 10% |
| 4 | 技术指标量化 | MACD/RSI/KDJ/布林 + 支撑阻力 | 10% |
| 5 | 形态综合 | 15+ K 线形态 + 图表形态（三角/箱体/杯柄） | 8% |
| 6 | 情绪舆情 | Perplexity 实时新闻 + 情感评分 | 8% |
| 7 | 缠论买卖点 | 分型→笔→中枢→三类买卖点 | 7% |
| 8 | MACD/RSI 背离 | 多周期顶底背离检测 | 7% |
| 9 | K 线视觉综合 | 多周期 K 线图表 AI 视觉分析 | 6% |
| 10 | 多周期共振 | 日/周/月趋势方向一致性 | 6% |
| 11 | 量价结构 | 放量突破/缩量回踩/OBV/天量滞涨 | 6% |
| 12 | 融资融券 | 融资余额变化 + 融券卖出 | 5% |

### 多数据源

| 优先级 | 数据源 | 说明 |
|--------|--------|------|
| 1 | 通达信本地 (mootdx) | TDX 目录读取日/周/月 K 线，零延迟 |
| 2 | Tushare | API 获取行情/财务/资金流向 |
| 3 | AKShare | 免费备选，自动降级 |

### 多模型并行 + Cloubic 桥接

```
Cloubic 桥接 (国内直连)                     直连 API
┌──────────┐ ┌──────────┐                 ┌──────────┐ ┌──────────┐
│  claude  │ │  openai  │                 │  minmax  │ │  doubao  │
│ cc/opus  │ │ gpt-5.4  │                 │  M2.7    │ │ seed-2.0 │
└──────────┘ └──────────┘                 └──────────┘ └──────────┘
         候选备选池: grok / kimi / deepseek / glm / qwen
```

- 每个 Provider 独立运行 12 个智能体 → 加权评分 → 取平均
- **Cloubic 桥接**：claude/gemini/openai/glm/qwen/deepseek 走国内直连，无需代理
- **模型降级链**：每个 Provider 最优→次优→次次优自动切换
- 直连 Provider（grok/kimi/doubao/minmax）同样支持 FALLBACK_MODEL 降级
- 单 Agent 超时 120s → 重试 → 候选 Provider 最多 4 次
- 每个 Provider 有两个模型：`{PROVIDER}_MODEL`（推理）+ `{PROVIDER}_VISION_MODEL`（视觉）

### 模型降级链

| Provider | 最优 | 次优 | 次次优 | 路由 |
|----------|------|------|--------|------|
| Claude | cc/claude-opus-4-6 | cc/claude-sonnet-4-6 | cc/claude-haiku-4-5 | Cloubic |
| Gemini | gemini-3.1-pro | gemini-3-pro | gemini-3-flash | Cloubic |
| OpenAI | gpt-5.4 | gpt-5.2 | gpt-5.1 | Cloubic |
| GLM | glm-5 | glm-4.7 | — | Cloubic |
| Qwen | qwen3-max-thinking | qwen3-max | — | Cloubic |
| DeepSeek | deepseek-v3.2 | glm-5 | qwen3-max | Cloubic 跨模型 |
| Grok | grok-4.20-reasoning | grok-4-1-fast-reasoning | grok-4-1-fast-non | 代理直连 |
| Kimi | kimi-k2.5 | kimi-k2-turbo | moonshot-v1-128k | 直连 |
| Doubao | seed-2.0-pro | seed-1.6 | 1.5-pro-256k | 直连 |
| MiniMax | M2.7 | M2.5 | M2.0 | 直连 |

### 结构化辩论

```
Phase 1: 团队汇报 (多空+中立三方)
    ↓
Phase 2: 多轮辩论 (bull vs bear 交叉质疑)
    ↓
Phase 3: 仲裁决策 (目标价 + 止损 + 盈亏比约束)
    ↓
Phase 4: 风险评估 (激进/保守/中立三方)
    ↓
融合: 加权评分 × 60% + 辩论评分 × 40%
```

### Vision K 线分析

- 4 个视觉智能体分析 1h/日/周/月 K 线图表
- K 线根数：1h=160, 日=250, 周=156, 月=120
- 图表含 MA5/10/20/60/120/250、布林(20,2)、MACD(12,26,9)、RSI6/12/24、KDJ(9,3,3)、趋势线
- 自动检测 Vision 模型支持：`{PROVIDER}_VISION_MODEL` 环境变量
- 不支持视觉的 Provider 自动降级为文本分析
- 无可用 Vision Provider 时自动回退到 `VISION_FALLBACK_PROVIDER`（默认 kimi）

### 技术分析引擎

本地计算，无需 LLM：

- **K 线形态**：锤子/吞噬/早晨之星/红三兵/W 底/头肩等 15+ 种（单根→两根→三根→五根 5 类层次）
- **背离检测**：MACD/RSI 顶底背离，多周期加权（日 0.50 + 周 0.40 + 月 0.10）
- **缠论**：K 线包含处理 → 顶底分型 → 笔构造(间隔≥4) → 中枢(≥3 笔重叠) → 三类买卖点
- **图表形态**：三角形(上升/下降/对称)/箱体/旗形/杯柄/圆弧顶底，20-60 根 K 线级别
- **量价关系**：放量突破/缩量回踩/底部放量/天量滞涨/OBV 趋势
- **支撑阻力**：前高前低/均线/密集成交区/跳空缺口/整数关口
- **多周期共振**：日/周/月趋势斜率 + 动量方向一致性检测
- **顶底结构**：多周期上下影线 + 动量反转检测，日线/周线/月线独立评估

### PDF 投资者报告

生成专业级 PDF 报告（全表格自动换行，无文字重叠），包含：

1. 多智能体框架说明表
2. 多模型权重分配对比表
3. 公司概览与关键价位表
4. 均线系统 MA5-MA250 指标
5. 三周期技术指标总览（日/周/月）
6. **29 智能体加权评分汇总表**（分短线参考 / 中长期参考两组，含分组小计）
   - 顶底结构智能体使用独立信号标签（非标准五级建议）
   - 显示信号所在周期级别（如"顶部显著(卖出警示) [日线,周线]"）
7. Bull vs Bear 多空辩论
8. 行业竞争格局分析
9. 三情景概率表 + 分批建仓策略

### 评分体系

```
评分 ≥ 70  →  买入
50 ≤ 评分 < 70  →  中立（持有）
评分 < 50  →  卖出
```

- 本地公式覆盖 [15, 85] 评分区间，充分区分度
- **逐 Agent 动态 LLM 权重**：按可公式化程度分三档（A=0.20 量化 / B=0.35 结构 / C=0.45 语义），再根据跨 Provider 评分标准差动态调整（σ 小→加权，σ 大→降权），最终 clamp 到 [0.15, 0.50]
- 新闻情绪截断 ±10，防止单因子主导
- PE/PB 多级偏差（亏损/深度价值/合理/偏高/高估/极度高估）
- TOP_STRUCTURE 在加权中自动反转（100 - score），使顶部信号拉低总分

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写：
- `TUSHARE_TOKEN` — Tushare API Token
- 至少一个 LLM API Key（如 `DEEPSEEK_API_KEY`、`GROK_API_KEY`、`GLM_API_KEY`）

可选配置：
- `LLM_PROXY` — 海外模型代理地址（如 `http://127.0.0.1:18182`）
- `TDX_DIR` — 通达信安装目录（默认 `D:/tdx`）
- `{PROVIDER}_VISION_MODEL` — 视觉模型 ID（启用 K 线图表视觉分析）
- `VISION_FALLBACK_PROVIDER` — 视觉回退 Provider（默认 `kimi`）

### 3. 运行分析

```bash
# 单只股票，使用默认 Provider (minmax, doubao, claude, openai)
python run.py analyze --symbol 600388 --name 龙净环保

# 指定 Provider
python run.py analyze --symbol 300617 --name 安靠智电 --providers grok,kimi,deepseek

# 从检查点恢复（跳过数据采集）
python run.py analyze --symbol 300617 --name 安靠智电 --run-dir output/runs/20260308_193446_300617
```

### 4. 查看输出

```
output/runs/<run_id>/
├── <代码>_<名称>_投资者报告.pdf    # 投资者报告
├── final_decision.json             # 最终决策 JSON
├── logs/manager_orchestrator.log   # 编排日志
├── logs/<agent_id>.log             # 智能体日志
├── messages/*.json                 # 智能体间消息
├── submissions/*.json              # 智能体提交结论
└── data/                           # 缓存数据与图表
    └── charts/kline_{tf}.png       # K 线图表
```

## 项目结构

```
stockagent-analysis/
├── run.py                          # CLI 入口
├── configs/
│   ├── project.json                # 全局配置
│   ├── agents/                     # 旧版 29 智能体配置
│   └── agents_v2/                  # v2 精简 12 智能体配置
│       ├── trend_momentum_agent.json
│       ├── chanlun_agent.json
│       └── ...
├── src/stockagent_analysis/
│   ├── orchestrator.py             # 流水线编排
│   ├── agents.py                   # 智能体实现（12 个 Agent）
│   ├── debate.py                   # 结构化辩论（多空+仲裁+风险评估）
│   ├── news_search.py              # Perplexity 新闻搜索
│   ├── data_backend.py             # 数据采集 + 技术分析引擎
│   ├── llm_client.py               # LLM 业务层（评分/权重/狙击点位）
│   ├── report_pdf.py               # PDF 报告生成
│   ├── chart_generator.py          # K 线图表生成
│   ├── score_history.py            # 评分历史记录与逐日对比
│   ├── main.py                     # CLI 参数解析
│   └── config_loader.py            # 配置加载
├── src/core/
│   ├── router.py                   # LLM 统一路由（Cloubic + 直连 + 降级链）
│   ├── runner.py                   # 并行执行引擎
│   └── progress.py                 # 终端进度渲染
├── docs/
│   ├── AGENT_SYSTEM_CN.md          # 智能体架构说明
│   ├── agents/*.md                 # 智能体定义文档
│   └── ...
└── output/runs/                    # 运行输出目录
```

## 智能体类继承（v2）

```
AnalystAgent (基类, 12 个实例)
├── KlineVisionAgent          # K 线视觉分析
├── DivergenceAgent           # MACD/RSI 背离
├── VolumePriceAgent          # 量价结构
├── TimeframeResonanceAgent   # 多周期共振
├── ChanlunAgent              # 缠论买卖点
└── KlinePatternAgent         # 形态综合
```

## 配置说明

### project.json 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `decision_threshold_buy` | 70 | 买入阈值 |
| `decision_threshold_sell` | 50 | 卖出阈值 |
| `debate_rounds` | 2 | 辩论轮次 |
| `multi_model_weight_mode` | true | 多模型权重模式 |
| `llm.default_providers` | ["minmax","doubao","claude","openai"] | 默认并行 Provider |
| `llm.candidate_providers` | ["grok","kimi","deepseek","glm","qwen"] | 候选备选池 |
| `structured_debate` | true | 启用结构化辩论 |
| `news_enhance` | true | 启用 Perplexity 新闻增强 |
| `llm.agent_call_timeout_sec` | 120 | 单 Agent 超时(秒) |
| `llm.provider_timeout_sec` | 600 | Provider 总超时(秒) |
| `llm.max_agent_concurrency` | 6 | Provider 内 Agent 并发数 |
| `breakout_detection.version` | "v2" | 突破检测版本（v1=旧版, v2=含确认+量能验证） |

### 支持的 LLM Provider

| Provider | 类型 | 环境变量 |
|----------|------|----------|
| deepseek | 国内直连 | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` |
| grok | 海外代理 | `GROK_API_KEY`, `GROK_MODEL` |
| glm (智谱) | 国内直连 | `GLM_API_KEY`, `GLM_MODEL` |
| kimi (月之暗面) | 国内直连 | `KIMI_API_KEY`, `KIMI_MODEL` |
| gemini | 海外代理 | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| claude | 海外代理 | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| openai/chatgpt | 海外代理 | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| doubao (豆包) | 国内直连 | `DOUBAO_API_KEY`, `DOUBAO_MODEL` |
| qwen (通义千问) | 国内直连 | `QWEN_API_KEY`, `QWEN_MODEL` |
| minmax | 国内直连 | `MINMAX_API_KEY`, `MINMAX_MODEL` |
| perplexity | 海外代理 | `PERPLEXITY_API_KEY`, `PERPLEXITY_MODEL` |

### 重试与容错机制

```
单 Agent 调用流程:
主 Provider (120s) → 失败 → 重试 1 次 (120s) → 失败 → 随机候选 ×4 (各 120s)
                                                         ↓
                                                  全部失败 → 使用同线程已成功 Agent 均值
```

- 默认 Provider 失败后自动从候选池 (kimi/gemini/doubao/minmax) 随机选择重试
- 最多 5 次失败后放弃，用该 Provider 已完成 Agent 的平均分替代
- Provider 级总超时 600s，超时后直接结束该 Provider 线程

## 运行可观测性

- **流水线阶段指示**：`[v]数据采集 → [>]并行分析 → [ ]合并评分 → ...`
- **Provider 并行进度表**：表格化显示每个 Agent × Provider 的实时状态
- **智能体独立日志**：`output/runs/<run_id>/logs/<agent_id>.log`
- **编排日志**：`output/runs/<run_id>/logs/manager_orchestrator.log`（Provider 状态、耗时、合并结果）
- **最终决策 JSON**：包含评分、决策、所有智能体投票明细、各 Provider 独立评分

## 依赖

```
akshare>=1.14.0        # AKShare 金融数据
tushare>=1.2.89        # Tushare API
mootdx==0.11.7         # 通达信本地数据
python-dotenv>=1.0.1   # 环境变量
requests>=2.32.0       # HTTP 客户端
reportlab>=4.2.0       # PDF 生成
mplfinance>=0.12.9     # K 线图可视化
pandas-ta>=0.3.14b     # 技术指标计算
matplotlib>=3.7.0      # 绑定图表库
```

### 评分历史与逐日对比

每次分析完成后自动保存评分快照到 `output/history/{symbol}/{data_date}.json`，支持：

- **逐日评分对比**：总分趋势、各 Agent 评分变化、关键指标变化
- **趋势识别**：连续上升/下降/V 型反弹/倒 V 回落/震荡
- **可解释性分析**：找出变化最大的 Agent 维度，给出分数变化原因

### 突破检测 v2

在 TRENDLINE / SUPPORT_RESISTANCE / CHART_PATTERN 三个智能体中启用增强突破检测：

- **真实趋势线构造**：连接局部极值点（峰/谷），验证多点触碰（≥3 点）
- **颈线突破**：W 底/M 顶（20-60 根 K 线级别）颈线检测 + 目标价计算
- **三角形突破**：上升/下降/对称三角形收敛边界突破检测
- **统一突破确认**：bars_above/below + volume_ratio + body_cross + retest 四维验证
- **v1/v2 兼容**：`breakout_detection.version` 控制，v2 数据不可用时自动回退 v1

### 盘中分析

支持盘中（9:30 后）分析，自动从 AKShare 获取当日 1h K 线数据，合成虚拟日线（O=首根开盘, H=最高, L=最低, C=最新收盘），参与完整研判流程。

### Gemini 多级模型降级

当 Gemini 遇到 429 限速时，自动按模型链降级：

```
GEMINI_MODEL → GEMINI_FALLBACK_MODEL → GEMINI_FALLBACK_MODEL2
```

## 更新日志

> 完整历史见 [HISTORY.md](HISTORY.md)

### 2026-03-22 (v2 里程碑)

- **refactor: Agent 精简 29→12** — 合并冗余维度，差异化评分公式，权重从 config 直读
- **feat: 结构化辩论机制** — 多空团队辩论→仲裁→风险评估四阶段，JSON 解析重试+fallback provider
- **feat: Cloubic 统一桥接** — claude/gemini/openai/glm/qwen/deepseek 走国内直连，逗号分隔模型降级链
- **feat: 全量模型升级** — glm-5, kimi-k2.5, grok-4.20, gpt-5.4, cc/claude-opus-4-6 等
- **feat: 新闻/舆情增强** — Perplexity sonar-pro 实时新闻搜索
- **feat: 数据获取梯次化** — 5 级 K 线降级 + PE 修正 + TDX 信号量 + 盘中虚拟 K 线
- **feat: 北交所数据源** — TDX .day/.lc5 二进制直接解析
- **feat: 狙击点位校验** — 盈亏比>=2 + 止损<=10% 后处理硬约束
- **fix: Cloubic provider 过滤** — ANTHROPIC_API_KEY 映射 + CLOUBIC_API_KEY 认可

## License

MIT
