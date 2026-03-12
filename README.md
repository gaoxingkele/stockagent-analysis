# stockagent-analysis

A 股个股多智能体分析决策系统。29 个专业智能体并行研判，多大模型交叉验证，输出投资者级 PDF 报告。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI (run.py)                           │
│         python run.py analyze --symbol 300617 ...           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Orchestrator 流水线                        │
│  数据采集 → 本地分析 → 并行分析 → 合并评分 → 输出报告         │
└──────┬───────────┬──────────┬──────────┬────────────────────┘
       │           │          │          │
  DataBackend   29 Agents  ParallelRunner  ReportPDF
  (数据层)     (智能体层)   (多模型并行)    (报告层)
```

### 核心设计理念

- **本地公式 + LLM 动态融合评分**：每个智能体根据"可公式化程度"分配不同的 LLM 权重（0.15-0.50），再由跨 Provider 一致性动态调整，量化信号以公式为主、语义判断给 LLM 更多话语权
- **多模型交叉验证**：默认 3 个 Provider 并行运行，各自独立分配权重，取加权平均作为最终分数
- **候选备选池**：默认 Provider 失败时自动切换到候选 Provider 重试

## 功能特性

### 29 智能体体系

#### 短线参考（5 日内）

| 智能体 | 说明 | 权重 |
|--------|------|------|
| TREND | 趋势结构分析（均线/动量/涨幅） | 18% |
| TECH | 技术指标综合（MACD/RSI/KDJ/布林） | 12% |
| CAPITAL_FLOW | 资金流向分析 | 15% |
| SENTIMENT | 新闻舆情情绪 | 5% |
| KLINE_PATTERN | 15+ 经典 K 线组合形态识别 | 10% |
| DIVERGENCE | MACD/RSI 顶底背离检测 | 10% |
| VOLUME_PRICE | 量价信号（放量突破/缩量回踩/OBV） | 8% |
| SUPPORT_RESISTANCE | 支撑阻力位（前高前低/缺口/整数关口） | 8% |
| TRENDLINE | 趋势线斜率+突破信号 | 6% |
| KLINE_1H / KLINE_DAY | 1h/日 K 线视觉分析 | 各 3% |
| TOP_STRUCTURE | 顶部结构识别（高分=卖出警示） | 3% |
| BOTTOM_STRUCTURE | 底部结构识别（高分=买入参考） | 3% |

#### 中长期参考（1 月以上）

| 智能体 | 说明 | 权重 |
|--------|------|------|
| FUNDAMENTAL | 基本面（PE/PB/ROE/营收增速） | 8% |
| BETA | Beta 风险系数 | 10% |
| LIQ | 流动性分析（换手率/成交额） | 5% |
| SECTOR_POLICY | 行业政策与板块轮动 | 5% |
| DERIV_MARGIN | 融资融券分析 | 12% |
| CHANLUN | 缠论（分型→笔→中枢→三类买卖点） | 10% |
| CHART_PATTERN | 图表形态（三角形/箱体/旗形/杯柄） | 8% |
| TIMEFRAME_RESONANCE | 日/周/月多周期共振 | 8% |
| KLINE_WEEK / KLINE_MONTH | 周/月 K 线视觉分析 | 各 3% |
| QUANT | 量化因子 | — |
| MACRO | 宏观经济 | — |
| INDUSTRY | 行业竞争格局 | — |
| FLOW_DETAIL | 资金细分（北向/主力/散户） | — |
| MM_BEHAVIOR | 主力行为分析 | — |
| NLP_SENTIMENT | NLP 情绪深度分析 | — |

> **顶底结构评分语义**：TOP_STRUCTURE 高分 = 顶部信号显著（卖出警示），低分 = 无顶部信号（中性）；BOTTOM_STRUCTURE 高分 = 底部信号显著（买入参考），低分 = 无底部信号（中性）。两者独立评估、交叉印证，并标注信号所在的周期级别（日线/周线/月线）。TOP_STRUCTURE 在最终加权评分中自动反转（100 - score），使顶部信号拉低总分。

### 多数据源

| 优先级 | 数据源 | 说明 |
|--------|--------|------|
| 1 | 通达信本地 (mootdx) | TDX 目录读取日/周/月 K 线，零延迟 |
| 2 | Tushare | API 获取行情/财务/资金流向 |
| 3 | AKShare | 免费备选，自动降级 |

### 多模型并行

```
默认 Provider (并行线程)          候选备选池 (按需激活)
┌──────────┐ ┌──────────┐ ┌──────────┐    ┌──────────────────────────┐
│ deepseek │ │   grok   │ │   glm    │    │ kimi / gemini / doubao / │
│  Thread  │ │  Thread  │ │  Thread  │    │ minmax (失败时自动切换)   │
└──────────┘ └──────────┘ └──────────┘    └──────────────────────────┘
```

- 每个 Provider 独立分配 29 个智能体权重 → 加权评分 → 取平均
- 单 Agent 超时 300s → 重试 1 次 → 随机候选 Provider 最多 4 次
- 海外模型 (grok/gemini/claude/openai) 走 `LLM_PROXY`，国内模型直连
- 每个 Provider 有两个模型：`{PROVIDER}_MODEL`（推理）+ `{PROVIDER}_VISION_MODEL`（视觉）

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
# 单只股票，使用默认 Provider (deepseek, grok, glm)
python run.py analyze --symbol 300617 --name 安靠智电

# 指定 Provider
python run.py analyze --symbol 300617 --name 安靠智电 --providers deepseek,grok,glm

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
│   └── agents/                     # 29 个智能体配置
│       ├── trend_agent.json
│       ├── chanlun_agent.json
│       ├── kline_day_agent.json
│       └── ...
├── src/stockagent_analysis/
│   ├── orchestrator.py             # 流水线编排
│   ├── agents.py                   # 智能体实现（9 个 Agent 类）
│   ├── parallel_runner.py          # 多 Provider 并行执行
│   ├── data_backend.py             # 数据采集 + 技术分析引擎
│   ├── llm_client.py               # LLM 路由 + Vision 支持
│   ├── report_pdf.py               # PDF 报告生成
│   ├── chart_generator.py          # K 线图表生成
│   ├── score_history.py             # 评分历史记录与逐日对比
│   ├── progress_display.py         # 终端进度表格渲染
│   ├── agent_data_mapping.py       # 智能体-数据映射表
│   ├── main.py                     # CLI 参数解析
│   ├── config_loader.py            # 配置加载
│   ├── io_utils.py                 # I/O 工具
│   └── doc_converter.py            # 文档转换
├── docs/
│   ├── AGENT_SYSTEM_CN.md          # 智能体架构说明
│   ├── agents/*.md                 # 智能体定义文档
│   └── ...
└── output/runs/                    # 运行输出目录
```

## 智能体类继承

```
AnalystAgent (基类)
├── KlineVisionAgent          # K 线视觉分析 (4 agents)
├── DivergenceAgent           # MACD/RSI 背离
├── VolumePriceAgent          # 量价信号
├── SupportResistanceAgent    # 支撑阻力
├── TimeframeResonanceAgent   # 多周期共振
├── TrendlineAgent            # 趋势线
├── ChartPatternAgent         # 图表形态
├── ChanlunAgent              # 缠论
└── KlinePatternAgent         # K 线形态组合
```

## 配置说明

### project.json 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `decision_threshold_buy` | 70 | 买入阈值 |
| `decision_threshold_sell` | 50 | 卖出阈值 |
| `debate_rounds` | 2 | 辩论轮次 |
| `multi_model_weight_mode` | true | 多模型权重模式 |
| `llm.default_providers` | ["deepseek","grok","glm"] | 默认并行 Provider |
| `llm.candidate_providers` | ["gemini","kimi","minmax","doubao","glm"] | 候选备选池 |
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

### 2026-03-13

- **feat: 逐 Agent 动态 LLM 融合权重** — 按可公式化程度分三档（A=0.20/B=0.35/C=0.45），根据跨 Provider 评分标准差动态调整，量化 Agent 公式主导、语义 Agent LLM 主导
- **feat: 评分历史记录** — 每次分析自动存储快照，支持逐日对比和趋势追踪
- **feat: 突破检测 v2** — 真实趋势线构造 + W 底/M 顶颈线突破 + 三角形边界突破 + 统一确认机制
- **feat: 盘中虚拟日线** — 1h K 线合成当日虚拟日线，支持盘中研判
- **feat: Gemini 三级模型降级** — 429 限速时自动降级到备用模型
- **perf: 短线权重优化** — 多周期加权调整为日 0.50/周 0.40/月 0.10，降低宏观/基本面 Agent 权重
- **fix: 候选 Provider 排序** — GLM 移到候选末位，minmax 提前

### 2026-03-09

- **feat: 相对强弱多层对标增强**（vs 行业/龙头 TOP3/行业 ETF）
- **feat: K 线形态识别增强**，新增 12 种形态 + 位置权重 + 连续性统计
- **feat: 国产视觉回退链**，无视觉能力 Provider 自动借用国产视觉 API
- **perf: 网络可靠性优化**，运行耗时从 300s 降至 107s
- **perf: 研判+评分合并为 1 次 LLM 调用**，Provider 内 Agent 并发执行

## License

MIT
