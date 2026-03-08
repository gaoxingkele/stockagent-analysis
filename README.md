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

- **LLM 管权重，本地公式管评分**：LLM 为每个智能体分配权重（擅长度），本地数据驱动公式产出 0-100 评分，避免 LLM 评分膨胀
- **多模型交叉验证**：默认 3 个 Provider 并行运行，各自独立分配权重，取加权平均作为最终分数
- **候选备选池**：默认 Provider 失败时自动切换到候选 Provider 重试

## 功能特性

### 29 智能体体系

| 类别 | 智能体 | 说明 |
|------|--------|------|
| **核心维度 (9)** | TREND, TECH, CAPITAL_FLOW, FUNDAMENTAL, SENTIMENT, BETA, SECTOR_POLICY, LIQ, DERIV_MARGIN | 趋势/技术/资金/基本面/情绪/风险/行业/流动性/融资融券 |
| **K线视觉 (4)** | KLINE_1H, KLINE_DAY, KLINE_WEEK, KLINE_MONTH | 多周期 K 线图表视觉分析（Vision 模型） |
| **技术形态 (7)** | DIVERGENCE, VOLUME_PRICE, SUPPORT_RESISTANCE, CHANLUN, CHART_PATTERN, TIMEFRAME_RESONANCE, TRENDLINE | 背离/量价/支撑阻力/缠论/图表形态/多周期共振/趋势线 |
| **结构识别 (2)** | TOP_STRUCTURE, BOTTOM_STRUCTURE | 顶底结构信号 |
| **K线形态 (1)** | KLINE_PATTERN | 15+ 经典 K 线组合形态识别 |
| **扩展分析 (5)** | QUANT, MACRO, INDUSTRY, FLOW_DETAIL, MM_BEHAVIOR, NLP_SENTIMENT | 量化/宏观/行业/资金细分/主力行为/NLP情绪 |
| **管理 (1)** | MANAGER | 主控编排智能体 |

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

### Vision K 线分析

- 4 个视觉智能体分析 1h/日/周/月 K 线图表
- 图表含 MA/布林/MACD/RSI/KDJ/趋势线等叠加指标
- 自动检测 Vision 模型支持：`{PROVIDER}_VISION_MODEL` 环境变量
- 不支持视觉的 Provider 自动降级为文本分析

### 技术分析引擎

本地计算，无需 LLM：

- **K 线形态**：锤子/吞噬/早晨之星/红三兵/W 底/头肩等 15+ 种
- **背离检测**：MACD/RSI 顶底背离，多周期加权
- **缠论**：分型 → 笔 → 中枢 → 三类买卖点
- **图表形态**：三角形/箱体/旗形/杯柄/圆弧顶底
- **量价关系**：放量突破/缩量回踩/底部放量/天量滞涨/OBV
- **支撑阻力**：前高前低/均线/密集成交区/跳空缺口/整数关口
- **多周期共振**：日/周/月趋势斜率 + 动量方向一致性检测

### PDF 投资者报告

生成专业级 PDF 报告，包含：

1. 全智能体加权评分汇总表（29 个智能体评分 + 五级建议）
2. 多模型权重分配对比表
3. 均线系统 MA5-MA250 指标
4. 三周期技术指标总览（日/周/月）
5. Bull vs Bear 多空辩论
6. 行业竞争格局分析
7. 三情景概率表 + 分批建仓策略

### 评分体系

```
评分 ≥ 70  →  买入
50 ≤ 评分 < 70  →  中立（持有）
评分 < 50  →  卖出
```

- 本地公式覆盖 [15, 85] 评分区间，充分区分度
- 新闻情绪截断 ±10，防止单因子主导
- PE/PB 多级偏差（亏损/深度价值/合理/偏高/高估/极度高估）

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
| `llm.candidate_providers` | ["kimi","gemini","doubao","minmax"] | 候选备选池 |
| `llm.agent_call_timeout_sec` | 300 | 单 Agent 超时(秒) |
| `llm.provider_timeout_sec` | 1800 | Provider 总超时(秒) |

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

## 运行可观测性

- **流水线阶段指示**：`[v]数据采集 → [>]并行分析 → [ ]合并评分 → ...`
- **Provider 并行进度表**：表格化显示每个 Agent × Provider 的实时状态
- **智能体独立日志**：`output/runs/<run_id>/logs/<agent_id>.log`
- **最终决策 JSON**：包含评分、决策、所有智能体投票明细

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

## License

MIT
