# stockagent-analysis

面向中国股市个股分析的多智能体决策项目。

## 目标

- 分析单只 A 股股票（示例：`600519`）
- 多智能体并行研判（基本面/技术面/情绪面/风险控制）
- 通过辩论（JSON 消息）、投票（带权重）形成最终决策
- 输出“买入 / 卖出 / 观望”建议及理由
- 每个智能体独立日志与独立结论文件
- 数据后端支持 AKShare / Tushare：可切换、可组合、可按智能体单独指定
- LLM 默认采用 Grok API，同时支持 Gemini / DeepSeek / Kimi / Claude / ChatGPT
- 新增支持：MinMax / GLM / 豆包 / 千问（通过配置启用）
- 每个智能体默认使用 `grok + gemini + kimi` 三路模型进行综合评价（可在 `project.json` 调整）

## 项目结构

- `configs/project.json`：全局配置（数据后端模式、轮次、阈值等）
- `configs/agents/*.json`：每个智能体独立配置文件
- `docs/agents/*.md`：每个智能体独立定义文件
- `docs/AGENT_SYSTEM_CN.md`：智能体关系与角色定义
- `docs/CONVERSION_GUIDE_CN.md`：他类智能体文档转中国股市定义的规范
- `src/stockagent_analysis/`：核心代码
- `output/runs/`：每次运行的日志、消息、提交结论

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量（建议）

- 复制 `.env.example` 为 `.env`
- 填写 `TUSHARE_TOKEN`
- 按需填写 LLM 的 API Key（默认 `LLM_PROVIDER=grok`）

3. 运行分析

```bash
python run.py analyze --symbol 600519 --name 贵州茅台
```

临时指定 LLM 提供商：

```bash
python run.py analyze --symbol 600519 --name 贵州茅台 --provider gemini
```

## 文档驱动设定

- 已读取并对齐 `src/MultiAgent-Stock.docx` 的 19-Agent 架构
- 当前包含：1个主控Agent + 9个核心Agent + 10个专家Agent
- 所有智能体都有独立 `.md` 定义与独立 `.json` 配置（含数据源定义）

## 运行可观测性

- 终端显示分析进度（分析阶段/辩论阶段/评分阶段/输出阶段）
- 智能体独立日志：`output/runs/<run_id>/logs/<agent_id>.log`
- 智能体沟通 JSON：`output/runs/<run_id>/messages/*.json`
- 智能体提交 JSON：`output/runs/<run_id>/submissions/*.json`
- 最终决策 JSON：`output/runs/<run_id>/final_decision.json`
- 最终投资者报告 PDF：`output/runs/<run_id>/<股票代码>_<中文名>_投资者报告.pdf`（宋体、小四、重点加粗）

4. 查看输出

- `output/runs/<run_id>/logs/*.log`：每个智能体日志
- `output/runs/<run_id>/messages/*.json`：智能体间商议消息
- `output/runs/<run_id>/submissions/*.json`：提交给管理智能体的结论
- `output/runs/<run_id>/final_decision.json`：最终决策
- `output/runs/<run_id>/<股票代码>_<中文名>_投资者报告.pdf`：投资者阅读版报告（PDF）

## 文档转换（他类智能体设计 → 中国股市）

```bash
python run.py convert-doc --input "D:\path\other-domain-agent-design.md"
```

转换结果位于 `docs/converted/`。
