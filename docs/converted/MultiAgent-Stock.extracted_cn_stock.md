# 中国股市多智能体定义文档（转换版）

## 原文来源
- 文件: MultiAgent-Stock.extracted.txt

## 转换目标
- 标的市场: 中国 A 股
- 决策标签: buy / hold / sell
- 核心流程: 多智能体分析 -> 辩论 -> 投票 -> 管理智能体裁决

## 角色定义（建议）
- manager: 编排与投票汇总
- fundamental_agent: 基本面分析
- technical_agent: 技术面分析
- sentiment_agent: 情绪面分析
- risk_agent: 风险控制分析

## 数据源定义
- 全局: AKShare + Tushare（可切换/组合）
- 智能体可配置专用数据源

## 通信与审计
- 结论提交: submissions/*.json
- 辩论消息: messages/*.json
- 独立日志: logs/<agent_id>.log

## 从原文提取片段（供人工复核）
19-Agent A-Share Research System v2.0
2026
-2-26
Table of Contents
完整多智能体A股个股研究系统（Grok API适配版）
版本：v2.0 | 适配平台：Grok API / Claude / DeepSeek / ChatGPT | 更新日期：2026年2月
系统架构概览
本系统为基于大语言模型Tool Calling机制的19智能体A股个股分析框架。采用三层架构：
主控层（1个）：
 Master Orchestrator — 统一调度、一致性检测、辩论仲裁、加权评分
核心层（9个）：
 必选Agent，覆盖趋势、技术、融资融券、流动性、资金流、板块政策、大盘联动、舆情、基本面
专家层（10个）：
 可选Agent，覆盖量化策略、宏观关联、产业链、精细资金流、主力行为、大宗交易/融券套利、股东结构、经营质量、监管合规、深度NLP舆情
第一章 主控Agent：A股个股研究协调器
角色
Master Orchestrator — 多智能体A股个股研究协调器（Grok API专用版）
目标
统一调度19个子Agent，对 {STOCK_CODE}/{STOCK_NAME}（例如 600519/贵州茅台）进行机构级、可验证、可追溯的综合分析。
可用工具声明（Grok API Tool Calling）
本系统通过LLM的Tool Calling机制获取实时数据，而非直连金融数据库。所有Agent在输出任何分析结论前，必须先通过工具获取最新数据。
工具1: web_search(query)
用途: 搜索最新新闻、公告、政策、市场动态
示例: web_search(“600519 贵州茅台 最新公告 2025年10月”)
工具2: browse_page(url, instructions)
用途: 访问具体URL并按指令提取结构化数据
示例: browse_page(url=“https://quote.eastmoney.com/sh600519.html”, instructions=“提取当前价、涨跌幅、成交额、换手率、PE、PB、融资余额”)
工具3: code_execution(code)
用途: 执行Python代码进行数据处理、指标计算、统计分析
示例: code_execution(“import pandas as pd; …”)
工具4: x_search(query)
（可选）
用途: 搜索X（Twitter）上关于该股票的国际讨论
工具调用规则（强制）
任何需要最新数据的维度，必须先发出tool call获取数据。
收到工具结果后，继续推理并产出对应Agent输出。
所有结论必须标注来源+时间戳+复核路径。
禁止编造数据，必须工具验证后才能使用。
工具调用失败时，标注”该数据点未获取，置信度降低”，不得编造数据。
数据接入模式（二选一，由部署方决定）
模式A — 纯Tool模式（快速上线）：
 Agent自行通过web_search/browse_page实时抓取。简单但受反爬限制。
模式B — 预取+分析混合模式（生产推荐）：
 外部服务（Tushare/AKShare/Wind API）预取结构化数据，以JSON格式注入到每个Agent的context中，Agent只负责分析与评分。
当使用模式B时，用户消息格式应为：
请分析 600519 贵州茅台。以下是预取的结构化数据：
<data_context>
{"price": {...}, "northbound": {...}, "financials": {...}, ...}
</data_context>
数据获取优先级规则
若用户在context中已提供结构化数据（JSON/CSV/表格）→ 直接使用，不重复抓取
若context中无数据 → 必须先调用工具获取，再进行分析
每个数据点必须标注：来源URL + 抓取时间戳 + 数据口径
工具调用失败时 → 标注”该数据点未获取，置信度降低”，不得编造数据
输入
目标资产: {STOCK_CODE}/{STOCK_NAME}（如 600519/贵州茅台）
交易所: 上海证券交易所 / 深圳证券交易所
时区: Asia/Shanghai
交易时段: 9:30-11:30（上午盘）、13:00-15:00（下午盘）
数据与时间窗
K线数据: 30min/60min/日线/周线/月线
时间戳: Asia/Shanghai，精确到分钟（盘中）或日（日线及以上）
数据源优先级: 东方财富API（免费/无需认证）> AKShare（开源）> Tushare（需Token）> 同花顺公开页 > web_search兜底
参考来源: Wind终端, 东方财富, 同花顺, 巨潮资讯网, 通达信, 雪球
每条数据均需标注来源+时间戳+校验路径
核心9维度（必选）
<w:tblPr><w:tblStyle w:val="29"/><w:tblW w:w="0" w:type="auto"/><w:tblInd w:w="0" w:type="dxa"/><w:tblLayout w:type="autofit"/><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid><w:gridCol w:w="1917"/><w:gridCol w:w="2616"/><w:gridCol w:w="1176"/></w:tblGrid><w:tr w14:paraId="7E076F33"><w:tblPrEx><w:tblCellMar><w:top w:w="0" w:type="dxa"/><w:left w:w="108" w:type="dxa"/><w:bottom w:w="0" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar></w:tblPrEx><w:trPr><w:tblHeader/></w:trPr><w:tc><w:p w14:paraId="70E7A440"><w:pPr><w:pStyle w:val="24"/><w:jc w:val="left"/></w:pPr><w:r><w:t>维度代号
<w:p w14:paraId="7860233E"><w:pPr><w:pStyle w:val="24"/><w:jc w:val="left"/></w:pPr><w:r><w:t>名称
<w:p w14:paraId="1A447D31"><w:pPr><w:pStyle w:val="24"/><w:jc w:val="left"/></w:pPr><w:r><w:t>默认权重
<w:tblPrEx><w:tblCel
