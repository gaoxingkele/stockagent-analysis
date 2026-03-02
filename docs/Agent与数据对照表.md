# Agent 与本地/云端数据对照表

调用大模型前，系统会检查每个 Agent 所需的 `required_data_points` 是否从本地数据提供。

## 数据来源说明

| 类型 | 说明 |
|------|------|
| **本地** | 从 Tushare/AKShare 获取并落地的数据（snapshot、fundamentals、news、kline、features） |
| **云端** | 大模型基于本地传入的 data_context 进行研判，不主动拉取外部数据 |

## Agent 与数据映射

| Agent | required_data_points | 本地数据来源 |
|-------|----------------------|--------------|
| trend_agent | 日/周/月K线, HH/HL/LH/LL结构点 | kline_indicators, features.trend_strength |
| tech_indicator_agent | RSI, MACD, KDJ, 布林带, StochRSI, 波动率, 3-5根K线组合, 长期趋势线, MA5-250 | kline_indicators, historical_daily |
| fundamental_agent | 营收利润, ROE/毛利率, 估值PE/PB, 资产负债结构 | fundamentals, snapshot |
| capital_flow_agent | 主力净流入, 超大单/大单, 筹码峰, 北向资金 | features, fundamentals, news |
| sentiment_agent | 新闻热度, 社媒情绪, 公告舆情, 事件冲击 | news, features.news_sentiment |
| macro_agent | 利率, 汇率, 大宗商品, 宏观事件 | fundamentals, news |
| beta_agent | 沪深300相关性, 行业Beta, 指数回归系数 | features, kline_indicators |
| quant_agent | 因子暴露, 回测片段, 短中期信号 | features, historical_daily |
| regulation_agent | 问询函, 处罚公告, 合规事件, 政策约束 | news |
| shareholder_agent | 股东户数, 前十大股东, 机构持仓变化 | fundamentals |
| mm_behavior_agent | 盘口异动, 拉升回落模式, 主力控盘迹象 | snapshot, features |
| flow_detail_agent | 分时资金流, 主力大单轨迹, 异常成交 | snapshot, features |
| deriv_margin_agent | 融资余额, 融券余额, 融资净买入 | fundamentals, news |
| nlp_sentiment_agent | 新闻文本, 社媒文本, 主题情绪网络 | news, features.news_sentiment |
| sector_policy_agent | 行业指数, 政策公告, 监管新闻, 板块轮动 | fundamentals, news |
| arb_agent | 大宗交易折溢价, 融券套利窗口, 价差结构 | fundamentals, kline_indicators |
| industry_agent | 行业景气度, 上下游价格, 供需结构 | fundamentals, news |
| quality_agent | 盈利质量, 现金流质量, 护城河指标 | fundamentals |
| liquidity_agent | 成交额, 换手率, 盘口深度, 委比量比 | snapshot, fundamentals, features |

## 运行输出

每次分析完成后，对照表会：
1. 打印到控制台
2. 保存至 `output/runs/<run_id>/data/agent_data_mapping.json`

符号说明：
- ✓ 本地：该数据点由本地提供
- ✗ 缺失：期望的本地数据未就绪
- ○ 云端：无本地映射，依赖大模型知识
