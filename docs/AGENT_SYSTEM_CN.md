# 多智能体关系与角色定义（中国股市，19-Agent版）

## 1. 共同任务

所有智能体共享同一任务：分析一只中国股票，经过辩论与投票，给出最终决策：

- `buy`（买入）
- `hold`（观望）
- `sell`（卖出）

## 2. 组织关系

- **主控层（1）**
  - `manager_orchestrator`：统一调度、进度追踪、一致性检测、辩论仲裁、加权评分
- **核心层（9，必选）**
  - `trend_agent`、`tech_indicator_agent`、`deriv_margin_agent`、`liquidity_agent`
  - `capital_flow_agent`、`sector_policy_agent`、`beta_agent`、`sentiment_agent`、`fundamental_agent`
- **专家层（10，可选）**
  - `quant_agent`、`macro_agent`、`industry_agent`、`flow_detail_agent`、`mm_behavior_agent`
  - `arb_agent`、`shareholder_agent`、`quality_agent`、`regulation_agent`、`nlp_sentiment_agent`

## 3. 协作流程

1. manager 下发任务（股票代码、名称、分析轮次）
2. 每个分析智能体拉取数据并形成初步结论
3. 辩论阶段：
   - 采用 Bull vs Bear + Judge 仲裁机制
   - 智能体间以 JSON 消息互相质询/补充
   - manager 广播仲裁意见
4. 投票阶段：
   - 各智能体提交 JSON 结论到 `submissions/`
   - manager 按权重聚合票值
5. manager 输出最终决策 JSON

## 4. 文件约定

- 智能体日志：`output/runs/<run_id>/logs/<agent_id>.log`
- 智能体消息：`output/runs/<run_id>/messages/*.json`
- 智能体提交：`output/runs/<run_id>/submissions/<agent_id>.json`
- 最终结果：`output/runs/<run_id>/final_decision.json`

## 5. 数据源策略

- 全局默认：`AKShare + Tushare` 组合
- 可切换：
  - 单源模式：只用 AKShare 或只用 Tushare
  - 组合模式：按优先级和回退顺序联合使用
- 可定制：
  - 每个智能体在自身配置中声明 `data_sources`（含 primary/fallback/required_data_points）
  - 若智能体需要专用数据源，可扩展至 `special_data_sources`
