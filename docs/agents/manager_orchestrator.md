# manager_orchestrator

- 角色：Master Orchestrator
- 职责：统一调度、进度跟踪、一致性检测、辩论仲裁、加权评分、最终 JSON 输出
- 输入：股票代码、股票名称、所有子智能体提交结果
- 输出：`final_decision.json`，并写入 `manager_orchestrator.log`
- 辩论机制：Bull vs Bear + Judge 仲裁（按文档设定）
