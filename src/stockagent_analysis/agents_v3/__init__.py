"""v3 LLM 角色化架构根模块。

设计: 从"因子打分"转向"角色辩论+流水线状态机"
参考: TradingAgents 项目
文档: docs/LLM-AGENT-REFACTOR-PLAN.md
基线: v2.9.0-prereform-baseline
"""
from .orchestrator_v3 import run_analysis_v3

__all__ = ["run_analysis_v3"]
