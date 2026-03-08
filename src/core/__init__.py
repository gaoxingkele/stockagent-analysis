# -*- coding: utf-8 -*-
"""core — 多智能体 + 多大模型并行执行通用框架。

从 stockagent-analysis 项目中提取的领域无关基础设施，可供其他项目复用。

组件:
  - router.LLMRouter: 统一多 Provider LLM 路由（10+ Provider，含 Vision）
  - runner: 并行执行工具（超时控制 / 重试 / 断点续传 / 候选 Provider 切换）
  - progress: 终端进度渲染（流水线阶段指示 + Provider×Agent 表格化进度）
"""
from .router import LLMRouter, _supports_vision, _get_llm_proxies
from .runner import (
    ProviderProgress, ProviderResult,
    _timed_call, _create_fallback_router,
    _save_provider_result, _load_existing_results,
)
from .progress import PipelineTracker, ProviderProgressDisplay, AgentNameRegistry, run_display_loop

__all__ = [
    "LLMRouter", "_supports_vision", "_get_llm_proxies",
    "ProviderProgress", "ProviderResult",
    "_timed_call", "_create_fallback_router",
    "_save_provider_result", "_load_existing_results",
    "PipelineTracker", "ProviderProgressDisplay", "AgentNameRegistry", "run_display_loop",
]
