# -*- coding: utf-8 -*-
"""进度渲染（向后兼容包装）——实际实现已移至 core.progress。

所有旧的 ``from .progress_display import PipelineTracker`` 仍然有效。
"""
from core.progress import (  # noqa: F401 — re-export
    _is_interactive_terminal,
    _display_width,
    _truncate_to_width,
    _pad_to_width,
    _center_in_width,
    AgentNameRegistry,
    _PIPELINE_STAGES,
    _STAGE_DONE,
    _STAGE_ACTIVE,
    _STAGE_PENDING,
    PipelineTracker,
    ProviderProgressDisplay,
    run_display_loop,
)
