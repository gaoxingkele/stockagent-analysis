"""Bear Analyst - 看空辩手。"""
from __future__ import annotations

import logging

from ...llm_client import LLMRouter
from ...prompts_v3 import load_prompt
from ..phase0_data import ReportBundle
from ..phase1_experts import ExpertResult
from .bull_analyst import _format_experts

logger = logging.getLogger(__name__)


def run_bear(
    bundle: ReportBundle,
    experts: dict[str, ExpertResult],
    round_idx: int,
    history: str,
    bull_last: str,
    past_memories: str = "",
    provider: str = "grok",
) -> str:
    """Bear 发言一轮。"""
    system = load_prompt("bear_analyst")
    experts_text = _format_experts(experts)
    parts = [
        system,
        "",
        "═══════════════════════════════",
        f"标的: {bundle.symbol} {bundle.name}  |  当前第 {round_idx} 轮辩论",
        "═══════════════════════════════",
        "",
        "【客观量化报告】",
        bundle.as_markdown(),
        "",
        "【专家分析师研判】",
        experts_text,
    ]
    if past_memories:
        parts += ["", "【历史类似情境的教训】", past_memories]
    if history:
        parts += ["", "【辩论历史(本场)】", history]
    if bull_last:
        parts += ["", "【多方上一轮论据】", bull_last, "", "请针对性反驳并构建你的新一轮看空论据。"]
    else:
        parts += ["", "请基于以上材料,直接构建你的首发看空论据。"]

    prompt = "\n".join(parts)
    router = LLMRouter(provider=provider, temperature=0.5, max_tokens=1500, request_timeout_sec=60.0)
    try:
        text = router._chat(prompt, multi_turn=True) or ""
    except Exception as e:
        logger.warning("[Bear] 调用失败: %s", e)
        text = f"(Bear 角色本轮失败: {e})"
    return f"[Bear {round_idx}] {text.strip()}"
