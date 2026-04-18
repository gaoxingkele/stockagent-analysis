"""Bull Analyst - 看多辩手。"""
from __future__ import annotations

import logging
from typing import Any

from ...llm_client import LLMRouter
from ...prompts_v3 import load_prompt
from ..phase0_data import ReportBundle
from ..phase1_experts import ExpertResult

logger = logging.getLogger(__name__)


def _format_experts(experts: dict[str, ExpertResult]) -> str:
    lines = []
    for role, er in experts.items():
        lines.append(f"### {er.role_cn} ({er.provider})")
        lines.append(f"- 评分: {er.score:.1f}")
        lines.append(f"- 研判: {er.analysis}")
        if er.risk:
            lines.append(f"- 风险: {er.risk}")
        if er.key_data:
            lines.append(f"- 关键输出: {er.key_data}")
        lines.append("")
    return "\n".join(lines)


def run_bull(
    bundle: ReportBundle,
    experts: dict[str, ExpertResult],
    round_idx: int,
    history: str,
    bear_last: str,
    past_memories: str = "",
    provider: str = "grok",
) -> str:
    """Bull 发言一轮。"""
    system = load_prompt("bull_analyst")
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
    if bear_last:
        parts += ["", "【空方上一轮论据】", bear_last, "", "请针对性反驳并构建你的新一轮看多论据。"]
    else:
        parts += ["", "请基于以上材料,直接构建你的首发看多论据。"]

    prompt = "\n".join(parts)
    router = LLMRouter(provider=provider, temperature=0.5, max_tokens=1500, request_timeout_sec=60.0)
    try:
        text = router._chat(prompt, multi_turn=True) or ""
    except Exception as e:
        logger.warning("[Bull] 调用失败: %s", e)
        text = f"(Bull 角色本轮失败: {e})"
    return f"[Bull {round_idx}] {text.strip()}"
