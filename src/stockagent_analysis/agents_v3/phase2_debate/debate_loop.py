"""Phase 2 辩论主循环: N 轮 Bull/Bear 交替 + Judge 仲裁。"""
from __future__ import annotations

import logging
from typing import Any

from ..phase0_data import ReportBundle
from ..phase1_experts import ExpertResult
from .bull_analyst import run_bull
from .bear_analyst import run_bear
from .research_judge import run_judge, InvestmentPlan

logger = logging.getLogger(__name__)


def run_investment_debate(
    bundle: ReportBundle,
    experts: dict[str, ExpertResult],
    rounds: int = 3,
    bull_provider: str = "grok",
    bear_provider: str = "grok",
    judge_provider: str = "grok",
    judge_model_override: str | None = None,
    past_memories: dict[str, str] | None = None,
) -> dict[str, Any]:
    """运行多空辩论 + 仲裁。

    Returns:
        {
          'transcript': str,                 # 完整辩论文本
          'bull_rounds': [str...],
          'bear_rounds': [str...],
          'investment_plan': InvestmentPlan,
        }
    """
    past_memories = past_memories or {}
    bull_mem = past_memories.get("bull", "")
    bear_mem = past_memories.get("bear", "")
    judge_mem = past_memories.get("judge", "")

    bull_rounds: list[str] = []
    bear_rounds: list[str] = []
    history_lines: list[str] = []
    bull_last = ""
    bear_last = ""

    for i in range(1, rounds + 1):
        # Bear 先发,Bull 反驳;也可以调换顺序
        bear_text = run_bear(
            bundle, experts, i, history="\n".join(history_lines),
            bull_last=bull_last, past_memories=bear_mem, provider=bear_provider,
        )
        bear_rounds.append(bear_text)
        history_lines.append(bear_text)

        bull_text = run_bull(
            bundle, experts, i, history="\n".join(history_lines),
            bear_last=bear_text, past_memories=bull_mem, provider=bull_provider,
        )
        bull_rounds.append(bull_text)
        history_lines.append(bull_text)

        bull_last = bull_text
        bear_last = bear_text

        logger.info("[Debate] 第 %d 轮完成 bull=%d字 bear=%d字",
                    i, len(bull_text), len(bear_text))

    transcript = "\n\n".join(history_lines)

    plan = run_judge(
        bundle, experts, transcript,
        past_memories=judge_mem,
        provider=judge_provider,
        model_override=judge_model_override,
    )

    return {
        "transcript": transcript,
        "bull_rounds": bull_rounds,
        "bear_rounds": bear_rounds,
        "investment_plan": plan,
    }
