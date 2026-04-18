"""Phase 4 风控辩论主循环: N 轮三方辩论 + PM 拍板。"""
from __future__ import annotations

import logging
from typing import Any

from ..phase0_data import ReportBundle
from ..phase2_debate import InvestmentPlan
from ..phase3_trader import TradingPlan
from .risk_debators import run_aggressive, run_conservative, run_neutral
from .portfolio_manager import run_portfolio_manager, RiskPolicy

logger = logging.getLogger(__name__)


def run_risk_debate(
    bundle: ReportBundle,
    investment_plan: InvestmentPlan,
    trader_plan: TradingPlan,
    rounds: int = 2,
    aggressive_provider: str = "grok",
    conservative_provider: str = "grok",
    neutral_provider: str = "grok",
    pm_provider: str = "grok",
    pm_model_override: str | None = None,
    past_memories: dict[str, str] | None = None,
) -> dict[str, Any]:
    """三方辩论 + PM 拍板。

    Returns:
        {
          'transcript': str,
          'aggressive_rounds': [...],
          'conservative_rounds': [...],
          'neutral_rounds': [...],
          'risk_policy': RiskPolicy,
        }
    """
    past_memories = past_memories or {}
    agg_mem = past_memories.get("aggressive", "")
    con_mem = past_memories.get("conservative", "")
    neu_mem = past_memories.get("neutral", "")
    pm_mem = past_memories.get("pm", "")

    aggressive_rounds: list[str] = []
    conservative_rounds: list[str] = []
    neutral_rounds: list[str] = []
    history_lines: list[str] = []
    last_agg = ""
    last_con = ""
    last_neu = ""

    for i in range(1, rounds + 1):
        # Conservative 先说, Aggressive 反驳, Neutral 最后平衡
        con_text = run_conservative(
            bundle, investment_plan, trader_plan, i, "\n".join(history_lines),
            other_last={"Aggressive": last_agg, "Neutral": last_neu},
            past_memories=con_mem, provider=conservative_provider,
        )
        conservative_rounds.append(con_text)
        history_lines.append(con_text)

        agg_text = run_aggressive(
            bundle, investment_plan, trader_plan, i, "\n".join(history_lines),
            other_last={"Conservative": con_text, "Neutral": last_neu},
            past_memories=agg_mem, provider=aggressive_provider,
        )
        aggressive_rounds.append(agg_text)
        history_lines.append(agg_text)

        neu_text = run_neutral(
            bundle, investment_plan, trader_plan, i, "\n".join(history_lines),
            other_last={"Conservative": con_text, "Aggressive": agg_text},
            past_memories=neu_mem, provider=neutral_provider,
        )
        neutral_rounds.append(neu_text)
        history_lines.append(neu_text)

        last_agg = agg_text
        last_con = con_text
        last_neu = neu_text

        logger.info("[RiskDebate] 第 %d 轮完成", i)

    transcript = "\n\n".join(history_lines)

    policy = run_portfolio_manager(
        bundle, investment_plan, trader_plan, transcript,
        past_memories=pm_mem,
        provider=pm_provider,
        model_override=pm_model_override,
    )

    return {
        "transcript": transcript,
        "aggressive_rounds": aggressive_rounds,
        "conservative_rounds": conservative_rounds,
        "neutral_rounds": neutral_rounds,
        "risk_policy": policy,
    }
