"""风控三辩手: Aggressive / Conservative / Neutral。"""
from __future__ import annotations

import json
import logging

from ...llm_client import LLMRouter
from ...prompts_v3 import load_prompt
from ..phase0_data import ReportBundle
from ..phase2_debate import InvestmentPlan
from ..phase3_trader import TradingPlan

logger = logging.getLogger(__name__)


def _base_prompt(
    system_key: str,
    bundle: ReportBundle,
    investment_plan: InvestmentPlan,
    trader_plan: TradingPlan,
    round_idx: int,
    history: str,
    other_last: dict[str, str],
    past_memories: str,
) -> str:
    system = load_prompt(system_key)
    parts = [
        system,
        "",
        "═══════════════════════════════",
        f"标的: {bundle.symbol} {bundle.name}  |  风控辩论第 {round_idx} 轮",
        "═══════════════════════════════",
        "",
        "【研究主管 investment_plan】",
        json.dumps(investment_plan.to_dict(), ensure_ascii=False, indent=2),
        "",
        "【首席交易员 trader_plan】",
        json.dumps(trader_plan.to_dict(), ensure_ascii=False, indent=2),
        "",
        "【客观量化报告(技术+资金)】",
        bundle.technical,
        "",
        bundle.capital,
    ]
    if past_memories:
        parts += ["", "【历史风控记忆】", past_memories]
    if history:
        parts += ["", "【本场辩论历史】", history]
    if other_last:
        for k, v in other_last.items():
            if v:
                parts += ["", f"【{k}上一轮论据】", v]
    parts += ["", "请输出你的辩论立场(不要 JSON, 文本 200-350 字):"]
    return "\n".join(parts)


def _run_debator(
    system_key: str,
    tag: str,
    bundle: ReportBundle,
    investment_plan: InvestmentPlan,
    trader_plan: TradingPlan,
    round_idx: int,
    history: str,
    other_last: dict[str, str],
    past_memories: str,
    provider: str,
) -> str:
    prompt = _base_prompt(system_key, bundle, investment_plan, trader_plan,
                          round_idx, history, other_last, past_memories)
    router = LLMRouter(provider=provider, temperature=0.5, max_tokens=1000, request_timeout_sec=60.0)
    try:
        text = router._chat(prompt, multi_turn=True) or ""
    except Exception as e:
        logger.warning("[%s] 调用失败: %s", tag, e)
        text = f"({tag} 角色本轮失败: {e})"
    return f"[{tag} {round_idx}] {text.strip()}"


def run_aggressive(bundle, investment_plan, trader_plan, round_idx, history,
                   other_last=None, past_memories="", provider="grok") -> str:
    return _run_debator("risk_aggressive", "Aggressive", bundle, investment_plan,
                        trader_plan, round_idx, history, other_last or {}, past_memories, provider)


def run_conservative(bundle, investment_plan, trader_plan, round_idx, history,
                     other_last=None, past_memories="", provider="grok") -> str:
    return _run_debator("risk_conservative", "Conservative", bundle, investment_plan,
                        trader_plan, round_idx, history, other_last or {}, past_memories, provider)


def run_neutral(bundle, investment_plan, trader_plan, round_idx, history,
                other_last=None, past_memories="", provider="grok") -> str:
    return _run_debator("risk_neutral", "Neutral", bundle, investment_plan,
                        trader_plan, round_idx, history, other_last or {}, past_memories, provider)
