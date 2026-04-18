"""Portfolio Manager - 风控主管, 综合三方辩论拍板 RiskPolicy。"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from ...llm_client import LLMRouter
from ...prompts_v3 import load_prompt
from ..phase0_data import ReportBundle
from ..phase2_debate import InvestmentPlan
from ..phase3_trader import TradingPlan

logger = logging.getLogger(__name__)


@dataclass
class RiskPolicy:
    final_risk_rating: str = "中"
    max_position_ratio: float = 0.5
    initial_position_ratio: float = 0.3
    stop_loss_discipline: dict[str, Any] = field(default_factory=dict)
    take_profit_rule: dict[str, Any] = field(default_factory=dict)
    add_position_condition: str = ""
    reduce_position_condition: str = ""
    black_swan_response: str = ""
    alignment_with: str = "neutral"
    override_comment: str = ""
    pm_summary: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s)
    if m:
        s = m.group(1)
    start = s.find("{")
    if start >= 0:
        depth, end = 0, -1
        for i in range(start, len(s)):
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > 0:
            s = s[start:end + 1]
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def run_portfolio_manager(
    bundle: ReportBundle,
    investment_plan: InvestmentPlan,
    trader_plan: TradingPlan,
    risk_transcript: str,
    past_memories: str = "",
    provider: str = "grok",
    model_override: str | None = None,
) -> RiskPolicy:
    system = load_prompt("portfolio_manager")
    parts = [
        system,
        "",
        "═══════════════════════════════",
        f"标的: {bundle.symbol} {bundle.name}",
        "═══════════════════════════════",
        "",
        "【investment_plan】",
        json.dumps(investment_plan.to_dict(), ensure_ascii=False, indent=2),
        "",
        "【trader_plan】",
        json.dumps(trader_plan.to_dict(), ensure_ascii=False, indent=2),
        "",
        "【风控三辩论全文】",
        risk_transcript,
    ]
    if past_memories:
        parts += ["", "【历史风控记忆】", past_memories]
    parts += ["", "请按 system prompt 要求, 仅输出 risk_policy JSON 对象:"]
    prompt = "\n".join(parts)

    router = LLMRouter(
        provider=provider,
        temperature=0.2,
        max_tokens=1500,
        request_timeout_sec=90.0,
        model_override=model_override,
    )
    raw = ""
    try:
        raw = router._chat(prompt, multi_turn=True) or ""
    except Exception as e:
        logger.warning("[PortfolioManager] 调用失败: %s", e)

    parsed = _parse_json(raw) or {}
    policy = RiskPolicy(raw_response=raw)
    if parsed:
        policy.final_risk_rating = str(parsed.get("final_risk_rating", "中"))
        try:
            policy.max_position_ratio = float(parsed.get("max_position_ratio", 0.5))
        except (ValueError, TypeError):
            policy.max_position_ratio = 0.5
        try:
            policy.initial_position_ratio = float(parsed.get("initial_position_ratio", 0.3))
        except (ValueError, TypeError):
            policy.initial_position_ratio = 0.3
        sl = parsed.get("stop_loss_discipline")
        if isinstance(sl, dict):
            policy.stop_loss_discipline = sl
        tp = parsed.get("take_profit_rule")
        if isinstance(tp, dict):
            policy.take_profit_rule = tp
        policy.add_position_condition = str(parsed.get("add_position_condition", ""))
        policy.reduce_position_condition = str(parsed.get("reduce_position_condition", ""))
        policy.black_swan_response = str(parsed.get("black_swan_response", ""))
        policy.alignment_with = str(parsed.get("alignment_with", "neutral"))
        policy.override_comment = str(parsed.get("override_comment", ""))
        policy.pm_summary = str(parsed.get("pm_summary", ""))
    else:
        policy.pm_summary = raw[:400] or "(PM 输出解析失败)"
    return policy
