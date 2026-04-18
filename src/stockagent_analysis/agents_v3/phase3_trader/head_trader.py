"""Head Trader - 把 investment_plan 转化为具体交易方案。"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from ...llm_client import LLMRouter
from ...prompts_v3 import load_prompt
from ..phase0_data import ReportBundle
from ..phase1_experts import ExpertResult
from ..phase2_debate import InvestmentPlan
from ..phase2_debate.bull_analyst import _format_experts

logger = logging.getLogger(__name__)


@dataclass
class TradingPlan:
    final_decision: str = "HOLD"
    entry_plans: list[dict[str, Any]] = field(default_factory=list)
    initial_position_ratio: float = 0.0
    time_horizon: str = "中线"
    primary_strategy: str = "回踩"
    hold_conditions: str = ""
    exit_conditions: str = ""
    reasoning: str = ""
    risk_alert: str = ""
    final_line: str = ""
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


def _extract_final_line(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"FINAL\s+TRANSACTION\s+PROPOSAL:\s*\**\s*(BUY|SELL|HOLD)", text, re.IGNORECASE)
    if m:
        return f"FINAL TRANSACTION PROPOSAL: **{m.group(1).upper()}**"
    return ""


def run_head_trader(
    bundle: ReportBundle,
    experts: dict[str, ExpertResult],
    investment_plan: InvestmentPlan,
    past_memories: str = "",
    provider: str = "grok",
    model_override: str | None = None,
) -> TradingPlan:
    """首席交易员根据 plan 产出具体交易方案。"""
    system = load_prompt("head_trader")
    plan_text = json.dumps(investment_plan.to_dict(), ensure_ascii=False, indent=2)

    parts = [
        system,
        "",
        "═══════════════════════════════",
        f"标的: {bundle.symbol} {bundle.name}",
        "═══════════════════════════════",
        "",
        "【研究主管的 investment_plan】",
        plan_text,
        "",
        "【专家分析师研判】",
        _format_experts(experts),
        "",
        "【客观量化报告(浓缩版)】",
        bundle.technical,
        "",
        bundle.structure,
    ]
    if past_memories:
        parts += ["", "【历史交易记忆】", past_memories]
    parts += ["", "请按 system prompt 要求输出 JSON + 末行 FINAL TRANSACTION PROPOSAL:"]
    prompt = "\n".join(parts)

    router = LLMRouter(
        provider=provider,
        temperature=0.25,
        max_tokens=2000,
        request_timeout_sec=90.0,
        model_override=model_override,
    )
    raw = ""
    try:
        raw = router._chat(prompt, multi_turn=True) or ""
    except Exception as e:
        logger.warning("[Trader] 调用失败: %s", e)

    parsed = _parse_json(raw) or {}
    tp = TradingPlan(raw_response=raw, final_line=_extract_final_line(raw))
    if parsed:
        tp.final_decision = str(parsed.get("final_decision", investment_plan.direction)).upper()
        entry_plans = parsed.get("entry_plans", [])
        if isinstance(entry_plans, list):
            tp.entry_plans = [p for p in entry_plans if isinstance(p, dict)]
        try:
            tp.initial_position_ratio = float(parsed.get("initial_position_ratio", 0.0))
        except (ValueError, TypeError):
            tp.initial_position_ratio = 0.0
        tp.time_horizon = str(parsed.get("time_horizon", "中线"))
        tp.primary_strategy = str(parsed.get("primary_strategy", "回踩"))
        tp.hold_conditions = str(parsed.get("hold_conditions", ""))
        tp.exit_conditions = str(parsed.get("exit_conditions", ""))
        tp.reasoning = str(parsed.get("reasoning", ""))
        tp.risk_alert = str(parsed.get("risk_alert", ""))
    else:
        tp.reasoning = raw[:400] or "(Trader 输出解析失败)"
        tp.final_decision = investment_plan.direction

    if not tp.final_line:
        tp.final_line = f"FINAL TRANSACTION PROPOSAL: **{tp.final_decision.upper()}**"
    return tp
