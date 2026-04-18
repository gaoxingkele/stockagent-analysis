"""Research Judge - 多空仲裁,产出 investment_plan。"""
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
from .bull_analyst import _format_experts

logger = logging.getLogger(__name__)


@dataclass
class InvestmentPlan:
    direction: str = "HOLD"
    confidence: float = 0.5
    winner: str = "draw"
    key_reasons: list[str] = field(default_factory=list)
    winning_points: list[str] = field(default_factory=list)
    losing_counter: str = ""
    target_price_up: float | None = None
    target_price_down: float | None = None
    key_support: float | None = None
    key_resistance: float | None = None
    time_horizon: str = "中线"
    overall_score: float = 50.0
    judge_comment: str = ""
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


def run_judge(
    bundle: ReportBundle,
    experts: dict[str, ExpertResult],
    debate_transcript: str,
    past_memories: str = "",
    provider: str = "grok",
    model_override: str | None = None,
) -> InvestmentPlan:
    """Research Judge 仲裁并产出 investment_plan。

    建议 provider 使用能力更强的模型(opus/gpt-5/gemini-pro)。
    """
    system = load_prompt("research_judge")
    parts = [
        system,
        "",
        "═══════════════════════════════",
        f"标的: {bundle.symbol} {bundle.name}",
        "═══════════════════════════════",
        "",
        "【客观量化报告】",
        bundle.as_markdown(),
        "",
        "【专家分析师研判】",
        _format_experts(experts),
        "",
        "【多空辩论全文】",
        debate_transcript,
    ]
    if past_memories:
        parts += ["", "【历史类似仲裁的教训】", past_memories]
    parts += ["", "请按 system prompt 要求, 仅输出 investment_plan 的 JSON 对象:"]
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
        logger.warning("[Judge] 调用失败: %s", e)

    parsed = _parse_json(raw) or {}
    plan = InvestmentPlan(raw_response=raw)
    if parsed:
        plan.direction = str(parsed.get("direction", "HOLD")).upper()
        try:
            plan.confidence = float(parsed.get("confidence", 0.5))
        except (ValueError, TypeError):
            plan.confidence = 0.5
        plan.winner = str(parsed.get("winner", "draw"))
        plan.key_reasons = list(parsed.get("key_reasons", [])) if isinstance(parsed.get("key_reasons"), list) else []
        plan.winning_points = list(parsed.get("winning_points", [])) if isinstance(parsed.get("winning_points"), list) else []
        plan.losing_counter = str(parsed.get("losing_counter", ""))
        for k in ("target_price_up", "target_price_down", "key_support", "key_resistance"):
            v = parsed.get(k)
            try:
                setattr(plan, k, float(v) if v is not None else None)
            except (ValueError, TypeError):
                setattr(plan, k, None)
        plan.time_horizon = str(parsed.get("time_horizon", "中线"))
        try:
            plan.overall_score = float(parsed.get("overall_score", 50))
        except (ValueError, TypeError):
            plan.overall_score = 50.0
        plan.judge_comment = str(parsed.get("judge_comment", ""))
    else:
        plan.judge_comment = raw[:400] or "(Judge 输出解析失败)"

    return plan
