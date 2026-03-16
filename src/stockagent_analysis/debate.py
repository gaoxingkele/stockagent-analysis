# -*- coding: utf-8 -*-
"""v2 结构化辩论模块：团队汇报 → Bull/Bear投资辩论 → 仲裁 → 风险评估。

借鉴 TradingAgents-CN 的多轮辩论+风险三方评估机制。
使用原生 Python 实现，不引入 LangGraph。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from core.router import LLMRouter

logger = logging.getLogger(__name__)

# ── Agent 团队分组 ──
TEAM_TECH = {"trend_momentum_agent", "tech_quant_agent", "divergence_agent",
             "chanlun_agent", "pattern_agent", "volume_structure_agent", "resonance_agent"}
TEAM_CAPITAL = {"capital_liquidity_agent", "sentiment_flow_agent", "kline_vision_agent"}
TEAM_FUNDAMENTAL = {"fundamental_agent", "deriv_margin_agent"}

TEAM_NAMES = {
    "tech": ("技术面", TEAM_TECH),
    "capital": ("资金面", TEAM_CAPITAL),
    "fundamental": ("基本面", TEAM_FUNDAMENTAL),
}


@dataclass
class DebateResult:
    """辩论最终输出。"""
    decision: str = "hold"             # buy / hold / sell
    target_price: float | None = None
    stop_loss: float | None = None
    confidence: float = 0.5            # 0-1
    risk_score: float = 0.5            # 0-1, 越高风险越大
    score_override: float | None = None  # 如果辩论产出评分, 可覆盖加权平均
    reasoning: str = ""
    team_reports: dict[str, str] = field(default_factory=dict)
    debate_transcript: list[str] = field(default_factory=list)
    risk_assessment: str = ""


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象。"""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 直接尝试花括号
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def _safe_chat(router: LLMRouter, prompt: str, label: str) -> str:
    """安全调用LLM, 捕获异常。"""
    try:
        text = router._chat(prompt, multi_turn=True)
        return (text or "").strip()
    except Exception as e:
        logger.warning("[辩论] %s 调用失败: %s", label, e)
        return ""


# ────────────────────────────────────────────────
# Phase 1: 团队汇报
# ────────────────────────────────────────────────

def _build_team_report_prompt(
    team_name: str,
    agent_summaries: list[dict[str, Any]],
    symbol: str,
    name: str,
) -> str:
    """构建团队汇报 prompt。"""
    agent_lines = []
    for s in agent_summaries:
        agent_lines.append(
            f"  - {s['role']}({s['dim_code']}): 评分{s['score']:.1f} | {s['reason'][:100]}"
        )
    agents_block = "\n".join(agent_lines)
    return (
        f"你是{symbol} {name}的{team_name}分析团队负责人。\n"
        f"以下是你团队各分析师的独立评估结果:\n\n"
        f"{agents_block}\n\n"
        f"请综合团队意见，给出一份简洁的{team_name}分析报告(3-5句话)：\n"
        f"1. {team_name}的整体判断（偏多/中性/偏空）\n"
        f"2. 最关键的1-2个信号\n"
        f"3. 主要风险点\n\n"
        f"直接输出报告内容，不要输出JSON。"
    )


def generate_team_reports(
    router: LLMRouter,
    submissions: list[dict[str, Any]],
    symbol: str,
    name: str,
) -> dict[str, str]:
    """Phase 1: 生成3个团队汇报。"""
    reports = {}
    submissions_map = {s["agent_id"]: s for s in submissions}

    for team_key, (team_name, agent_ids) in TEAM_NAMES.items():
        team_subs = [submissions_map[aid] for aid in agent_ids if aid in submissions_map]
        if not team_subs:
            reports[team_key] = f"{team_name}：数据不足"
            continue
        prompt = _build_team_report_prompt(team_name, team_subs, symbol, name)
        text = _safe_chat(router, prompt, f"团队汇报-{team_name}")
        reports[team_key] = text or f"{team_name}：汇报生成失败"
        logger.info("[辩论] 团队汇报 %s 完成 (%d字符)", team_name, len(text))

    return reports


# ────────────────────────────────────────────────
# Phase 2: 投资辩论 (Bull vs Bear)
# ────────────────────────────────────────────────

def run_investment_debate(
    router: LLMRouter,
    team_reports: dict[str, str],
    symbol: str,
    name: str,
    current_price: float,
    rounds: int = 1,
) -> list[str]:
    """Phase 2: Bull/Bear 多轮辩论。返回辩论记录。"""
    reports_block = "\n".join(
        f"【{TEAM_NAMES[k][0]}报告】\n{v}" for k, v in team_reports.items()
    )
    transcript = []

    # Bull立论
    bull_prompt = (
        f"你是{symbol} {name}的看多分析师。基于以下三份团队分析报告，论证买入理由：\n\n"
        f"{reports_block}\n\n"
        f"当前价格: {current_price}\n\n"
        f"请给出2-3个最强买入论据，包含具体数据支撑。直接输出你的论点。"
    )
    bull_arg = _safe_chat(router, bull_prompt, "Bull立论")
    transcript.append(f"【看多方立论】\n{bull_arg}")

    # Bear反驳
    bear_prompt = (
        f"你是{symbol} {name}的看空分析师。\n"
        f"看多方论点如下:\n{bull_arg}\n\n"
        f"原始分析报告:\n{reports_block}\n\n"
        f"请针对看多论点逐一反驳，提出看空证据和风险因素。直接输出你的反驳。"
    )
    bear_arg = _safe_chat(router, bear_prompt, "Bear反驳")
    transcript.append(f"【看空方反驳】\n{bear_arg}")

    # 额外轮次
    for r in range(1, rounds):
        bull_rebuttal_prompt = (
            f"你是看多方。看空方的反驳如下:\n{bear_arg}\n\n"
            f"请回应看空方的质疑，强化你的买入论据。"
        )
        bull_rebuttal = _safe_chat(router, bull_rebuttal_prompt, f"Bull反驳R{r+1}")
        transcript.append(f"【看多方反驳 Round{r+1}】\n{bull_rebuttal}")

        bear_rebuttal_prompt = (
            f"你是看空方。看多方的回应如下:\n{bull_rebuttal}\n\n"
            f"请继续反驳，指出遗漏的风险。"
        )
        bear_rebuttal = _safe_chat(router, bear_rebuttal_prompt, f"Bear反驳R{r+1}")
        transcript.append(f"【看空方反驳 Round{r+1}】\n{bear_rebuttal}")
        bear_arg = bear_rebuttal  # 用于下一轮

    return transcript


# ────────────────────────────────────────────────
# Phase 3: 仲裁
# ────────────────────────────────────────────────

def run_arbitration(
    router: LLMRouter,
    team_reports: dict[str, str],
    debate_transcript: list[str],
    symbol: str,
    name: str,
    current_price: float,
) -> dict:
    """Phase 3: 仲裁者综合辩论, 给出投资计划。"""
    reports_block = "\n".join(
        f"【{TEAM_NAMES[k][0]}报告】\n{v}" for k, v in team_reports.items()
    )
    debate_block = "\n\n".join(debate_transcript)

    prompt = (
        f"你是{symbol} {name}的投资委员会主席。综合以下分析报告和辩论内容，做出最终投资决策。\n\n"
        f"当前价格: {current_price}\n\n"
        f"【团队分析报告】\n{reports_block}\n\n"
        f"【投资辩论】\n{debate_block}\n\n"
        f"请输出JSON格式的投资计划:\n"
        f'{{"decision": "buy/hold/sell",'
        f' "score": 0到100的整数,'
        f' "target_price": 目标价格,'
        f' "stop_loss": 止损价格,'
        f' "confidence": 0到1的置信度,'
        f' "reasoning": "2-3句核心理由"}}\n\n'
        f"必须给出具体的 target_price 和 stop_loss 数值。"
    )
    text = _safe_chat(router, prompt, "仲裁决策")
    result = _extract_json(text)
    if not result:
        result = {"decision": "hold", "score": 50, "reasoning": text[:200] if text else "仲裁失败"}
    logger.info("[辩论] 仲裁完成: decision=%s score=%s", result.get("decision"), result.get("score"))
    return result


# ────────────────────────────────────────────────
# Phase 4: 风险评估 (三方辩论)
# ────────────────────────────────────────────────

def run_risk_assessment(
    router: LLMRouter,
    arbitration_result: dict,
    team_reports: dict[str, str],
    symbol: str,
    name: str,
) -> dict:
    """Phase 4: 激进/保守/中立三方风险评估 → 风险经理最终决策。"""
    arb_block = json.dumps(arbitration_result, ensure_ascii=False, indent=2)
    reports_block = "\n".join(
        f"【{TEAM_NAMES[k][0]}】{v[:150]}" for k, v in team_reports.items()
    )

    # 激进派
    aggressive_prompt = (
        f"你是激进投资分析师。以下是{symbol} {name}的投资计划:\n{arb_block}\n\n"
        f"请从激进角度评估: 目标价是否过于保守？是否有上行空间被忽略？"
        f"给出你的调整建议(2-3句)。"
    )
    aggressive = _safe_chat(router, aggressive_prompt, "激进派")

    # 保守派
    conservative_prompt = (
        f"你是保守投资分析师。以下是{symbol} {name}的投资计划:\n{arb_block}\n\n"
        f"请从保守角度评估: 止损位是否合理？风险是否被低估？"
        f"给出你的调整建议(2-3句)。"
    )
    conservative = _safe_chat(router, conservative_prompt, "保守派")

    # 风险经理仲裁
    risk_prompt = (
        f"你是{symbol} {name}的风险经理。综合以下三方意见做出最终风险评估:\n\n"
        f"【投资计划】\n{arb_block}\n\n"
        f"【激进派意见】\n{aggressive}\n\n"
        f"【保守派意见】\n{conservative}\n\n"
        f"请输出JSON格式:\n"
        f'{{"decision": "buy/hold/sell",'
        f' "score": 0到100的整数(风险调整后),'
        f' "risk_score": 0到1(越高风险越大),'
        f' "target_price": 风险调整后目标价,'
        f' "stop_loss": 风险调整后止损价,'
        f' "reasoning": "2-3句核心理由"}}'
    )
    text = _safe_chat(router, risk_prompt, "风险经理")
    result = _extract_json(text)
    if not result:
        result = arbitration_result.copy()
        result["risk_score"] = 0.5
    result["_aggressive"] = aggressive
    result["_conservative"] = conservative
    logger.info("[辩论] 风险评估完成: decision=%s risk=%s", result.get("decision"), result.get("risk_score"))
    return result


# ────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────

def run_structured_debate(
    router: LLMRouter,
    submissions: list[dict[str, Any]],
    symbol: str,
    name: str,
    current_price: float,
    debate_rounds: int = 1,
) -> DebateResult:
    """运行完整结构化辩论流程。

    Args:
        router: 深度推理模型的 LLMRouter (建议用 deep_model)
        submissions: Agent结果列表, 每项包含 agent_id, role, dim_code, score, reason
        symbol: 股票代码
        name: 股票名称
        current_price: 当前价格
        debate_rounds: 辩论轮次 (默认1轮)

    Returns:
        DebateResult 包含最终决策、辩论记录等
    """
    result = DebateResult()

    # Phase 1: 团队汇报
    logger.info("[辩论] Phase 1: 团队汇报 (%s %s)", symbol, name)
    team_reports = generate_team_reports(router, submissions, symbol, name)
    result.team_reports = team_reports

    # Phase 2: 投资辩论
    logger.info("[辩论] Phase 2: 投资辩论 (轮次=%d)", debate_rounds)
    transcript = run_investment_debate(
        router, team_reports, symbol, name, current_price, rounds=debate_rounds
    )
    result.debate_transcript = transcript

    # Phase 3: 仲裁
    logger.info("[辩论] Phase 3: 仲裁")
    arb_result = run_arbitration(
        router, team_reports, transcript, symbol, name, current_price
    )

    # Phase 4: 风险评估
    logger.info("[辩论] Phase 4: 风险评估")
    risk_result = run_risk_assessment(
        router, arb_result, team_reports, symbol, name
    )

    # 填充最终结果
    result.decision = risk_result.get("decision", arb_result.get("decision", "hold"))
    result.score_override = risk_result.get("score") or arb_result.get("score")
    result.target_price = risk_result.get("target_price") or arb_result.get("target_price")
    result.stop_loss = risk_result.get("stop_loss") or arb_result.get("stop_loss")
    result.confidence = float(risk_result.get("confidence", arb_result.get("confidence", 0.5)))
    result.risk_score = float(risk_result.get("risk_score", 0.5))
    result.reasoning = risk_result.get("reasoning", arb_result.get("reasoning", ""))
    result.risk_assessment = (
        f"激进派: {risk_result.get('_aggressive', '')[:150]}\n"
        f"保守派: {risk_result.get('_conservative', '')[:150]}"
    )

    logger.info(
        "[辩论] 完成: decision=%s score=%s target=%s stop=%s confidence=%.2f risk=%.2f",
        result.decision, result.score_override, result.target_price,
        result.stop_loss, result.confidence, result.risk_score,
    )
    return result
