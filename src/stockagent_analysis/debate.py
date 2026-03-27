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
    plans: list[dict] = field(default_factory=list)  # A/B/C 三入场方案
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


def _safe_chat_multi_agent(router: LLMRouter, prompt: str, label: str) -> str:
    """通过 Grok Multi-Agent API (/v1/responses) 调用，16路推理线程并行分析。"""
    try:
        sys_msg = "你是一位专业的中国股市投资委员会主席，请基于团队分析和辩论内容做出客观决策。"
        text = router.chat_multi_agent(prompt, system_message=sys_msg)
        return (text or "").strip()
    except Exception as e:
        logger.warning("[辩论] %s multi-agent调用失败: %s", label, e)
        return ""


def _chat_for_json(
    routers: list[LLMRouter],
    prompt: str,
    label: str,
    max_retries: int = 2,
    use_multi_agent: bool = False,
) -> dict:
    """调用LLM获取JSON结果，失败时重试同一provider，再切换备用provider。
    use_multi_agent=True时，优先尝试 Grok Multi-Agent API，失败后降级到普通模式。
    """
    # ── Multi-Agent 优先尝试 ──
    if use_multi_agent and routers:
        grok_router = next((r for r in routers if getattr(r, "provider", "") == "grok"), routers[0])
        for attempt in range(2):
            text = _safe_chat_multi_agent(grok_router, prompt, f"{label}(multi-agent #{attempt+1})")
            if not text:
                continue
            result = _extract_json(text)
            if result and result.get("decision"):
                logger.info("[辩论] %s multi-agent JSON解析成功 (attempt=%d)", label, attempt + 1)
                return result
            logger.warning("[辩论] %s multi-agent JSON解析失败 (attempt=%d), 原文: %s",
                           label, attempt + 1, text[:200])
        logger.info("[辩论] %s multi-agent 失败, 降级到普通模式", label)

    # ── 普通模式：逐个 provider 重试 ──
    for idx, router in enumerate(routers):
        provider_name = getattr(router, "provider", f"router_{idx}")
        for attempt in range(max_retries):
            text = _safe_chat(router, prompt, f"{label}({provider_name} #{attempt+1})")
            if not text:
                continue
            result = _extract_json(text)
            if result and result.get("decision"):
                logger.info("[辩论] %s JSON解析成功 (provider=%s attempt=%d)", label, provider_name, attempt + 1)
                return result
            logger.warning("[辩论] %s JSON解析失败 (provider=%s attempt=%d), 原文: %s",
                           label, provider_name, attempt + 1, text[:200])
        logger.info("[辩论] %s provider=%s 重试耗尽, 尝试下一个", label, provider_name)
    return {}


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
    fibonacci: dict | None = None,
    atr: float | None = None,
    fallback_routers: list[LLMRouter] | None = None,
    use_multi_agent: bool = False,
) -> dict:
    """Phase 3: 仲裁者综合辩论, 给出投资计划。
    use_multi_agent=True 时优先用 Grok Multi-Agent (16路推理) 做仲裁。"""
    reports_block = "\n".join(
        f"【{TEAM_NAMES[k][0]}报告】\n{v}" for k, v in team_reports.items()
    )
    debate_block = "\n\n".join(debate_transcript)

    # 斐波那契锚定参考
    fib_block = ""
    if fibonacci and fibonacci.get("ok"):
        direction = "上涨回调" if fibonacci.get("uptrend") else "下跌反弹"
        fib_block = (
            f"\n【斐波那契参考（{direction}）】\n"
            f"近期高点: {fibonacci.get('swing_high')}  低点: {fibonacci.get('swing_low')}\n"
            f"回调位: 23.6%={fibonacci.get('retrace_236')} | 38.2%={fibonacci.get('retrace_382')} | "
            f"50%={fibonacci.get('retrace_500')} | 61.8%={fibonacci.get('retrace_618')}\n"
            f"延伸位: 127.2%={fibonacci.get('extend_1272')} | 161.8%={fibonacci.get('extend_1618')}\n"
        )

    # ATR 动态止损参考
    atr_block = ""
    if atr and current_price:
        atr_stop = round(current_price - 2 * atr, 2)
        atr_pct = round(2 * atr / current_price * 100, 1)
        atr_block = f"\n【ATR动态止损参考】ATR={atr:.2f}, 2倍ATR止损={atr_stop}（距当前价-{atr_pct}%）\n"

    prompt = (
        f"你是{symbol} {name}的投资委员会主席。综合以下分析报告和辩论内容，做出最终投资决策。\n\n"
        f"当前价格: {current_price}\n\n"
        f"【团队分析报告】\n{reports_block}\n\n"
        f"【投资辩论】\n{debate_block}\n"
        f"{fib_block}{atr_block}\n"
        f"请输出JSON格式的投资计划（含3种入场方案）:\n"
        f'{{"decision": "buy/hold/sell",'
        f' "score": 0到100的投资价值分(越高越看好,70+买入,50以下卖出,与decision方向一致),'
        f' "target_price": 目标价格,'
        f' "stop_loss": 止损价格,'
        f' "confidence": 0到1的置信度,'
        f' "reasoning": "2-3句核心理由",'
        f' "plans": ['
        f'  {{"name": "追涨", "entry": 入场价, "target": 目标价, "stop": 止损价, "rr": 风险收益比数值}},'
        f'  {{"name": "回踩", "entry": 回踩买入价, "target": 目标价, "stop": 止损价, "rr": 风险收益比数值}},'
        f'  {{"name": "确认", "entry": 确认突破后买入价, "target": 目标价, "stop": 止损价, "rr": 风险收益比数值}}'
        f' ]}}\n\n'
        f"必须给出具体的 target_price 和 stop_loss 数值。plans中每个方案的止损价可参考ATR或斐波那契位。\n"
        f"【盈亏比硬约束】(target_price - 当前价) >= 2 * (当前价 - stop_loss)，止损距当前价不超过10%。\n"
        f"注意：只输出JSON，不要添加其他内容。"
    )
    routers = [router] + (fallback_routers or [])
    result = _chat_for_json(routers, prompt, "仲裁决策", use_multi_agent=use_multi_agent)
    if not result:
        result = {"decision": "hold", "score": 50, "reasoning": "仲裁JSON解析全部失败"}
        logger.warning("[辩论] 仲裁所有provider均解析失败, 使用默认 hold/50")
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
    fallback_routers: list[LLMRouter] | None = None,
    use_multi_agent: bool = False,
) -> dict:
    """Phase 4: 激进/保守/中立三方风险评估 → 风险经理最终决策。JSON解析失败时重试+切换provider。"""
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

    # 风险经理仲裁 — 使用重试+切换provider
    risk_prompt = (
        f"你是{symbol} {name}的风险经理。综合以下三方意见做出最终风险评估:\n\n"
        f"【投资计划】\n{arb_block}\n\n"
        f"【激进派意见】\n{aggressive}\n\n"
        f"【保守派意见】\n{conservative}\n\n"
        f"请输出JSON格式:\n"
        f'{{"decision": "buy/hold/sell",'
        f' "score": 0到100的投资价值分(越高越看好,70+买入,50以下卖出,与decision方向一致),'
        f' "risk_score": 0到1(越高风险越大),'
        f' "target_price": 风险调整后目标价,'
        f' "stop_loss": 风险调整后止损价,'
        f' "reasoning": "2-3句核心理由"}}\n'
        f"【盈亏比硬约束】(target_price - 当前价) >= 2 * (当前价 - stop_loss)，止损距当前价不超过10%。\n"
        f"注意：只输出JSON，不要添加其他内容。"
    )
    routers = [router] + (fallback_routers or [])
    result = _chat_for_json(routers, risk_prompt, "风险经理", use_multi_agent=use_multi_agent)
    if not result:
        result = arbitration_result.copy()
        result["risk_score"] = 0.5
        logger.warning("[辩论] 风险评估所有provider均解析失败, 继承仲裁结果")
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
    fallback_routers: list[LLMRouter] | None = None,
    use_multi_agent: bool = False,
    fibonacci: dict | None = None,
    atr: float | None = None,
) -> DebateResult:
    """运行完整结构化辩论流程。

    Args:
        router: 深度推理模型的 LLMRouter (建议用 deep_model)
        submissions: Agent结果列表, 每项包含 agent_id, role, dim_code, score, reason
        symbol: 股票代码
        name: 股票名称
        current_price: 当前价格
        debate_rounds: 辩论轮次 (默认1轮)
        fallback_routers: 备用LLMRouter列表, JSON解析失败时依次切换
        use_multi_agent: 仲裁阶段使用 Grok Multi-Agent (16路推理并行)

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

    # Phase 3: 仲裁 (multi-agent 或 普通模式)
    _mode = "multi-agent" if use_multi_agent else "普通"
    logger.info("[辩论] Phase 3: 仲裁 (模式=%s)", _mode)
    arb_result = run_arbitration(
        router, team_reports, transcript, symbol, name, current_price,
        fibonacci=fibonacci, atr=atr,
        fallback_routers=fallback_routers,
        use_multi_agent=use_multi_agent,
    )

    # Phase 4: 风险评估 (multi-agent 或 普通模式)
    logger.info("[辩论] Phase 4: 风险评估 (模式=%s)", _mode)
    risk_result = run_risk_assessment(
        router, arb_result, team_reports, symbol, name,
        fallback_routers=fallback_routers,
        use_multi_agent=use_multi_agent,
    )

    # 填充最终结果
    result.decision = risk_result.get("decision", arb_result.get("decision", "hold"))
    result.score_override = risk_result.get("score") or arb_result.get("score")
    result.target_price = risk_result.get("target_price") or arb_result.get("target_price")
    result.stop_loss = risk_result.get("stop_loss") or arb_result.get("stop_loss")
    result.confidence = float(risk_result.get("confidence", arb_result.get("confidence", 0.5)))
    result.risk_score = float(risk_result.get("risk_score", 0.5))
    result.reasoning = risk_result.get("reasoning", arb_result.get("reasoning", ""))
    result.plans = arb_result.get("plans", [])
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
