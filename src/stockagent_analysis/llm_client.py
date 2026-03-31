# -*- coding: utf-8 -*-
"""LLM 客户端：核心路由从 core.router 导入，本文件保留领域相关的 prompt 函数。

向后兼容：所有旧的 ``from .llm_client import LLMRouter`` 仍然有效。
"""
import json
import logging
import re
from typing import Any

# ── 从 core 导入通用基础设施（向后兼容 re-export） ──
from core.router import (  # noqa: F401 — re-export
    LLMRouter,
    _DOMESTIC_PROVIDERS,
    _get_llm_proxies,
    _VISION_PROVIDERS,
    _DEFAULT_VISION_FALLBACK,
    _DOMESTIC_VISION_ENV_MAP,
    _supports_vision,
    _parse_score_from_response,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────
# 领域相关：权重分配 / 评分 / 情景分析
# ────────────────────────────────────────────────

def assign_agent_weights(
    router: Any,
    agents: list[dict[str, Any]],
    symbol: str,
    name: str,
    provider_name: str = "",
    task_summary: str | None = None,
) -> dict[str, float]:
    """
    总协调在调用本模型时，先传入多智能体任务分工综述；大模型据此为各 Agent 分配权重，且权重须有差异。
    agents: [{"agent_id", "role", "dim_code", "weight"(可选，用于后备差异化)}, ...]
    task_summary: 任务分工综述文本，由总协调在调用各模型前统一生成
    返回: {agent_id: weight_0_1}，保证不全相同
    """
    lines = []
    for a in agents:
        lines.append(f"- {a['agent_id']} ({a['role']}, {a.get('dim_code','')})")
    agents_block = "\n".join(lines)
    identity = f"你当前是【{provider_name}】模型。" if provider_name else ""
    summary_block = ""
    if task_summary and task_summary.strip():
        summary_block = task_summary.strip() + "\n\n"

    prompt = (
        f"{identity}作为中国股市综合研判专家，总协调将以下多智能体任务分工综述发给你，请据此为各智能体分配权重（用于合成最终评分）。\n\n"
        f"{summary_block}"
        f"标的：{symbol} {name}\n\n"
        f"智能体列表：\n{agents_block}\n\n"
        "要求：\n"
        "1. 每个智能体分配 0~1 之间的权重（如 0.12 表示 12%）；\n"
        "2. 所有权重之和必须等于 1.0（100%）；\n"
        "3. 根据任务分工与各维度对研判的重要性，给出差异化权重，务必体现你的独特性。\n"
        "4. 权重必须有所差异，不得给所有智能体分配相同权重；至少应区分高、中、低重要性（例如核心维度权重更高）。\n\n"
        "请仅输出一个 JSON 对象，格式如：{\"agent_id1\":0.15,\"agent_id2\":0.10,...}，不要输出其他文字。"
    )
    try:
        provider_hint = getattr(router, "provider", "")
        agent_ids = [a["agent_id"] for a in agents]
        logger.info(
            "[LLM提交] 权重分配 provider=%s | agents(%d)=%s",
            provider_hint, len(agents), agent_ids,
        )
        text = router._chat(prompt, multi_turn=True)
        if not text:
            return _fallback_weights(agents)
        text = text.strip()
        obj = {}
        to_try = [text]
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            to_try.insert(0, m.group(1))
        for raw in to_try:
            if not raw or "{" not in raw:
                continue
            start = raw.find("{")
            if start < 0:
                continue
            depth, end = 0, -1
            for i, c in enumerate(raw[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end < 0:
                continue
            try:
                obj = json.loads(raw[start : end + 1])
                if isinstance(obj, dict):
                    break
            except json.JSONDecodeError:
                pass
        weights = {}
        for a in agents:
            aid = a["agent_id"]
            v = obj.get(aid, obj.get(aid.replace("_agent", ""), 1.0 / len(agents)))
            weights[aid] = max(0.0, min(1.0, float(v)))
        total = sum(weights.values()) or 1.0
        for aid in weights:
            weights[aid] /= total
        weights = _ensure_weight_differentiation(weights, agents)
        return weights
    except Exception:
        return _fallback_weights(agents)


def _fallback_weights(agents: list[dict[str, Any]]) -> dict[str, float]:
    """均等权重作为后备；再经差异化处理避免完全一致。"""
    n = len(agents) or 1
    w = 1.0 / n
    weights = {a["agent_id"]: w for a in agents}
    return _ensure_weight_differentiation(weights, agents)


def _ensure_weight_differentiation(weights: dict[str, float], agents: list[dict[str, Any]]) -> dict[str, float]:
    """若所有权重相同，则按 config weight 或 mandatory 做差异化，避免完全一致。"""
    if not weights or len(weights) <= 1:
        return weights
    vals = list(weights.values())
    if len(set(round(v, 6) for v in vals)) > 1:
        return weights
    config_weights = {a["agent_id"]: max(0.01, float(a.get("weight", 1.0))) for a in agents}
    total_c = sum(config_weights.get(aid, 1.0) for aid in weights)
    if total_c <= 0:
        return weights
    out = {aid: config_weights.get(aid, 1.0) / total_c for aid in weights}
    return out


def score_agent_analysis(
    router: Any,
    role: str,
    agent_id: str,
    symbol: str,
    name: str,
    reason: str,
    data_context: str | None,
) -> float | None:
    """请大模型对某 Agent 的分析结论打分（0-100）。"""
    ctx_block = f"\n\n【数据摘要】\n{data_context}\n" if data_context else ""
    prompt = (
        f"你是中国股市综合研判评分员。请对以下分析师对 {symbol} {name} 的研判进行打分（0-100）。\n"
        f"分析师角色：{role}（{agent_id}）\n"
        f"研判结论：{reason}"
        f"{ctx_block}\n"
        "请仅输出一个数字（0-100），表示你的评分。不要输出其他文字。若给分，直接写数字即可，例如：72"
    )
    try:
        provider_hint = getattr(router, "provider", "")
        ctx_len = len(data_context or "")
        reason_brief = (reason or "")[:100].replace("\n", " ")
        logger.info(
            "[LLM提交] 评分请求 provider=%s agent=%s | 研判=%s... | 数据摘要=%d字符",
            provider_hint, agent_id, reason_brief, ctx_len,
        )
        text = router._chat(prompt, multi_turn=True)
        if not text:
            logger.warning("score_agent_analysis: empty response for %s (provider=%s)", agent_id, provider_hint)
            return None
        text = text.strip()
        provider_hint = getattr(router, "provider", "")
        score = _parse_score_from_response(text, provider_hint)
        if score is not None:
            return score
        snippet = (text[:400] + "…") if len(text) > 400 else text
        logger.warning(
            "score_agent_analysis: parse failed (provider=%s agent=%s), raw snippet: %s",
            provider_hint, agent_id, snippet,
        )
        return None
    except Exception as e:
        logger.warning("score_agent_analysis: exception provider=%s agent=%s %s", getattr(router, "provider", ""), agent_id, e)
        return None


def enrich_and_score(
    router: Any,
    role: str,
    agent_id: str,
    symbol: str,
    name: str,
    base_reason: str,
    data_context: str | None,
) -> tuple[str | None, float | None]:
    """合并研判增强+评分为一次LLM调用，返回 (enriched_text, score)。"""
    ctx_block = ""
    if data_context:
        ctx_block = f"\n【本地已获取数据】\n{data_context}\n"
    prompt = (
        f"你是中国股市{role}分析员。基于以下数据与本地分析结论，给出你的独立研判。\n\n"
        f"股票: {symbol} {name}\n"
        f"分析师: {role}（{agent_id}）\n"
        f"本地结论: {base_reason}"
        f"{ctx_block}\n"
        f"【评分标准校准】\n"
        f"- 80-100: 强烈看多，未来20日上涨概率>75%\n"
        f"- 60-79: 偏多，上涨概率55-75%\n"
        f"- 40-59: 中性/不确定\n"
        f"- 20-39: 偏空，下跌概率55-75%\n"
        f"- 0-19: 强烈看空，下跌概率>75%\n\n"
        f"【输出示例】\n"
        f'{{"analysis":"MACD金叉配合放量突破20日均线，趋势转强，但RSI接近超买区需警惕短期回调。",'
        f'"score":68,'
        f'"risk":"RSI超买+上方前高压力"}}\n\n'
        f"请仅输出一个JSON对象，包含 analysis(2-3句精炼结论)、score(0-100整数)、risk(主要风险，1句话)："
    )
    try:
        provider_hint = getattr(router, "provider", "")
        reason_brief = (base_reason or "")[:100].replace("\n", " ")
        logger.info(
            "[LLM提交] 合并研判+评分 provider=%s agent=%s | 结论=%s...",
            provider_hint, agent_id, reason_brief,
        )
        text = router._chat(prompt, multi_turn=True)
        if not text:
            return None, None
        text = text.strip()
        # 优先JSON解析
        try:
            # 尝试从可能包裹的markdown代码块中提取JSON
            json_text = text
            m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
            if m:
                json_text = m.group(1)
            elif "{" in text:
                start = text.find("{")
                depth, end = 0, -1
                for i, c in enumerate(text[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end >= 0:
                    json_text = text[start:end + 1]
            obj = json.loads(json_text)
            if isinstance(obj, dict):
                analysis = obj.get("analysis", "")
                risk = obj.get("risk", "")
                score_val = obj.get("score")
                if score_val is not None:
                    score_val = float(score_val)
                    if 0 <= score_val <= 100:
                        enriched = analysis or text
                        if risk:
                            enriched += f"\n[风险提示] {risk}"
                        return enriched, round(score_val, 2)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # JSON失败：用正则提score，全文当enrichment
        score = _parse_score_from_response(text, getattr(router, "provider", ""))
        return text, score
    except Exception as e:
        logger.warning("enrich_and_score exception provider=%s agent=%s: %s",
                       getattr(router, "provider", ""), agent_id, e)
        return None, None


def config_weights(agents: list[dict]) -> dict[str, float]:
    """从配置文件权重直接计算归一化权重，无需LLM调用。"""
    weights = {a["agent_id"]: max(0.01, float(a.get("weight", 1 / max(len(agents), 1))))
               for a in agents}
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _fix_sniper_points(sp: dict, current_price: float | None) -> dict:
    """后处理校验狙击点位，确保盈亏比>=2且止损距离合理。"""
    if not sp or not current_price or current_price <= 0:
        return sp

    def _f(v):
        try:
            return float(v) if v is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    buy = _f(sp.get("ideal_buy"))
    sl = _f(sp.get("stop_loss"))
    tp1 = _f(sp.get("take_profit_1"))
    tp2 = _f(sp.get("take_profit_2"))
    sec = _f(sp.get("secondary_buy"))

    if buy <= 0:
        buy = round(current_price * 0.97, 2)
    if sl <= 0 or sl >= buy:
        sl = round(buy * 0.95, 2)

    risk = buy - sl  # 止损距离

    # 约束1: 止损距买点不超过10%
    max_risk = buy * 0.10
    if risk > max_risk:
        sl = round(buy - max_risk, 2)
        risk = max_risk

    # 约束2: 盈亏比 >= 2
    if risk > 0:
        min_tp1 = buy + risk * 2
        if tp1 < min_tp1:
            tp1 = round(min_tp1, 2)
        # tp2 至少比 tp1 高一个 risk
        if tp2 <= tp1:
            tp2 = round(tp1 + risk, 2)

    # 约束3: secondary_buy 在 buy 和 sl 之间
    if sec <= 0 or sec >= buy:
        sec = round((buy + sl) / 2, 2)
    if sec <= sl:
        sec = round(sl + (buy - sl) * 0.3, 2)

    sp["ideal_buy"] = buy
    sp["secondary_buy"] = sec
    sp["stop_loss"] = sl
    sp["take_profit_1"] = tp1
    sp["take_profit_2"] = tp2
    return sp


def generate_scenario_and_position(
    router: Any,
    symbol: str,
    name: str,
    final_score: float,
    decision_level_cn: str,
    key_levels_summary: str = "",
    current_price: float | None = None,
    memory_context: str = "",
) -> dict:
    """生成结构化情景分析、狙击点位、建仓策略、持仓建议、决策摘要。

    返回: dict {
        scenarios, sniper_points, position_strategy, position_advice,
        rating, executive_summary, investment_thesis
    }
    """
    price_info = f"当前价: {current_price:.2f}\n" if current_price else ""
    kl_info = f"关键价位：{key_levels_summary}\n" if key_levels_summary else ""
    mem_info = f"{memory_context}\n" if memory_context else ""
    prompt = (
        f"你是中国股市策略分析师。标的：{symbol} {name}。\n"
        f"综合评分={final_score:.1f}，决策等级：{decision_level_cn}。\n"
        f"{price_info}{kl_info}{mem_info}"
        f"请仅输出一个JSON对象，包含以下字段：\n"
        f'{{"scenarios": {{"optimistic": {{"probability": 35, "target": 价格数字, "reason": "一句话触发条件"}},'
        f' "neutral": {{"probability": 45, "target": 价格数字, "reason": "一句话"}},'
        f' "pessimistic": {{"probability": 20, "target": 价格数字, "reason": "一句话"}}}},'
        f' "sniper_points": {{"ideal_buy": 首选买入价, "secondary_buy": 次选买入价,'
        f' "stop_loss": 止损价, "take_profit_1": 第一目标价, "take_profit_2": 第二目标价}},'
        f' "position_strategy": "分批建仓策略文字描述（2-3句）",'
        f' "position_advice": {{"no_position": "空仓者操作建议（1-2句）",'
        f' "has_position": "持仓者操作建议（1-2句）",'
        f' "position_ratio": "建议仓位比例如50%"}}}}\n'
        f"注意：价格必须为数字，不要加单位。\n"
        f"【盈亏比硬约束】止盈距买点的距离 必须 >= 止损距买点距离的2倍，即 (take_profit_1 - ideal_buy) >= 2 * (ideal_buy - stop_loss)。"
        f"止损距买点不应超过买点的10%。如果找不到合理的止损位，宁可缩小止损距离也要保证盈亏比>=2。"
    )
    try:
        text = router._chat(prompt, multi_turn=True)
        if not text:
            return _empty_scenario_result()
        text = text.strip()
        # 尝试JSON解析（markdown代码块 或 裸JSON）
        json_text = text
        # 1) 尝试markdown代码块（贪婪匹配，防止嵌套JSON截断）
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
        if m:
            candidate = m.group(1)
            try:
                json.loads(candidate)
                json_text = candidate
            except json.JSONDecodeError:
                pass  # 解析失败，交给下面 brace-matching
        # 2) 裸JSON：找最外层 { ... }（brace-depth匹配）
        if json_text is text and "{" in text:
            start = text.find("{")
            depth, end = 0, -1
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end >= 0:
                json_text = text[start:end + 1]
        try:
            obj = json.loads(json_text)
            if isinstance(obj, dict):
                scenarios = obj.get("scenarios", {})
                sniper_points = obj.get("sniper_points", {})
                sniper_points = _fix_sniper_points(sniper_points, current_price)
                position_strategy = str(obj.get("position_strategy", ""))
                position_advice = obj.get("position_advice", {})
                return {
                "scenarios": scenarios,
                "sniper_points": sniper_points,
                "position_strategy": position_strategy,
                "position_advice": position_advice,
                "rating": str(obj.get("rating", decision_level_cn)),
                "executive_summary": str(obj.get("executive_summary", "")),
                "investment_thesis": str(obj.get("investment_thesis", "")),
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # 回退: 自由文本解析
        return _parse_freetext_scenario_v2(decision_level_cn)
    except Exception:
        return {}, {}, "", {}


def _empty_scenario_result() -> dict:
    """返回空的结构化结果（失败时使用）。"""
    return {
        "scenarios": {}, "sniper_points": {}, "position_strategy": "",
        "position_advice": {}, "rating": "", "executive_summary": "", "investment_thesis": "",
    }

def _parse_freetext_scenario_v2(decision_level_cn: str = "") -> dict:
    """从自由文本中解析情景和策略（回退逻辑），返回新格式 dict。"""
    return {
        "scenarios": {}, "sniper_points": {}, "position_strategy": "",
        "position_advice": {}, "rating": decision_level_cn,
        "executive_summary": "", "investment_thesis": "",
    }


