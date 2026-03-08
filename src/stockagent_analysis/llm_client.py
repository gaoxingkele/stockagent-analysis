# -*- coding: utf-8 -*-
import json
import logging
import os
import re
import threading
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


# 国内大模型，不走代理（直连）
_DOMESTIC_PROVIDERS = {"kimi", "deepseek", "glm", "doubao", "qwen", "minmax"}


def _get_llm_proxies(provider: str = "") -> dict | None:
    """读取 LLM 调用专用代理。国内模型（kimi/deepseek/glm/doubao/qwen/minmax）直连不走代理。
    返回 None 时 requests 使用系统默认代理；返回空 dict {} 时强制直连。"""
    if provider.lower() in _DOMESTIC_PROVIDERS:
        return None
    proxy = os.getenv("LLM_PROXY", "").strip()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


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
        # 若权重完全一致，按配置 weight 做后备差异化，避免全部相同
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
    # 完全一致：用配置中的 weight 做相对比例，再归一化
    config_weights = {a["agent_id"]: max(0.01, float(a.get("weight", 1.0))) for a in agents}
    total_c = sum(config_weights.get(aid, 1.0) for aid in weights)
    if total_c <= 0:
        return weights
    out = {aid: config_weights.get(aid, 1.0) / total_c for aid in weights}
    return out


def _parse_score_from_response(text: str, provider_hint: str = "") -> float | None:
    """
    从大模型回复中解析 0-100 的评分。优先识别「评分：72」、最后一个在范围内的数字等，避免把 "0-100" 里的 0 当成分数。
    解析失败返回 None，由调用方决定默认值。
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    # 1) 显式模式：评分：72 / score: 72 / 72分 / 72/100
    for pat in [
        r"[评打]分[：:\s]*([0-9]{1,3}\.?\d*)",
        r"score\s*[：:\s]*([0-9]{1,3}\.?\d*)",
        r"([0-9]{1,3}\.?\d*)\s*分",
        r"([0-9]{1,3}\.?\d*)\s*/\s*100",
    ]:
        m = re.search(pat, text, re.I)
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return round(v, 2)
    # 2) JSON 或纯数字：{"score": 72} 或 72 或 "72"
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            v = float(obj["score"])
            if 0 <= v <= 100:
                return round(v, 2)
        if isinstance(obj, (int, float)):
            v = float(obj)
            if 0 <= v <= 100:
                return round(v, 2)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if text.isdigit():
        v = float(text)
        if 0 <= v <= 100:
            return round(v, 2)
    # 3) 收集所有 0-100 的数字，取最后一个（避免 "0-100" 中的 0 或 100 被误取为第一个）
    candidates = []
    for m in re.finditer(r"\b(100|[0-9]{1,2})(?:\.\d+)?\b", text):
        v = float(m.group(1))
        if 0 <= v <= 100:
            candidates.append(v)
    if candidates:
        return round(candidates[-1], 2)
    return None


def score_agent_analysis(
    router: Any,
    role: str,
    agent_id: str,
    symbol: str,
    name: str,
    reason: str,
    data_context: str | None,
) -> float | None:
    """
    请大模型对某 Agent 的分析结论打分（0-100）。
    reason: Agent 的原始结论/研判；data_context: 数据摘要
    返回: 0~100 的分数，解析失败时返回 None（由调用方决定是否回退到备选Provider）
    """
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
        # 解析失败时记录原始回复前段，便于排查 Grok/Perplexity 等全部 50 的异常
        snippet = (text[:400] + "…") if len(text) > 400 else text
        logger.warning(
            "score_agent_analysis: parse failed (provider=%s agent=%s), raw snippet: %s",
            provider_hint,
            agent_id,
            snippet,
        )
        return None
    except Exception as e:
        logger.warning("score_agent_analysis: exception provider=%s agent=%s %s", getattr(router, "provider", ""), agent_id, e)
        return None


def generate_scenario_and_position(
    router: Any,
    symbol: str,
    name: str,
    final_score: float,
    decision_level_cn: str,
    key_levels_summary: str = "",
) -> tuple[str, str]:
    """生成情景分析与分批建仓/止损建议，各2-3句。"""
    prompt = (
        f"你是中国股市策略分析师。标的：{symbol} {name}。"
        f"综合评分={final_score:.1f}，决策等级：{decision_level_cn}。"
    )
    if key_levels_summary:
        prompt += f"\n关键价位：{key_levels_summary}"
    prompt += (
        "\n请用2-3句话给出：1）乐观/中性/悲观情景及简要触发条件；"
        "2）分批建仓或止损建议（价位或条件）。不要编造具体数字，可给出原则性建议。"
        "直接输出两段，第一段以「情景：」开头，第二段以「策略：」开头。"
    )
    try:
        text = router._chat(prompt, multi_turn=True)
        if not text:
            return "", ""
        text = text.strip()
        scenario, position = "", ""
        if "情景：" in text:
            parts = text.split("策略：", 1)
            scenario = parts[0].replace("情景：", "").strip()[:300]
            position = parts[1].strip()[:300] if len(parts) > 1 else ""
        elif "策略：" in text:
            parts = text.split("策略：", 1)
            scenario = parts[0].strip()[:300]
            position = parts[1].strip()[:300] if len(parts) > 1 else ""
        else:
            scenario = text[:300]
        return scenario, position
    except Exception:
        return "", ""


# 支持图像输入的提供商（视觉模型原生支持）
_VISION_PROVIDERS = {"gemini", "claude", "openai", "chatgpt", "grok"}

# 默认视觉回退提供商：当当前 provider 不支持视觉时使用（可通过 VISION_FALLBACK_PROVIDER 环境变量覆盖）
_DEFAULT_VISION_FALLBACK = "kimi"

# 各国内提供商的视觉模型与文字模型环境变量映射
_DOMESTIC_VISION_ENV_MAP: dict[str, tuple[str, str]] = {
    # provider: (VISION_MODEL_ENV, TEXT_MODEL_ENV)
    "kimi":     ("KIMI_VISION_MODEL",     "KIMI_MODEL"),
    "deepseek": ("DEEPSEEK_VISION_MODEL", "DEEPSEEK_MODEL"),
    "glm":      ("GLM_VISION_MODEL",      "GLM_MODEL"),
    "qwen":     ("QWEN_VISION_MODEL",     "QWEN_MODEL"),
    "minmax":   ("MINMAX_VISION_MODEL",   "MINMAX_MODEL"),
    "doubao":   ("DOUBAO_VISION_MODEL",   "DOUBAO_MODEL"),
}


def _supports_vision(provider: str) -> bool:
    """判断某提供商是否支持图像输入（Vision API）。
    检查顺序：
    1. 原生视觉提供商（gemini/grok/claude/openai）直接 True
    2. {PROVIDER}_VISION_MODEL 已配置（非空）→ 用户明确声明了视觉模型，认为支持
    3. {PROVIDER}_VISION_MODEL 名称含 vision / -vl / vl / 4v 关键字
    4. {PROVIDER}_MODEL 名称含上述关键字（兼容旧配置）
    """
    p = provider.lower()
    if p in _VISION_PROVIDERS:
        return True
    vision_env, model_env = _DOMESTIC_VISION_ENV_MAP.get(p, (None, None))
    if vision_env:
        vision_model = os.getenv(vision_env, "").strip()
        # 只要 _VISION_MODEL 已明确配置（非空），即认为该提供商有视觉能力
        if vision_model:
            return True
    if model_env:
        model = os.getenv(model_env, "").lower()
        if "vision" in model or "-vl" in model or "4v" in model:
            return True
    return False


class LLMRouter:
    """统一 LLM 路由：默认多轮对话模式，支持多家主流模型 API。"""

    def __init__(
        self,
        provider: str = "grok",
        temperature: float = 0.3,
        max_tokens: int = 600,
        request_timeout_sec: float = 8.0,
        multi_turn: bool = True,
    ):
        self.provider = (provider or "grok").lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.multi_turn = multi_turn  # 默认多轮对话模式
        self._session = self._create_session()
        self._session_lock = threading.Lock()

    # ── Session 管理 ──

    def _create_session(self) -> requests.Session:
        """创建新的 requests.Session，配置连接池参数。"""
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def close_session(self) -> None:
        """关闭当前 Session（中断正在进行的 HTTP 请求）并重建连接池。
        由超时取消线程调用，强制终止卡住的 requests.post()。"""
        with self._session_lock:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = self._create_session()

    def _safe_post(self, url: str, **kwargs) -> requests.Response:
        """带连接错误恢复的 POST 请求。首次连接失败时重建 Session 并重试一次。"""
        try:
            return self._session.post(url, **kwargs)
        except (requests.exceptions.ConnectionError, ConnectionResetError, OSError) as e:
            logger.warning("连接异常，重建Session并重试: %s", e)
            with self._session_lock:
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = self._create_session()
            return self._session.post(url, **kwargs)

    def supports_vision(self) -> bool:
        return _supports_vision(self.provider)

    def enrich_reason(
        self,
        role: str,
        symbol: str,
        base_reason: str,
        data_context: str | None = None,
    ) -> str:
        ctx_block = ""
        if data_context:
            ctx_block = (
                f"\n\n【本地已获取数据（供参考，不得编造）】\n{data_context}\n"
            )
        prompt = (
            f"你是中国股市{role}分析员。请基于以下数据与结论，精炼为2-3句，"
            f"保持客观、可执行，不夸张，不编造数据。\n"
            f"股票: {symbol}\n"
            f"原结论: {base_reason}"
            f"{ctx_block}"
        )
        try:
            text = self._chat(prompt, multi_turn=self.multi_turn)
            if text:
                return text.strip()
        except Exception:
            pass
        return base_reason

    def chat_with_image(self, prompt: str, image_b64: str, mime_type: str = "image/png") -> Optional[str]:
        """发送图片+文字提示给视觉模型。image_b64: base64编码的图片数据（无前缀）。"""
        if not self.supports_vision():
            return None
        try:
            if self.provider == "gemini":
                return self._chat_gemini_vision(prompt, image_b64, mime_type)
            if self.provider == "claude":
                return self._chat_claude_vision(prompt, image_b64, mime_type)
            # OpenAI兼容（grok/openai/chatgpt/kimi-vision/qwen-vl等）
            return self._chat_openai_vision(prompt, image_b64, mime_type)
        except Exception as e:
            logger.warning("chat_with_image failed provider=%s: %s", self.provider, e)
            return None

    def _chat(self, prompt: str, multi_turn: bool = True) -> Optional[str]:
        if self.provider in {"grok", "deepseek", "kimi", "chatgpt", "openai", "minmax", "glm", "doubao", "qwen", "perplexity"}:
            return self._chat_openai_compatible(prompt, multi_turn=multi_turn)
        if self.provider == "gemini":
            return self._chat_gemini(prompt, multi_turn=multi_turn)
        if self.provider == "claude":
            return self._chat_claude(prompt, multi_turn=multi_turn)
        return None

    def _chat_openai_compatible(self, prompt: str, multi_turn: bool = True) -> Optional[str]:
        if self.provider == "grok":
            api_key = os.getenv("GROK_API_KEY", "")
            model = os.getenv("GROK_MODEL", "grok-2")
            base_url = "https://api.x.ai/v1/chat/completions"
        elif self.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-v3")
            base_url = "https://api.deepseek.com/chat/completions"
        elif self.provider == "kimi":
            api_key = os.getenv("KIMI_API_KEY", "")
            model = os.getenv("KIMI_MODEL", "kimi-k2-latest")
            base_url = "https://api.moonshot.cn/v1/chat/completions"
        elif self.provider == "minmax":
            api_key = os.getenv("MINMAX_API_KEY", "")
            model = os.getenv("MINMAX_MODEL", "MiniMax-M2.5")
            base_url = os.getenv("MINMAX_BASE_URL", "https://api.minimaxi.com/v1/chat/completions")
        elif self.provider == "glm":
            api_key = os.getenv("GLM_API_KEY", "")
            model = os.getenv("GLM_MODEL", "glm-4.5")
            base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        elif self.provider == "doubao":
            api_key = os.getenv("DOUBAO_API_KEY", "")
            model = os.getenv("DOUBAO_MODEL", "doubao-1.5-pro-32k")
            base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        elif self.provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", "")
            model = os.getenv("QWEN_MODEL", "qwen-max-latest")
            base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        elif self.provider == "perplexity":
            api_key = os.getenv("PERPLEXITY_API_KEY", "")
            model = os.getenv("PERPLEXITY_MODEL", "sonar")
            base_url = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai/chat/completions")
        elif self.provider in ("openai", "chatgpt"):
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_MODEL", "gpt-4.1")
            base_url = "https://api.openai.com/v1/chat/completions"

        if not api_key:
            return None
        # 多轮对话模式：使用 messages 数组，支持后续追加历史
        messages = [{"role": "user", "content": prompt}]
        if multi_turn:
            messages = [{"role": "system", "content": "你是一位专业的中国股市分析员，请基于提供的数据给出客观、可执行的研判。"}] + messages
        # 部分推理模型（kimi-k2.5、deepseek-reasoner 等）只允许 temperature=1
        temp = self.temperature
        _FIXED_TEMP_MODELS = {"kimi-k2.5", "deepseek-reasoner"}
        if model.lower() in _FIXED_TEMP_MODELS:
            temp = 1.0
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = self._safe_post(base_url, headers=headers, json=payload, timeout=self.request_timeout_sec, proxies=_get_llm_proxies(self.provider))
        if not resp.ok:
            if resp.status_code == 400 and "perplexity" in base_url.lower():
                try:
                    err_body = resp.text[:500] if resp.text else ""
                    logger.warning("Perplexity API 400 Bad Request (check model name/temperature/max_tokens). response: %s", err_body)
                except Exception:
                    pass
            resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content")
            if content is None:
                return None
            # 部分 API（如 OpenAI 多模态）返回 content 为 list：[{"type":"text","text":"72"}]
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(parts) if parts else ""
            if not isinstance(content, str):
                content = str(content)
            # 部分模型（如 MiniMax-M2.5, DeepSeek-R1）返回 <think>...</think> 思考过程，提取正文
            if "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        return None

    def _chat_gemini(self, prompt: str, multi_turn: bool = True) -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY", "")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        if not api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        resp = self._safe_post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=self.request_timeout_sec,
            proxies=_get_llm_proxies("gemini"),
        )
        resp.raise_for_status()
        data = resp.json()
        cands = data.get("candidates", [])
        if not cands:
            return None
        return cands[0]["content"]["parts"][0].get("text", "")

    def _chat_claude(self, prompt: str, multi_turn: bool = True) -> Optional[str]:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-0")
        if not api_key:
            return None
        body: dict = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if multi_turn:
            body["system"] = "你是一位专业的中国股市分析员，请基于提供的数据给出客观、可执行的研判。"
        resp = self._safe_post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=self.request_timeout_sec,
            proxies=_get_llm_proxies("claude"),
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            return None
        return content[0].get("text", "")

    # ── Vision 方法 ────────────────────────────────────────────────

    def _chat_gemini_vision(self, prompt: str, image_b64: str, mime_type: str) -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY", "")
        model = os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        if not api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                    {"text": prompt},
                ]
            }]
        }
        resp = self._safe_post(url, json=body, timeout=self.request_timeout_sec,
                              proxies=_get_llm_proxies("gemini"))
        resp.raise_for_status()
        data = resp.json()
        cands = data.get("candidates", [])
        if not cands:
            return None
        return cands[0]["content"]["parts"][0].get("text", "")

    def _chat_claude_vision(self, prompt: str, image_b64: str, mime_type: str) -> Optional[str]:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        model = os.getenv("ANTHROPIC_VISION_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6"))
        if not api_key:
            return None
        body = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": "你是一位专业的中国股市技术分析员，请基于K线图给出客观、可执行的研判。",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        resp = self._safe_post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json=body,
            timeout=self.request_timeout_sec,
            proxies=_get_llm_proxies("claude"),
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            return None
        return content[0].get("text", "")

    def _chat_openai_vision(self, prompt: str, image_b64: str, mime_type: str) -> Optional[str]:
        """OpenAI-兼容 vision API（grok/openai/kimi-vision 等）。
        图像调用读 {PROVIDER}_VISION_MODEL；文字调用读 {PROVIDER}_MODEL（见 _chat_openai_compatible）。
        """
        if self.provider == "grok":
            api_key = os.getenv("GROK_API_KEY", "")
            model = os.getenv("GROK_VISION_MODEL", os.getenv("GROK_MODEL", "grok-2-vision-1212"))
            base_url = "https://api.x.ai/v1/chat/completions"
        elif self.provider in ("openai", "chatgpt"):
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        elif self.provider == "kimi":
            api_key = os.getenv("KIMI_API_KEY", "")
            model = os.getenv("KIMI_VISION_MODEL", os.getenv("KIMI_MODEL", "moonshot-v1-8k-vision-preview"))
            base_url = "https://api.moonshot.cn/v1/chat/completions"
        elif self.provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", "")
            model = os.getenv("QWEN_VISION_MODEL", os.getenv("QWEN_MODEL", "qwen-vl-max"))
            base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
            base_url = "https://api.openai.com/v1/chat/completions"
        if not api_key:
            return None
        data_url = f"data:{mime_type};base64,{image_b64}"
        messages = [
            {"role": "system", "content": "你是一位专业的中国股市技术分析员，请基于K线图给出客观、可执行的研判。"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ]},
        ]
        payload = {"model": model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = self._safe_post(base_url, headers=headers, json=payload,
                              timeout=self.request_timeout_sec,
                              proxies=_get_llm_proxies(self.provider))
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content", "")
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(parts)
            return content if isinstance(content, str) else str(content)
        return None
