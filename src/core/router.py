# -*- coding: utf-8 -*-
"""统一 LLM 路由：多 Provider 抽象层，支持 10+ 大模型 API + Vision。

通用基础设施，不含领域业务逻辑。
"""
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
    """读取 LLM 调用专用代理。国内模型强制直连；海外模型走 LLM_PROXY。
    返回 {} 强制直连（绕过系统代理）；返回 {"http":..., "https":...} 走指定代理。"""
    if provider.lower() in _DOMESTIC_PROVIDERS:
        return {}  # 空 dict = 强制直连，不走系统代理
    proxy = os.getenv("LLM_PROXY", "").strip()
    if not proxy:
        return {}  # 未配置代理时也强制直连
    return {"http": proxy, "https": proxy}


# 支持图像输入的提供商（视觉模型原生支持）
_VISION_PROVIDERS = {"gemini", "claude", "openai", "chatgpt", "grok"}

# 默认视觉回退提供商
_DEFAULT_VISION_FALLBACK = "kimi"

# 各国内提供商的视觉模型与文字模型环境变量映射
_DOMESTIC_VISION_ENV_MAP: dict[str, tuple[str, str]] = {
    "kimi":     ("KIMI_VISION_MODEL",     "KIMI_MODEL"),
    "deepseek": ("DEEPSEEK_VISION_MODEL", "DEEPSEEK_MODEL"),
    "glm":      ("GLM_VISION_MODEL",      "GLM_MODEL"),
    "qwen":     ("QWEN_VISION_MODEL",     "QWEN_MODEL"),
    "minmax":   ("MINMAX_VISION_MODEL",   "MINMAX_MODEL"),
    "doubao":   ("DOUBAO_VISION_MODEL",   "DOUBAO_MODEL"),
}


def _supports_vision(provider: str) -> bool:
    """判断某提供商是否支持图像输入（Vision API）。"""
    p = provider.lower()
    if p in _VISION_PROVIDERS:
        return True
    vision_env, model_env = _DOMESTIC_VISION_ENV_MAP.get(p, (None, None))
    if vision_env:
        vision_model = os.getenv(vision_env, "").strip()
        if vision_model:
            return True
    if model_env:
        model = os.getenv(model_env, "").lower()
        if "vision" in model or "-vl" in model or "4v" in model:
            return True
    return False


def _parse_score_from_response(text: str, provider_hint: str = "") -> float | None:
    """从大模型回复中解析 0-100 的评分。通用解析器。"""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
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
    candidates = []
    for m in re.finditer(r"\b(100|[0-9]{1,2})(?:\.\d+)?\b", text):
        v = float(m.group(1))
        if 0 <= v <= 100:
            candidates.append(v)
    if candidates:
        return round(candidates[-1], 2)
    return None


class LLMRouter:
    """统一 LLM 路由：支持多家主流模型 API（OpenAI 兼容 / Gemini / Claude），含 Vision。

    这是通用的 LLM 传输层，负责：
    - 多 Provider API 适配（grok/deepseek/kimi/glm/gemini/claude/openai/doubao/qwen/minmax/perplexity）
    - 代理路由（海外走 LLM_PROXY，国内直连）
    - Session 管理与连接池
    - Vision 多模态支持（text + image）
    - 超时恢复（close_session 强制中断）

    领域相关的 prompt 构造（如权重分配、评分等）应放在业务层。
    """

    def __init__(
        self,
        provider: str = "grok",
        temperature: float = 0.3,
        max_tokens: int = 600,
        request_timeout_sec: float = 8.0,
        multi_turn: bool = True,
        system_message: str | None = None,
    ):
        self.provider = (provider or "grok").lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.multi_turn = multi_turn
        self.system_message = system_message  # None = 使用各方法的默认值
        self._session = self._create_session()
        self._session_lock = threading.Lock()

    # ── Session 管理 ──

    def _create_session(self) -> requests.Session:
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def close_session(self) -> None:
        """关闭当前 Session（中断正在进行的 HTTP 请求）并重建连接池。"""
        with self._session_lock:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = self._create_session()

    def _safe_post(self, url: str, **kwargs) -> requests.Response:
        """带连接错误恢复的 POST 请求。"""
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

    # ── 通用对话 ──

    def _get_system_message(self, default: str = "") -> str:
        """获取 system message：优先用构造时的自定义值，否则用 default。"""
        return self.system_message or default

    def enrich_reason(
        self,
        role: str,
        symbol: str,
        base_reason: str,
        data_context: str | None = None,
    ) -> str:
        """请大模型精炼专家分析结论为 2-3 句。"""
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

    def chat(self, prompt: str, multi_turn: bool = True) -> Optional[str]:
        """公开的对话接口。"""
        return self._chat(prompt, multi_turn=multi_turn)

    def chat_with_image(self, prompt: str, image_b64: str, mime_type: str = "image/png") -> Optional[str]:
        """发送图片+文字提示给视觉模型。"""
        if not self.supports_vision():
            return None
        try:
            if self.provider == "gemini":
                return self._chat_gemini_vision(prompt, image_b64, mime_type)
            if self.provider == "claude":
                return self._chat_claude_vision(prompt, image_b64, mime_type)
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
        messages = [{"role": "user", "content": prompt}]
        if multi_turn:
            sys_msg = self._get_system_message("你是一位专业的中国股市分析员，请基于提供的数据给出客观、可执行的研判。")
            messages = [{"role": "system", "content": sys_msg}] + messages
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
                    logger.warning("Perplexity API 400 Bad Request. response: %s", err_body)
                except Exception:
                    pass
            resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0].get("message", {}).get("content")
            if content is None:
                return None
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(parts) if parts else ""
            if not isinstance(content, str):
                content = str(content)
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
            body["system"] = self._get_system_message("你是一位专业的中国股市分析员，请基于提供的数据给出客观、可执行的研判。")
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

    # ── Vision 方法 ──

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
        sys_msg = self._get_system_message("你是一位专业的中国股市技术分析员，请基于K线图给出客观、可执行的研判。")
        body = {
            "model": model,
            "max_tokens": self.max_tokens,
            "system": sys_msg,
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
        """OpenAI-兼容 vision API（grok/openai/kimi-vision 等）。"""
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
        sys_msg = self._get_system_message("你是一位专业的中国股市技术分析员，请基于K线图给出客观、可执行的研判。")
        messages = [
            {"role": "system", "content": sys_msg},
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
