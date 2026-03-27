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
from urllib3.util.retry import Retry

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


# ── Cloubic 多模型统一调度平台 ──

def _load_cloubic_env():
    """加载 .env.cloubic 配置（若存在）。"""
    from pathlib import Path
    cloubic_env = Path(__file__).resolve().parents[2] / ".env.cloubic"
    if not cloubic_env.is_file():
        return
    for line in cloubic_env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if key and key not in os.environ:  # 不覆盖已有环境变量
                os.environ[key] = val

_load_cloubic_env()

_CLOUBIC_ENABLED: bool | None = None

def _is_cloubic_mode() -> bool:
    """Cloubic 是否启用：CLOUBIC_ENABLED=true 且 CLOUBIC_API_KEY 非空。"""
    global _CLOUBIC_ENABLED
    if _CLOUBIC_ENABLED is None:
        _CLOUBIC_ENABLED = (
            os.getenv("CLOUBIC_ENABLED", "").strip().lower() == "true"
            and bool(os.getenv("CLOUBIC_API_KEY", "").strip())
        )
    return _CLOUBIC_ENABLED


def _should_route_via_cloubic(provider: str) -> bool:
    """判断某 provider 是否应走 Cloubic 路由。"""
    if not _is_cloubic_mode():
        return False
    p = provider.lower()
    if p == "kimi":
        return False  # kimi 始终直连
    whitelist = os.getenv("CLOUBIC_ROUTED_PROVIDERS", "").strip()
    if not whitelist:
        return True  # 空白名单 = 全部走 Cloubic
    return p in {x.strip().lower() for x in whitelist.split(",")}


def _get_cloubic_model_chain(provider: str) -> list[str]:
    """获取 provider 在 Cloubic 上的模型降级链（逗号分隔）。
    例: 'cc/claude-opus-4-6,cc/claude-sonnet-4-6,cc/claude-haiku-4-5-20251001'
    → ['cc/claude-opus-4-6', 'cc/claude-sonnet-4-6', 'cc/claude-haiku-4-5-20251001']
    """
    p = provider.lower()
    env_key = f"CLOUBIC_{p.upper()}_MODEL"
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return []
    return [m.strip() for m in raw.split(",") if m.strip()]


def _get_cloubic_model(provider: str) -> str:
    """获取 provider 在 Cloubic 上的首选模型名（向后兼容）。"""
    chain = _get_cloubic_model_chain(provider)
    return chain[0] if chain else ""


def _get_direct_model_chain(provider: str, primary_model: str) -> list[str]:
    """获取直连 provider 的模型降级链：主模型 → FALLBACK → FALLBACK2。
    与 Gemini 已有的 FALLBACK_MODEL/FALLBACK_MODEL2 模式一致。"""
    p = provider.upper()
    chain = [primary_model]
    fb1 = os.getenv(f"{p}_FALLBACK_MODEL", "").strip()
    fb2 = os.getenv(f"{p}_FALLBACK_MODEL2", "").strip()
    if fb1:
        chain.append(fb1)
    if fb2:
        chain.append(fb2)
    # 去重保序
    seen: set[str] = set()
    return [m for m in chain if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]


# 支持图像输入的提供商（视觉模型原生支持）
_VISION_PROVIDERS = {"gemini", "claude", "openai", "chatgpt", "grok"}

# 默认视觉回退提供商（海外兜底）
_DEFAULT_VISION_FALLBACK = "gemini"

# 国产视觉回退顺序：当Provider无视觉能力时，按此顺序尝试国产视觉API
# minmax 的 OpenAI 兼容接口不支持图片输入，不纳入回退链
_DOMESTIC_VISION_FALLBACK_ORDER = ["qwen", "glm", "kimi", "doubao"]

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
    """从大模型回复中解析 0-100 的评分。通用解析器。优先JSON，回退正则。"""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    # 1. 优先尝试提取 JSON 中的 score
    try:
        json_match = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', text, re.S)
        if json_match:
            obj = json.loads(json_match.group())
            s = float(obj["score"])
            if 0 <= s <= 100:
                return round(s, 2)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        pass
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
    # 2. 正则模式匹配
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
    # 3. 纯数字
    if text.isdigit():
        v = float(text)
        if 0 <= v <= 100:
            return round(v, 2)
    # 4. 最后一个合理数字
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
        request_timeout_sec: float = 45.0,
        multi_turn: bool = True,
        system_message: str | None = None,
        model_override: str | None = None,
        no_fallback: bool = False,
    ):
        self.provider = (provider or "grok").lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout_sec = request_timeout_sec
        self.multi_turn = multi_turn
        self.system_message = system_message  # None = 使用各方法的默认值
        self.model_override = model_override  # 别名模式：锁定指定模型
        self.no_fallback = no_fallback        # 别名模式：禁用降级链
        self._session = self._create_session()
        self._session_lock = threading.Lock()

    # ── Session 管理 ──

    def _create_session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 502, 503],
            allowed_methods=["POST", "GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def _timeout(self, is_vision: bool = False) -> tuple[float, float]:
        """返回 (connect_timeout, read_timeout) 元组。"""
        read = self.request_timeout_sec
        if is_vision:
            read = max(read * 1.5, 60.0)
        return (8.0, read)

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
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                ConnectionResetError, OSError) as e:
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
                if _should_route_via_cloubic("gemini"):
                    return self._chat_openai_vision(prompt, image_b64, mime_type)
                return self._chat_gemini_vision(prompt, image_b64, mime_type)
            if self.provider == "claude":
                if _should_route_via_cloubic("claude"):
                    return self._chat_openai_vision(prompt, image_b64, mime_type)
                return self._chat_claude_vision(prompt, image_b64, mime_type)
            return self._chat_openai_vision(prompt, image_b64, mime_type)
        except Exception as e:
            logger.warning("chat_with_image failed provider=%s: %s", self.provider, e)
            return None

    def _chat(self, prompt: str, multi_turn: bool = True) -> Optional[str]:
        if self.provider in {"grok", "deepseek", "kimi", "chatgpt", "openai", "minmax", "glm", "doubao", "qwen", "perplexity"}:
            return self._chat_openai_compatible(prompt, multi_turn=multi_turn)
        if self.provider == "gemini":
            # Cloubic 路由时走 OpenAI 兼容接口
            if _should_route_via_cloubic("gemini"):
                return self._chat_openai_compatible(prompt, multi_turn=multi_turn)
            return self._chat_gemini(prompt, multi_turn=multi_turn)
        if self.provider == "claude":
            # Cloubic 路由时走 OpenAI 兼容接口（国内直连，无需 Anthropic API Key）
            if _should_route_via_cloubic("claude"):
                return self._chat_openai_compatible(prompt, multi_turn=multi_turn)
            return self._chat_claude(prompt, multi_turn=multi_turn)
        return None

    def _chat_openai_compatible(self, prompt: str, multi_turn: bool = True) -> Optional[str]:
        # ── Cloubic 路由：统一 API Key + Base URL，无代理直连 ──
        via_cloubic = _should_route_via_cloubic(self.provider)

        if self.provider == "grok":
            api_key = os.getenv("GROK_API_KEY", "")
            model = os.getenv("GROK_MODEL", "grok-4.20-0309-reasoning")
            base_url = "https://api.x.ai/v1/chat/completions"
        elif self.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
            base_url = "https://api.deepseek.com/chat/completions"
        elif self.provider == "kimi":
            api_key = os.getenv("KIMI_API_KEY", "")
            model = os.getenv("KIMI_MODEL", "kimi-k2.5")
            base_url = "https://api.moonshot.cn/v1/chat/completions"
        elif self.provider == "minmax":
            api_key = os.getenv("MINMAX_API_KEY", "")
            model = os.getenv("MINMAX_MODEL", "MiniMax-M2.7")
            base_url = os.getenv("MINMAX_BASE_URL", "https://api.minimaxi.com/v1/chat/completions")
        elif self.provider == "glm":
            api_key = os.getenv("GLM_API_KEY", "")
            model = os.getenv("GLM_MODEL", "glm-5")
            base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        elif self.provider == "doubao":
            api_key = os.getenv("DOUBAO_API_KEY", "")
            model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215")
            base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        elif self.provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", "")
            model = os.getenv("QWEN_MODEL", "qwen3.5-plus")
            base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        elif self.provider == "perplexity":
            api_key = os.getenv("PERPLEXITY_API_KEY", "")
            model = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
            base_url = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai/chat/completions")
        elif self.provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
            base_url = "https://api.anthropic.com/v1/messages"  # 仅 Cloubic 模式下实际走此分支
        elif self.provider in ("openai", "chatgpt"):
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_MODEL", "gpt-4o")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_MODEL", "gpt-4.1")
            base_url = "https://api.openai.com/v1/chat/completions"

        # Cloubic 覆盖：替换 api_key / base_url / model，支持降级链
        if via_cloubic:
            api_key = os.getenv("CLOUBIC_API_KEY", "")
            base_url = os.getenv("CLOUBIC_BASE_URL", "https://api.cloubic.com/v1/chat/completions")
            model_chain = _get_cloubic_model_chain(self.provider)
            if model_chain:
                model = model_chain[0]  # 首选模型，降级在下方循环处理

        # model_override 覆盖（别名模式）
        if self.model_override:
            model = self.model_override

        if not api_key:
            return None
        messages = [{"role": "user", "content": prompt}]
        if multi_turn:
            sys_msg = self._get_system_message("你是一位专业的中国股市分析员，请基于提供的数据给出客观、可执行的研判。")
            messages = [{"role": "system", "content": sys_msg}] + messages

        # 模型降级链：别名模式锁定单模型；否则 Cloubic 用 Cloubic 链，直连用 FALLBACK
        if self.no_fallback:
            models_to_try = [model]
        elif via_cloubic:
            models_to_try = _get_cloubic_model_chain(self.provider) or [model]
        else:
            models_to_try = _get_direct_model_chain(self.provider, model)
        last_error: Exception | None = None

        has_fallback = len(models_to_try) > 1
        route_tag = "Cloubic" if via_cloubic else self.provider

        for model_idx, try_model in enumerate(models_to_try):
            can_retry = model_idx < len(models_to_try) - 1
            try:
                temp = self.temperature
                _FIXED_TEMP_MODELS = {"kimi-k2.5", "deepseek-reasoner"}
                if try_model.lower() in _FIXED_TEMP_MODELS:
                    temp = 1.0
                payload = {
                    "model": try_model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": self.max_tokens,
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                proxies = {} if via_cloubic else _get_llm_proxies(self.provider)
                resp = self._safe_post(base_url, headers=headers, json=payload, timeout=self._timeout(), proxies=proxies)
                if not resp.ok:
                    if resp.status_code == 400 and "perplexity" in base_url.lower():
                        try:
                            err_body = resp.text[:500] if resp.text else ""
                            logger.warning("Perplexity API 400 Bad Request. response: %s", err_body)
                        except Exception:
                            pass
                    if can_retry:
                        logger.warning("[%s] %s model=%s HTTP %d，降级到 %s",
                                       route_tag, self.provider, try_model, resp.status_code, models_to_try[model_idx + 1])
                        continue
                    resp.raise_for_status()
                data = resp.json()
                if "choices" in data and data["choices"]:
                    msg = data["choices"][0].get("message", {})
                    content = msg.get("content")
                    if (content is None or content == "") and msg.get("reasoning_content"):
                        content = msg["reasoning_content"]
                    if content is None:
                        if can_retry:
                            logger.warning("[%s] %s model=%s 返回空内容，降级到 %s",
                                           route_tag, self.provider, try_model, models_to_try[model_idx + 1])
                            continue
                        return None
                    if isinstance(content, list):
                        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                        content = " ".join(parts) if parts else ""
                    if not isinstance(content, str):
                        content = str(content)
                    if "<think>" in content:
                        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    return content
                if can_retry:
                    logger.warning("[%s] %s model=%s 无choices，降级到 %s",
                                   route_tag, self.provider, try_model, models_to_try[model_idx + 1])
                    continue
                return None
            except Exception as e:
                last_error = e
                if can_retry:
                    logger.warning("[%s] %s model=%s 异常: %s，降级到 %s",
                                   route_tag, self.provider, try_model, e, models_to_try[model_idx + 1])
                    continue
                raise
        # 所有模型都走完且无返回
        if last_error:
            raise last_error
        return None

    # ── Grok Multi-Agent (/v1/responses) ──

    def chat_multi_agent(self, prompt: str, system_message: str | None = None) -> Optional[str]:
        """调用 Grok Multi-Agent API (/v1/responses)。
        内部启动 4-16 个推理线程并行分析同一问题，合成综合回答。
        effort: low/medium=4agents, high/xhigh=16agents。
        """
        api_key = os.getenv("GROK_API_KEY", "")
        model = os.getenv("GROK_MULTI_AGENT_MODEL", "grok-4.20-multi-agent-0309")
        effort = os.getenv("GROK_MULTI_AGENT_EFFORT", "high")
        if not api_key:
            return None
        url = "https://api.x.ai/v1/responses"
        input_messages = []
        if system_message:
            input_messages.append({"role": "system", "content": system_message})
        input_messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "input": input_messages,
            "reasoning": {"effort": effort},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        proxies = _get_llm_proxies("grok")
        resp = self._safe_post(url, headers=headers, json=payload,
                               timeout=self._timeout(), proxies=proxies)
        resp.raise_for_status()
        data = resp.json()
        # /v1/responses 返回格式: output[] 数组，取 type=message 的 content
        output = data.get("output", [])
        for item in output:
            if item.get("type") == "message":
                content_parts = item.get("content", [])
                texts = [p.get("text", "") for p in content_parts
                         if isinstance(p, dict) and p.get("type") == "output_text"]
                if texts:
                    return "\n".join(texts).strip()
        # fallback: 直接取 output_text 字段
        if data.get("output_text"):
            return data["output_text"].strip()
        return None

    def _gemini_model_chain(self) -> list[str]:
        """返回 Gemini 模型降级链：主模型 → fallback → fallback2。"""
        chain = [os.getenv("GEMINI_MODEL", "gemini-2.5-pro")]
        fb1 = os.getenv("GEMINI_FALLBACK_MODEL", "")
        fb2 = os.getenv("GEMINI_FALLBACK_MODEL2", "")
        if fb1:
            chain.append(fb1)
        if fb2:
            chain.append(fb2)
        # 去重保序
        seen: set[str] = set()
        return [m for m in chain if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]

    def _chat_gemini(self, prompt: str, multi_turn: bool = True) -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return None
        models = self._gemini_model_chain()
        for i, m in enumerate(models):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            resp = self._safe_post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=self._timeout(),
                proxies=_get_llm_proxies("gemini"),
            )
            if resp.status_code == 429 and i < len(models) - 1:
                print(f"[Gemini] {m} 限流(429)，降级到 {models[i+1]}", flush=True)
                continue
            resp.raise_for_status()
            data = resp.json()
            cands = data.get("candidates", [])
            if not cands:
                return None
            return cands[0]["content"]["parts"][0].get("text", "")
        return None

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
            timeout=self._timeout(),
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
        if not api_key:
            return None
        # 视觉模型链：VISION_MODEL → 通用降级链
        vision_model = os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        models = [vision_model] + [m for m in self._gemini_model_chain() if m != vision_model]
        body = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                    {"text": prompt},
                ]
            }]
        }
        for i, m in enumerate(models):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            resp = self._safe_post(url, json=body, timeout=self._timeout(is_vision=True),
                                  proxies=_get_llm_proxies("gemini"))
            if resp.status_code == 429 and i < len(models) - 1:
                print(f"[Gemini Vision] {m} 限流(429)，降级到 {models[i+1]}", flush=True)
                continue
            resp.raise_for_status()
            data = resp.json()
            cands = data.get("candidates", [])
            if not cands:
                return None
            return cands[0]["content"]["parts"][0].get("text", "")
        return None

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
            timeout=self._timeout(is_vision=True),
            proxies=_get_llm_proxies("claude"),
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("content", [])
        if not content:
            return None
        return content[0].get("text", "")

    def _chat_openai_vision(self, prompt: str, image_b64: str, mime_type: str) -> Optional[str]:
        """OpenAI-兼容 vision API（grok/openai/kimi-vision/claude-via-cloubic 等）。"""
        via_cloubic = _should_route_via_cloubic(self.provider)

        if self.provider == "grok":
            api_key = os.getenv("GROK_API_KEY", "")
            model = os.getenv("GROK_VISION_MODEL", os.getenv("GROK_MODEL", "grok-4.20-0309-reasoning"))
            base_url = "https://api.x.ai/v1/chat/completions"
        elif self.provider in ("openai", "chatgpt"):
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        elif self.provider == "kimi":
            api_key = os.getenv("KIMI_API_KEY", "")
            model = os.getenv("KIMI_VISION_MODEL", os.getenv("KIMI_MODEL", "kimi-k2.5"))
            base_url = "https://api.moonshot.cn/v1/chat/completions"
        elif self.provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY", "")
            model = os.getenv("QWEN_VISION_MODEL", os.getenv("QWEN_MODEL", "qwen3.5-plus"))
            base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        elif self.provider == "glm":
            api_key = os.getenv("GLM_API_KEY", "")
            model = os.getenv("GLM_VISION_MODEL", os.getenv("GLM_MODEL", "glm-4.6v"))
            base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        elif self.provider == "doubao":
            api_key = os.getenv("DOUBAO_API_KEY", "")
            model = os.getenv("DOUBAO_VISION_MODEL", os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-pro-260215"))
            base_url = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        elif self.provider == "minmax":
            api_key = os.getenv("MINMAX_API_KEY", "")
            model = os.getenv("MINMAX_VISION_MODEL", os.getenv("MINMAX_MODEL", "MiniMax-VL-01"))
            base_url = os.getenv("MINMAX_BASE_URL", "https://api.minimaxi.com/v1/chat/completions")
        elif self.provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            model = os.getenv("ANTHROPIC_VISION_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6"))
            base_url = "https://api.anthropic.com/v1/messages"
        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            model = os.getenv("GEMINI_VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview"))
            base_url = "https://generativelanguage.googleapis.com/v1beta"
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
            base_url = "https://api.openai.com/v1/chat/completions"

        # Cloubic 覆盖 + 降级链
        if via_cloubic:
            api_key = os.getenv("CLOUBIC_API_KEY", "")
            base_url = os.getenv("CLOUBIC_BASE_URL", "https://api.cloubic.com/v1/chat/completions")
            model_chain = _get_cloubic_model_chain(self.provider)
            if model_chain:
                model = model_chain[0]

        # model_override 覆盖（别名模式）
        if self.model_override:
            model = self.model_override

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

        # Vision 降级链：别名模式锁定单模型
        if self.no_fallback:
            models_to_try = [model]
        elif via_cloubic:
            models_to_try = _get_cloubic_model_chain(self.provider) or [model]
        else:
            models_to_try = [model]  # Vision 不走 FALLBACK（视觉模型与文字模型不同）
        last_error: Exception | None = None
        route_tag = "Cloubic Vision" if via_cloubic else f"{self.provider} Vision"

        for model_idx, try_model in enumerate(models_to_try):
            can_retry = model_idx < len(models_to_try) - 1
            try:
                payload = {"model": try_model, "messages": messages,
                           "temperature": self.temperature, "max_tokens": self.max_tokens}
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                proxies = {} if via_cloubic else _get_llm_proxies(self.provider)
                resp = self._safe_post(base_url, headers=headers, json=payload,
                                      timeout=self._timeout(is_vision=True),
                                      proxies=proxies)
                if not resp.ok and can_retry:
                    logger.warning("[%s] %s model=%s HTTP %d，降级到 %s",
                                   route_tag, self.provider, try_model, resp.status_code, models_to_try[model_idx + 1])
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    if isinstance(content, list):
                        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                        content = " ".join(parts)
                    return content if isinstance(content, str) else str(content)
                if can_retry:
                    logger.warning("[%s] %s model=%s 无choices，降级到 %s",
                                   route_tag, self.provider, try_model, models_to_try[model_idx + 1])
                    continue
                return None
            except Exception as e:
                last_error = e
                if can_retry:
                    logger.warning("[%s] %s model=%s 异常: %s，降级到 %s",
                                   route_tag, self.provider, try_model, e, models_to_try[model_idx + 1])
                    continue
                raise
        if last_error:
            raise last_error
        return None
