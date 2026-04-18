"""Phase 1 BaseExpert - 4 个专业视角 LLM 角色的共享基类。"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from ...llm_client import LLMRouter
from ...prompts_v3 import load_prompt
from ..phase0_data import ReportBundle

logger = logging.getLogger(__name__)


@dataclass
class ExpertResult:
    """专家视角分析输出。"""
    role: str                        # 角色 id (structure_expert/wave_expert/...)
    role_cn: str                     # 中文角色名
    provider: str                    # 使用的 LLM provider
    score: float = 50.0              # 0-100
    analysis: str = ""
    risk: str = ""
    key_data: dict[str, Any] = field(default_factory=dict)  # 角色专属结构化输出
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseExpert:
    """4 个专家共享骨架: 读 ReportBundle → 发 prompt → 解析 JSON 输出。"""

    role: str = "base_expert"
    role_cn: str = "基础专家"
    prompt_file: str = ""           # prompts_v3/<prompt_file>.txt
    bundle_role_key: str = ""       # ReportBundle.get_for_role 的 key
    temperature: float = 0.3
    max_tokens: int = 1000
    timeout_sec: float = 60.0

    def __init__(self, provider: str = "grok", model_override: str | None = None):
        self.provider = provider
        self.model_override = model_override
        self.router = LLMRouter(
            provider=provider,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout_sec=self.timeout_sec,
            model_override=model_override,
        )

    def build_prompt(self, bundle: ReportBundle, extra_context: str = "") -> str:
        system = load_prompt(self.prompt_file)
        report_text = bundle.get_for_role(self.bundle_role_key)
        parts = [
            system,
            "",
            "═══════════════════════════════",
            f"标的: {bundle.symbol} {bundle.name}",
            "═══════════════════════════════",
            "",
            "【客观量化报告】",
            report_text,
        ]
        if extra_context:
            parts.extend(["", "【补充上下文】", extra_context])
        parts.extend([
            "",
            "请严格按上方要求输出 JSON,仅输出 JSON 对象,不要附加其他文字:",
        ])
        return "\n".join(parts)

    def analyze(self, bundle: ReportBundle, extra_context: str = "") -> ExpertResult:
        prompt = self.build_prompt(bundle, extra_context)
        logger.info("[Expert] %s provider=%s 提交 | prompt=%d字",
                    self.role, self.provider, len(prompt))
        raw = ""
        try:
            raw = self.router._chat(prompt, multi_turn=True) or ""
        except Exception as e:
            logger.warning("[Expert] %s 调用失败: %s", self.role, e)

        parsed = self._parse_json(raw)
        result = ExpertResult(
            role=self.role,
            role_cn=self.role_cn,
            provider=self.provider,
            raw_response=raw,
        )
        if parsed:
            result.score = float(parsed.get("score", 50.0))
            result.analysis = str(parsed.get("analysis", ""))
            result.risk = str(parsed.get("risk", ""))
            # 其它字段全部入 key_data
            reserved = {"score", "analysis", "risk"}
            result.key_data = {k: v for k, v in parsed.items() if k not in reserved}
        else:
            result.analysis = (raw[:400] or "(LLM 无响应)")
            result.risk = "LLM 输出解析失败, 数据可能不完整"
        return result

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        if not text:
            return None
        s = text.strip()
        # 去掉 ```json 包裹
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s)
        if m:
            s = m.group(1)
        # 提取第一个 { ... } 块
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
