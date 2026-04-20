"""波浪理论分析师 - 多模态视觉版。

架构与 StructureExpert 对称:
1. 基于 run_dir/data/kline/week.csv + month.csv 生成专门的波浪图
   (简洁 K + ZigZag 摆动点 + 斐波那契回撤 + RSI+ADX 副图)
2. 图单独一张 wave.png(不和 K 走势的 merged.png 共用)
3. 调用 vision LLM 做波浪识别
"""
from __future__ import annotations

import logging
from pathlib import Path

from ...llm_client import LLMRouter
from ...prompts_v3 import load_prompt
from ..phase0_data import ReportBundle
from ..kline_charts import generate_wave_expert_chart, image_to_base64
from .base_expert import BaseExpert, ExpertResult

logger = logging.getLogger(__name__)


class WaveExpert(BaseExpert):
    role = "wave_expert"
    role_cn = "波浪理论分析师"
    prompt_file = "expert_wave"
    bundle_role_key = "wave_expert"
    max_tokens = 1500
    temperature = 0.3

    def __init__(self, provider: str = "grok", model_override: str | None = None,
                 run_dir: Path | None = None):
        super().__init__(provider=provider, model_override=model_override)
        self.run_dir = Path(run_dir) if run_dir else None

    def analyze(self, bundle: ReportBundle, extra_context: str = "") -> ExpertResult:
        if not self.run_dir or not self.run_dir.exists():
            logger.warning("[%s] run_dir 缺失, 降级文本模式", self.role)
            return super().analyze(bundle, extra_context)

        # 生成波浪专用图(周+月)
        try:
            chart_path = generate_wave_expert_chart(self.run_dir, bundle.symbol, bundle.name)
        except Exception as e:
            logger.warning("[%s] 绘图失败 %s, 降级文本", self.role, e)
            return super().analyze(bundle, extra_context)

        if not chart_path or not Path(chart_path).exists():
            logger.warning("[%s] 波浪图缺失(可能 week/month csv 不存在), 降级文本", self.role)
            return super().analyze(bundle, extra_context)
        logger.info("[%s] 合成波浪图: %s (%d KB)",
                    self.role, chart_path.name, chart_path.stat().st_size // 1024)

        system = load_prompt(self.prompt_file)
        report_text = bundle.get_for_role(self.bundle_role_key)
        prompt = (
            f"{system}\n\n"
            f"═══════════════════════════════\n"
            f"标的: {bundle.symbol} {bundle.name}\n"
            f"═══════════════════════════════\n\n"
            f"【客观量化报告(辅助参考)】\n{report_text}\n\n"
            f"【附图说明】\n图片是 2 列布局:\n"
            f"  左列=周线(100 根), 右列=月线(60 根)\n"
            f"  主图: K + ZigZag 摆动点(带编号 1-6) + 斐波那契回撤 + 前高/前低\n"
            f"  副图: RSI(紫) + ADX(橙)\n"
            f"请先按【月线>周线】优先级观察图像摆动点, 识别当前波浪位置, 输出 JSON。\n"
        )

        router = LLMRouter(
            provider=self.provider,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout_sec=90.0,
            model_override=self.model_override,
        )
        raw = ""
        try:
            if router.supports_vision():
                img_b64, mime = image_to_base64(chart_path)
                raw = router.chat_with_image(prompt, img_b64, mime) or ""
                logger.info("[%s] vision 成功 provider=%s 返回 %d 字",
                            self.role, self.provider, len(raw))
            else:
                import os
                fb = os.getenv("VISION_FALLBACK_PROVIDER", "kimi")
                fb_router = LLMRouter(provider=fb, temperature=self.temperature,
                                       max_tokens=self.max_tokens, request_timeout_sec=90.0)
                if fb_router.supports_vision():
                    img_b64, mime = image_to_base64(chart_path)
                    raw = fb_router.chat_with_image(prompt, img_b64, mime) or ""
                else:
                    raw = router._chat(prompt, multi_turn=True) or ""
        except Exception as e:
            logger.warning("[%s] vision 失败: %s", self.role, e)

        parsed = self._parse_json(raw)
        result = ExpertResult(
            role=self.role, role_cn=self.role_cn,
            provider=self.provider, raw_response=raw,
        )
        if parsed:
            result.score = float(parsed.get("score", 50.0))
            result.analysis = str(parsed.get("analysis", ""))
            result.risk = str(parsed.get("risk", ""))
            reserved = {"score", "analysis", "risk"}
            result.key_data = {k: v for k, v in parsed.items() if k not in reserved}
        else:
            result.analysis = (raw[:400] or "(波浪 vision 无响应)")
            result.risk = "波浪 vision 输出解析失败"

        result.key_data["_chart_path"] = str(chart_path.relative_to(self.run_dir))
        return result
