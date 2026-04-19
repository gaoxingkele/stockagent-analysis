"""K 走势结构分析师 - 多模态视觉版。

不同于其他专家, 此角色会:
1. 基于 run_dir/data/kline/ 生成日+周 K 图(主图+MACD+RSI 副图)
2. 合成一张大图传给 vision LLM
3. 用修订版 prompt(放宽评分锚点, 避免系统性偏严)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ...llm_client import LLMRouter
from ...prompts_v3 import load_prompt
from ..phase0_data import ReportBundle
from ..kline_charts import (
    generate_expert_charts, merge_charts_vertically, image_to_base64,
)
from .base_expert import BaseExpert, ExpertResult

logger = logging.getLogger(__name__)


class StructureExpert(BaseExpert):
    role = "structure_expert"
    role_cn = "K 走势结构分析师"
    prompt_file = "expert_structure"
    bundle_role_key = "structure_expert"
    # 覆盖 BaseExpert 的默认 provider 能力, vision 优先
    max_tokens = 1400

    def __init__(self, provider: str = "grok", model_override: str | None = None,
                 run_dir: Path | None = None):
        super().__init__(provider=provider, model_override=model_override)
        self.run_dir = Path(run_dir) if run_dir else None

    def analyze(self, bundle: ReportBundle, extra_context: str = "") -> ExpertResult:
        """视觉增强分析: 生成 K 图 → 合成 → vision LLM 研判。"""
        if not self.run_dir or not self.run_dir.exists():
            logger.warning("[%s] run_dir 缺失, 降级到文本模式", self.role)
            return super().analyze(bundle, extra_context)

        # 1) 生成图片
        try:
            charts = generate_expert_charts(self.run_dir, bundle.symbol, bundle.name,
                                             timeframes=("day", "week"))
        except Exception as e:
            logger.warning("[%s] 绘图失败 %s, 降级文本模式", self.role, e)
            return super().analyze(bundle, extra_context)

        day_path = charts.get("day")
        week_path = charts.get("week")
        if not (day_path and week_path):
            logger.warning("[%s] 日周图缺, 降级文本模式", self.role)
            return super().analyze(bundle, extra_context)

        # 2) 合成一张大图
        merged_path = self.run_dir / "charts" / "expert" / "day_week_merged.png"
        try:
            merge_charts_vertically(day_path, week_path, merged_path)
        except Exception as e:
            logger.warning("[%s] 合成失败 %s, 降级", self.role, e)
            return super().analyze(bundle, extra_context)

        # 3) 构造 prompt
        system = load_prompt(self.prompt_file)
        report_text = bundle.get_for_role(self.bundle_role_key)
        prompt = (
            f"{system}\n\n"
            f"═══════════════════════════════\n"
            f"标的: {bundle.symbol} {bundle.name}\n"
            f"═══════════════════════════════\n\n"
            f"【客观量化报告】\n{report_text}\n\n"
            f"【附图】\n上半部分为日线(100 根)K 图, 下半部分为周线(80 根)K 图。\n"
            f"主图: K 线 + MA5/10/20/60 + 布林带 + Ichimoku 云 + SAR + 成交量\n"
            f"副图: MACD (DIF/DEA/柱) + RSI(14) + 自动趋势线\n\n"
            f"请先观察图像, 再结合量化报告, 按要求输出 JSON。\n"
        )

        # 4) 调用 vision LLM
        vision_router = LLMRouter(
            provider=self.provider,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout_sec=90.0,
            model_override=self.model_override,
        )
        raw = ""
        try:
            if vision_router.supports_vision():
                img_b64, mime = image_to_base64(merged_path)
                raw = vision_router.chat_with_image(prompt, img_b64, mime) or ""
                logger.info("[%s] vision 调用成功 provider=%s 返回 %d 字", self.role, self.provider, len(raw))
            else:
                # 降级到 vision fallback
                import os
                fallback = os.getenv("VISION_FALLBACK_PROVIDER", "kimi")
                logger.info("[%s] %s 不支持 vision, 降级到 %s", self.role, self.provider, fallback)
                fallback_router = LLMRouter(provider=fallback, temperature=self.temperature,
                                             max_tokens=self.max_tokens, request_timeout_sec=90.0)
                if fallback_router.supports_vision():
                    img_b64, mime = image_to_base64(merged_path)
                    raw = fallback_router.chat_with_image(prompt, img_b64, mime) or ""
                else:
                    # 实在不行降级为纯文本
                    logger.warning("[%s] 无可用 vision provider, 纯文本模式", self.role)
                    raw = vision_router._chat(prompt, multi_turn=True) or ""
        except Exception as e:
            logger.warning("[%s] vision 调用失败: %s", self.role, e)

        # 5) 解析 JSON
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
            reserved = {"score", "analysis", "risk"}
            result.key_data = {k: v for k, v in parsed.items() if k not in reserved}
        else:
            result.analysis = (raw[:400] or "(vision 无响应)")
            result.risk = "vision 输出解析失败"

        result.key_data["_chart_path"] = str(merged_path.relative_to(self.run_dir))
        return result
