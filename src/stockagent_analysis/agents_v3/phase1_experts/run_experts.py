"""Phase 1 并行调度: 4 个专家角色同时分析 ReportBundle。"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

from ..phase0_data import ReportBundle
from .base_expert import ExpertResult
from .structure_expert import StructureExpert
from .wave_expert import WaveExpert
from .intraday_t_expert import IntradayTExpert
from .martingale_expert import MartingaleExpert

logger = logging.getLogger(__name__)


def run_all_experts(
    bundle: ReportBundle,
    providers: dict[str, str] | None = None,
    parallel: bool = True,
    run_dir: "Path | None" = None,
) -> dict[str, ExpertResult]:
    """并行执行 4 个专家角色。

    Args:
        bundle: Phase 0 产出的 6 份报告集合
        providers: 角色 → LLM provider 映射; 未提供则用默认:
            structure_expert → grok (vision, 需要 run_dir 生成 K 图)
            wave_expert → grok
            intraday_t_expert → doubao
            martingale_expert → grok
        parallel: True=并行, False=串行(调试用)
        run_dir: v3 run 目录, StructureExpert 依赖它读 kline/ 并生成图

    Returns:
        {role: ExpertResult}
    """
    default_providers = {
        "structure_expert": "grok",
        "wave_expert": "grok",
        "intraday_t_expert": "doubao",
        "martingale_expert": "grok",
    }
    providers = providers or default_providers

    experts = [
        StructureExpert(provider=providers.get("structure_expert", "grok"), run_dir=run_dir),
        WaveExpert(provider=providers.get("wave_expert", "grok"), run_dir=run_dir),
        IntradayTExpert(provider=providers.get("intraday_t_expert", "doubao")),
        MartingaleExpert(provider=providers.get("martingale_expert", "grok")),
    ]

    results: dict[str, ExpertResult] = {}

    def _one(expert):
        logger.info("[Phase1] 启动 %s via %s", expert.role, expert.provider)
        try:
            return expert.role, expert.analyze(bundle)
        except Exception as e:
            logger.warning("[Phase1] %s 失败: %s", expert.role, e)
            return expert.role, ExpertResult(
                role=expert.role,
                role_cn=expert.role_cn,
                provider=expert.provider,
                analysis=f"(角色执行失败: {e})",
                risk="角色失败, 建议用量化兜底",
            )

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(experts)) as pool:
            for role, res in pool.map(_one, experts):
                results[role] = res
    else:
        for expert in experts:
            role, res = _one(expert)
            results[role] = res

    return results
