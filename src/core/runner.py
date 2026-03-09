# -*- coding: utf-8 -*-
"""并行执行基础设施：超时控制、进度追踪、断点续传、候选 Provider 切换。

通用工具集，不含领域业务逻辑（worker 函数由业务层实现）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .router import LLMRouter

logger = logging.getLogger(__name__)

# 默认每个 Provider 的最大超时（秒）
_DEFAULT_PROVIDER_TIMEOUT_SEC = 720

# 单 Agent 调用超时（秒）
_AGENT_CALL_TIMEOUT_SEC = 300

# 备选 Provider 列表（按优先级排序）
_FALLBACK_PROVIDERS = ["doubao", "qwen", "gemini", "minmax"]

# 候选 Provider 最大尝试次数
_MAX_CANDIDATE_ATTEMPTS = 4


class _ProviderCancelled(Exception):
    """Worker 线程检测到取消信号时抛出，用于快速退出。"""
    pass


def _timed_call(fn, timeout_sec: float, cancel_event: threading.Event | None = None):
    """在子线程中执行 fn()，超过 timeout_sec 秒返回 (None, True)。
    成功返回 (result, False)，异常返回 (None, False) 并抛出原异常。
    如果传入 cancel_event，会每秒检查取消信号，提前退出。"""
    result_box: dict[str, Any] = {"value": None, "error": None}

    def wrapper():
        try:
            result_box["value"] = fn()
        except Exception as e:
            result_box["error"] = e

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()

    if cancel_event:
        deadline = time.time() + timeout_sec
        while t.is_alive() and time.time() < deadline:
            if cancel_event.is_set():
                return None, True
            t.join(timeout=1.0)
    else:
        t.join(timeout=timeout_sec)

    if t.is_alive():
        return None, True  # 超时
    if result_box["error"]:
        raise result_box["error"]
    return result_box["value"], False


def _create_fallback_router(
    fallback_provider: str,
    ref_router: LLMRouter,
) -> LLMRouter | None:
    """为备选 Provider 创建 LLMRouter，API_KEY 不存在则返回 None。"""
    api_key = os.getenv(f"{fallback_provider.upper()}_API_KEY", "").strip()
    if not api_key:
        return None
    return LLMRouter(
        provider=fallback_provider,
        temperature=ref_router.temperature,
        max_tokens=ref_router.max_tokens,
        request_timeout_sec=ref_router.request_timeout_sec,
    )


@dataclass
class ProviderProgress:
    """每个 Provider 线程的进度状态。"""
    provider: str
    weight_done: bool = False
    enrich_done: int = 0
    enrich_total: int = 0
    score_done: int = 0
    score_total: int = 0
    finished: bool = False
    error: str | None = None
    start_time: float = field(default_factory=time.time)
    current_stage: str = ""
    current_agent: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    agent_scores: dict[str, float] = field(default_factory=dict)
    agent_weights: dict[str, float] = field(default_factory=dict)
    agent_fallbacks: dict[str, str] = field(default_factory=dict)
    agents_in_progress: set[str] = field(default_factory=set)
    _agents_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class ProviderResult:
    """每个 Provider 线程的输出结果。"""
    provider: str
    weights: dict[str, float] = field(default_factory=dict)
    enrichments: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    vision_results: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    elapsed_sec: float = 0.0


def _save_provider_result(run_dir: Path, provider: str, result: ProviderResult) -> None:
    """每个 Provider 完成后立即将结果序列化存盘，实现增量持久化。"""
    results_dir = run_dir / "data" / "provider_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{provider}.json"
    data = {
        "provider": result.provider,
        "weights": result.weights,
        "enrichments": result.enrichments,
        "scores": result.scores,
        "vision_results": result.vision_results,
        "error": result.error,
        "elapsed_sec": result.elapsed_sec,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("provider result saved: %s -> %s", provider, path)


def _load_existing_results(run_dir: Path, providers: list[str]) -> dict[str, ProviderResult]:
    """从 run_dir/data/provider_results/ 加载已完成的 Provider 结果（跳过有 error 的）。"""
    results_dir = run_dir / "data" / "provider_results"
    if not results_dir.exists():
        return {}
    loaded: dict[str, ProviderResult] = {}
    for p in providers:
        path = results_dir / f"{p}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("error"):
                logger.info("skip cached result for %s (has error: %s)", p, data["error"])
                continue
            loaded[p] = ProviderResult(
                provider=data["provider"],
                weights=data.get("weights", {}),
                enrichments=data.get("enrichments", {}),
                scores=data.get("scores", {}),
                vision_results=data.get("vision_results", {}),
                error=None,
                elapsed_sec=data.get("elapsed_sec", 0.0),
            )
            logger.info("loaded cached result for %s (elapsed=%.1fs)", p, loaded[p].elapsed_sec)
        except Exception as e:
            logger.warning("failed to load cached result for %s: %s", p, e)
    return loaded
