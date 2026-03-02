# -*- coding: utf-8 -*-
"""并行化执行引擎：每个 LLM Provider 作为独立线程同时执行权重分配+研判增强+评分。"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import AgentBaseResult
from .llm_client import LLMRouter, assign_agent_weights, score_agent_analysis, _supports_vision, _DEFAULT_VISION_FALLBACK

logger = logging.getLogger(__name__)


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


@dataclass
class ProviderResult:
    """每个 Provider 线程的输出结果。"""
    provider: str
    weights: dict[str, float] = field(default_factory=dict)
    enrichments: dict[str, str] = field(default_factory=dict)   # {agent_id: enriched_text}
    scores: dict[str, float] = field(default_factory=dict)      # {agent_id: score_0_100}
    vision_results: dict[str, str] = field(default_factory=dict) # {agent_id: vision_text}
    error: str | None = None
    elapsed_sec: float = 0.0


def _save_provider_result(run_dir: Path, provider: str, result: ProviderResult) -> None:
    """每个Provider完成后立即将结果序列化存盘，实现增量持久化。"""
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
    """从 run_dir/data/provider_results/ 加载已完成的Provider结果（跳过有error的）。"""
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


def _provider_worker(
    provider: str,
    router: LLMRouter,
    base_results: dict[str, AgentBaseResult],
    analyst_cfg: list[dict[str, Any]],
    symbol: str,
    name: str,
    task_summary: str | None,
    progress: ProviderProgress,
) -> ProviderResult:
    """单个 Provider 线程的工作函数：权重分配 → 研判增强 → 独立评分。"""
    t0 = time.time()
    result = ProviderResult(provider=provider)
    agent_count = len(base_results)
    progress.enrich_total = agent_count
    progress.score_total = agent_count

    try:
        # ── Step 1: 权重分配 ──
        agent_list = [
            {
                "agent_id": c["agent_id"],
                "role": c["role"],
                "dim_code": c.get("dim_code", ""),
                "weight": float(c.get("weight", 1.0 / max(len(analyst_cfg), 1))),
            }
            for c in analyst_cfg
        ]
        result.weights = assign_agent_weights(
            router, agent_list, symbol, name,
            provider_name=provider, task_summary=task_summary,
        )
        progress.weight_done = True

        # ── Step 2: 研判增强 / 视觉分析 ──
        has_vision = router.supports_vision()
        for agent_id, base in base_results.items():
            try:
                if (base.agent_type == "kline_vision"
                        and base.image_b64
                        and base.vision_prompt
                        and has_vision):
                    # K线视觉：发送图像给视觉模型
                    text = router.chat_with_image(base.vision_prompt, base.image_b64)
                    if text:
                        result.vision_results[agent_id] = text.strip()
                        result.enrichments[agent_id] = text.strip()
                    else:
                        # 视觉调用失败，降级到文本增强
                        text = router.enrich_reason(
                            base.role, symbol, base.reason,
                            data_context=base.data_context,
                        )
                        if text:
                            result.enrichments[agent_id] = text.strip()
                else:
                    # 普通文本增强
                    text = router.enrich_reason(
                        base.role, symbol, base.reason,
                        data_context=base.data_context,
                    )
                    if text:
                        result.enrichments[agent_id] = text.strip()
            except Exception as e:
                logger.warning(
                    "enrich failed provider=%s agent=%s: %s", provider, agent_id, e,
                )
            progress.enrich_done += 1

        # ── Step 3: 独立评分 ──
        for agent_id, base in base_results.items():
            try:
                reason_for_score = result.enrichments.get(agent_id, base.reason)
                result.scores[agent_id] = score_agent_analysis(
                    router, base.role, agent_id, symbol, name,
                    reason_for_score, base.data_context,
                )
            except Exception as e:
                logger.warning(
                    "score failed provider=%s agent=%s: %s", provider, agent_id, e,
                )
                result.scores[agent_id] = 50.0
            progress.score_done += 1

    except Exception as e:
        logger.error("provider worker %s failed: %s", provider, e)
        result.error = str(e)
        progress.error = str(e)

    result.elapsed_sec = time.time() - t0
    progress.finished = True
    return result


def _display_progress(
    progresses: dict[str, ProviderProgress],
    print_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    """后台守护线程：每2秒刷新一次进度表。"""
    while not stop_event.is_set():
        stop_event.wait(2.0)
        if stop_event.is_set():
            break
        _print_progress_table(progresses, print_lock)


def _print_progress_table(
    progresses: dict[str, ProviderProgress],
    print_lock: threading.Lock,
) -> None:
    finished_count = sum(1 for p in progresses.values() if p.finished)
    total = len(progresses)
    lines = [f"\n[并行] {total}个大模型子任务运行中 (已完成 {finished_count}/{total})"]
    for provider, prog in progresses.items():
        elapsed = time.time() - prog.start_time
        w_status = "[OK]权重" if prog.weight_done else "权重中"
        e_status = f"研判 {prog.enrich_done:2d}/{prog.enrich_total}" if prog.enrich_total else "研判  0/0"
        s_status = f"评分 {prog.score_done:2d}/{prog.score_total}" if prog.score_total else "评分  0/0"
        if prog.error:
            state = f"[X]失败 {elapsed:.0f}s"
        elif prog.finished:
            state = f"[OK]完成 {elapsed:.0f}s"
        else:
            state = f"运行中 {elapsed:.0f}s"
        lines.append(f"  {provider:10s} | {w_status} | {e_status} | {s_status} | {state}")
    with print_lock:
        print("\n".join(lines), flush=True)


def _ensure_vision_fallback(
    llm_routers: dict[str, LLMRouter],
    base_results: dict[str, AgentBaseResult],
) -> dict[str, LLMRouter]:
    """若当前 providers 均不支持视觉但有K线图需处理，自动添加视觉回退 provider。"""
    has_vision_need = any(
        b.agent_type == "kline_vision" and b.image_b64
        for b in base_results.values()
    )
    if not has_vision_need:
        return llm_routers

    has_vision_provider = any(r.supports_vision() for r in llm_routers.values())
    if has_vision_provider:
        return llm_routers

    fallback_p = os.getenv("VISION_FALLBACK_PROVIDER", _DEFAULT_VISION_FALLBACK)
    fallback_key = os.getenv(f"{fallback_p.upper()}_API_KEY", "").strip()
    if not fallback_key or not _supports_vision(fallback_p):
        return llm_routers

    # 如果回退provider已在router列表中，无需重复添加
    if fallback_p in llm_routers:
        return llm_routers

    ref = next(iter(llm_routers.values()))
    fallback_router = LLMRouter(
        provider=fallback_p,
        temperature=ref.temperature,
        max_tokens=ref.max_tokens,
        request_timeout_sec=ref.request_timeout_sec,
    )
    routers = dict(llm_routers)
    routers[fallback_p] = fallback_router
    logger.info(
        "[并行] 当前providers无视觉能力，添加 %s 作为视觉回退", fallback_p,
    )
    print(f"[并行] 添加 {fallback_p} 视觉回退provider", flush=True)
    return routers


def run_providers_parallel(
    llm_routers: dict[str, LLMRouter],
    base_results: dict[str, AgentBaseResult],
    analyst_cfg: list[dict[str, Any]],
    symbol: str,
    name: str,
    task_summary: str | None,
    run_dir: Path | None = None,
) -> dict[str, ProviderResult]:
    """主调度：为每个 Provider 启动线程并行执行，收集结果。支持增量持久化与断点续传。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 检查视觉回退
    routers = _ensure_vision_fallback(llm_routers, base_results)
    all_providers = list(routers.keys())

    # ── 断点续传：加载已完成的Provider结果 ──
    cached_results: dict[str, ProviderResult] = {}
    if run_dir:
        cached_results = _load_existing_results(run_dir, all_providers)
        if cached_results:
            cached_list = list(cached_results.keys())
            print(f"[断点续传] 已加载 {len(cached_results)} 个已完成Provider结果: {cached_list}", flush=True)

    # 计算待运行的Providers
    pending_providers = [p for p in all_providers if p not in cached_results]

    if not pending_providers:
        print("[断点续传] 所有Provider已有缓存结果，跳过并行执行", flush=True)
        return cached_results

    if cached_results:
        print(f"[并行] 待运行Providers: {pending_providers} (跳过已完成: {list(cached_results.keys())})", flush=True)

    print_lock = threading.Lock()
    stop_event = threading.Event()

    progresses = {
        p: ProviderProgress(provider=p, start_time=time.time())
        for p in pending_providers
    }

    # 启动进度显示守护线程
    display_thread = threading.Thread(
        target=_display_progress,
        args=(progresses, print_lock, stop_event),
        daemon=True,
    )
    display_thread.start()

    new_results: dict[str, ProviderResult] = {}

    with ThreadPoolExecutor(max_workers=len(pending_providers)) as executor:
        futures = {
            executor.submit(
                _provider_worker,
                p, routers[p], base_results, analyst_cfg,
                symbol, name, task_summary,
                progresses[p],
            ): p
            for p in pending_providers
        }
        for future in as_completed(futures):
            p = futures[future]
            try:
                new_results[p] = future.result()
            except Exception as e:
                logger.error("provider %s future exception: %s", p, e)
                new_results[p] = ProviderResult(provider=p, error=str(e))
            # ── 增量持久化：每个Provider完成后立即存盘 ──
            if run_dir:
                _save_provider_result(run_dir, p, new_results[p])

    stop_event.set()
    display_thread.join(timeout=3)

    # 打印最终状态
    _print_progress_table(progresses, print_lock)

    # 合并缓存结果 + 新结果
    results = {**cached_results, **new_results}

    # 汇总统计
    ok_count = sum(1 for r in results.values() if not r.error)
    fail_count = sum(1 for r in results.values() if r.error)
    total_elapsed = max((r.elapsed_sec for r in results.values()), default=0)
    cached_note = f" 缓存:{len(cached_results)}" if cached_results else ""
    print(
        f"\n[并行] 完成: {ok_count}成功 {fail_count}失败{cached_note} | 总耗时 {total_elapsed:.1f}s",
        flush=True,
    )

    return results
