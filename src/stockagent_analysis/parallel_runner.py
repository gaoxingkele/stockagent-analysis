# -*- coding: utf-8 -*-
"""并行化执行引擎：每个 LLM Provider 作为独立线程同时执行权重分配+研判增强+评分。"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents import AgentBaseResult
from .llm_client import LLMRouter, assign_agent_weights, score_agent_analysis, _supports_vision, _DEFAULT_VISION_FALLBACK

logger = logging.getLogger(__name__)

# 默认每个 Provider 的最大超时（秒），可通过 project.json llm.provider_timeout_sec 覆盖
_DEFAULT_PROVIDER_TIMEOUT_SEC = 720

# 单Agent调用超时（秒）：单次enrich或score调用超过此时间则切换到备选Provider
_AGENT_CALL_TIMEOUT_SEC = 300

# 备选Provider列表（按优先级排序），用于单Agent超时后的切换
_FALLBACK_PROVIDERS = ["doubao", "qwen", "gemini", "minmax"]

# 候选Provider最大尝试次数
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
        # 每秒轮询，及时响应取消信号
        deadline = time.time() + timeout_sec
        while t.is_alive() and time.time() < deadline:
            if cancel_event.is_set():
                return None, True  # 外部取消，视为超时
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
    """为备选Provider创建LLMRouter，API_KEY不存在则返回None。"""
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
    current_stage: str = ""      # 当前阶段: "权重", "研判", "评分"
    current_agent: str = ""      # 当前正在处理的 agent_id
    cancel_event: threading.Event = field(default_factory=threading.Event)  # 超时取消信号
    agent_scores: dict[str, float] = field(default_factory=dict)   # 逐agent评分结果
    agent_weights: dict[str, float] = field(default_factory=dict)  # 权重分配结果
    agent_fallbacks: dict[str, str] = field(default_factory=dict)  # {agent_id: fallback_provider} 超时切换记录


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
    vision_fallback_router: LLMRouter | None = None,
    agent_call_timeout_sec: float = _AGENT_CALL_TIMEOUT_SEC,
    fallback_providers: list[str] | None = None,
) -> ProviderResult:
    """单个 Provider 线程的工作函数：权重分配 → 研判增强 → 独立评分。
    当单个Agent调用超过 agent_call_timeout_sec 时，自动切换到备选Provider。"""
    t0 = time.time()
    result = ProviderResult(provider=provider)
    agent_count = len(base_results)
    progress.enrich_total = agent_count
    progress.score_total = agent_count

    cancelled = progress.cancel_event
    _fb_providers = fallback_providers or _FALLBACK_PROVIDERS
    # 备选Router缓存（避免重复创建）
    _fb_router_cache: dict[str, LLMRouter | None] = {}

    def _enrich_with_fallback(agent_id: str, base: AgentBaseResult, is_vision: bool,
                              vision_rtr: LLMRouter | None) -> str | None:
        """研判增强：主Router→重试一次→随机候选Provider(最多4个)。"""
        def _do_call(rtr: LLMRouter) -> str | None:
            if is_vision and vision_rtr:
                # 视觉agent用vision_router
                actual_rtr = vision_rtr if vision_rtr else rtr
                txt = actual_rtr.chat_with_image(base.vision_prompt, base.image_b64)
                if txt:
                    result.vision_results[agent_id] = txt.strip()
                    return txt.strip()
                # 视觉失败降级文本
                txt = rtr.enrich_reason(base.role, symbol, base.reason,
                                        data_context=base.data_context)
                return txt.strip() if txt else None
            else:
                txt = rtr.enrich_reason(base.role, symbol, base.reason,
                                        data_context=base.data_context)
                return txt.strip() if txt else None

        # 第1次：主Router
        try:
            text, timed_out = _timed_call(lambda: _do_call(router), agent_call_timeout_sec, cancelled)
            if not timed_out and text:
                return text
        except Exception as e:
            logger.warning("enrich primary error provider=%s agent=%s: %s",
                          provider, agent_id, e)

        if cancelled.is_set():
            return None

        # 第2次：重试主Router
        logger.info("[重试] %s agent=%s 研判首次失败, 重试中...", provider, agent_id)
        try:
            text, timed_out = _timed_call(lambda: _do_call(router), agent_call_timeout_sec, cancelled)
            if not timed_out and text:
                return text
        except Exception as e:
            logger.warning("enrich retry error provider=%s agent=%s: %s",
                          provider, agent_id, e)

        if cancelled.is_set():
            return None

        # 第3~6次：随机候选Provider（最多4个）
        fb_list = [fp for fp in _fb_providers if fp != provider]
        random.shuffle(fb_list)
        tried = 0
        logger.info("[切换] %s agent=%s 研判重试仍失败, 尝试候选Provider (池: %s)",
                   provider, agent_id, fb_list[:_MAX_CANDIDATE_ATTEMPTS])
        for fp in fb_list:
            if tried >= _MAX_CANDIDATE_ATTEMPTS or cancelled.is_set():
                break
            if fp not in _fb_router_cache:
                _fb_router_cache[fp] = _create_fallback_router(fp, router)
            fb_rtr = _fb_router_cache[fp]
            if fb_rtr is None:
                continue
            tried += 1
            try:
                text, timed_out2 = _timed_call(lambda r=fb_rtr: _do_call(r), agent_call_timeout_sec, cancelled)
                if not timed_out2 and text:
                    progress.agent_fallbacks[agent_id] = fp
                    logger.info("[切换] %s agent=%s 研判已切换到 %s", provider, agent_id, fp)
                    print(f"[切换] {provider}→{fp} agent={agent_id} (研判)", flush=True)
                    return text
            except Exception:
                logger.warning("[切换] %s agent=%s 候选%s也失败", provider, agent_id, fp)
                continue

        # 全部失败（默认+重试+候选共5次）
        logger.warning("[放弃] %s agent=%s 研判全部失败(尝试%d次)", provider, agent_id, 2 + tried)
        return None

    def _score_with_fallback(agent_id: str, base: AgentBaseResult,
                             reason_text: str) -> float:
        """评分：主Router→重试一次→随机候选Provider(最多4个)→已成功agent平均分。"""
        def _do_score(rtr: LLMRouter) -> float:
            return score_agent_analysis(
                rtr, base.role, agent_id, symbol, name,
                reason_text, base.data_context,
            )

        def _avg_existing_score() -> float:
            """全部失败时，采用已成功agent的平均分数。"""
            existing = list(progress.agent_scores.values())
            if existing:
                avg = sum(existing) / len(existing)
                logger.info("[平均分] %s agent=%s 全部失败, 使用已成功%d个agent平均分 %.1f",
                           provider, agent_id, len(existing), avg)
                return avg
            return 50.0

        # 第1次：主Router
        try:
            score, timed_out = _timed_call(lambda: _do_score(router), agent_call_timeout_sec, cancelled)
            if not timed_out and score is not None:
                return score
        except Exception as e:
            logger.warning("score primary error provider=%s agent=%s: %s",
                          provider, agent_id, e)

        if cancelled.is_set():
            return _avg_existing_score()

        # 第2次：重试主Router
        logger.info("[重试] %s agent=%s 评分首次失败, 重试中...", provider, agent_id)
        try:
            score, timed_out = _timed_call(lambda: _do_score(router), agent_call_timeout_sec, cancelled)
            if not timed_out and score is not None:
                return score
        except Exception as e:
            logger.warning("score retry error provider=%s agent=%s: %s",
                          provider, agent_id, e)

        if cancelled.is_set():
            return _avg_existing_score()

        # 第3~6次：候选Provider（优先用研判阶段已成功的候选保持一致性）
        preferred_fb = progress.agent_fallbacks.get(agent_id)
        fb_order = [fp for fp in _fb_providers if fp != provider]
        random.shuffle(fb_order)
        # 优先用研判阶段已成功的候选
        if preferred_fb and preferred_fb in fb_order:
            fb_order.remove(preferred_fb)
            fb_order.insert(0, preferred_fb)

        tried = 0
        logger.info("[切换] %s agent=%s 评分重试仍失败, 尝试候选Provider (池: %s)",
                   provider, agent_id, fb_order[:_MAX_CANDIDATE_ATTEMPTS])
        for fp in fb_order:
            if tried >= _MAX_CANDIDATE_ATTEMPTS or cancelled.is_set():
                break
            if fp not in _fb_router_cache:
                _fb_router_cache[fp] = _create_fallback_router(fp, router)
            fb_rtr = _fb_router_cache[fp]
            if fb_rtr is None:
                continue
            tried += 1
            try:
                score, timed_out2 = _timed_call(lambda r=fb_rtr: _do_score(r), agent_call_timeout_sec, cancelled)
                if not timed_out2 and score is not None:
                    if agent_id not in progress.agent_fallbacks:
                        progress.agent_fallbacks[agent_id] = fp
                    logger.info("[切换] %s agent=%s 评分已切换到 %s", provider, agent_id, fp)
                    print(f"[切换] {provider}→{fp} agent={agent_id} (评分)", flush=True)
                    return score
            except Exception:
                logger.warning("[切换] %s agent=%s 评分候选%s也失败", provider, agent_id, fp)
                continue

        # 全部失败（默认+重试+候选共5次），采用已成功agent平均分
        logger.warning("[放弃] %s agent=%s 评分全部失败(尝试%d次), 使用平均分", provider, agent_id, 2 + tried)
        print(f"[放弃] {provider} agent={agent_id} 评分{2 + tried}次全部失败, 使用已成功agent平均分", flush=True)
        return _avg_existing_score()

    try:
        # ── Step 1: 权重分配 ──
        progress.current_stage = "权重"
        progress.current_agent = ""
        if cancelled.is_set():
            raise _ProviderCancelled(provider)
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
        progress.agent_weights = dict(result.weights)

        # ── Step 2: 研判增强 / 视觉分析（含超时切换） ──
        progress.current_stage = "研判"
        has_vision = router.supports_vision()
        for agent_id, base in base_results.items():
            if cancelled.is_set():
                raise _ProviderCancelled(provider)
            progress.current_agent = agent_id
            try:
                is_vision_agent = (
                    base.agent_type == "kline_vision"
                    and base.image_b64
                    and base.vision_prompt
                )
                # 选择视觉路由
                vision_router = None
                if is_vision_agent:
                    if has_vision:
                        vision_router = router
                    elif vision_fallback_router:
                        vision_router = vision_fallback_router

                text = _enrich_with_fallback(agent_id, base, is_vision_agent, vision_router)
                if text:
                    result.enrichments[agent_id] = text
            except _ProviderCancelled:
                raise
            except Exception as e:
                logger.warning(
                    "enrich failed provider=%s agent=%s: %s", provider, agent_id, e,
                )
            progress.enrich_done += 1

        # ── Step 3: 独立评分（含超时切换） ──
        progress.current_stage = "评分"
        for agent_id, base in base_results.items():
            if cancelled.is_set():
                raise _ProviderCancelled(provider)
            progress.current_agent = agent_id
            try:
                reason_for_score = result.enrichments.get(agent_id, base.reason)
                score = _score_with_fallback(agent_id, base, reason_for_score)
                result.scores[agent_id] = score
                progress.agent_scores[agent_id] = score
            except _ProviderCancelled:
                raise
            except Exception as e:
                logger.warning(
                    "score failed provider=%s agent=%s: %s", provider, agent_id, e,
                )
                result.scores[agent_id] = 50.0
                progress.agent_scores[agent_id] = 50.0
            progress.score_done += 1

    except _ProviderCancelled:
        elapsed = time.time() - t0
        msg = f"超时取消 ({elapsed:.0f}s)"
        logger.warning("provider worker %s cancelled after %.0fs", provider, elapsed)
        result.error = msg
        progress.error = msg
    except Exception as e:
        logger.error("provider worker %s failed: %s", provider, e)
        result.error = str(e)
        progress.error = str(e)

    # 记录备选切换统计
    fb_count = len(progress.agent_fallbacks)
    if fb_count > 0:
        fb_summary = {}
        for aid, fp in progress.agent_fallbacks.items():
            fb_summary[fp] = fb_summary.get(fp, 0) + 1
        logger.info("[切换统计] %s: %d个agent切换, 分布: %s", provider, fb_count, fb_summary)

    result.elapsed_sec = time.time() - t0
    progress.finished = True
    return result


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
    provider_timeout_sec: float | None = None,
    agent_names: dict[str, str] | None = None,
    agent_call_timeout_sec: float | None = None,
    fallback_providers: list[str] | None = None,
    default_providers: list[str] | None = None,
    candidate_providers: list[str] | None = None,
) -> dict[str, ProviderResult]:
    """主调度：为每个 Provider 启动线程并行执行，收集结果。支持增量持久化、断点续传与超时取消。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
    from .progress_display import ProviderProgressDisplay, run_display_loop

    max_wait = provider_timeout_sec or _DEFAULT_PROVIDER_TIMEOUT_SEC
    _agent_timeout = agent_call_timeout_sec or _AGENT_CALL_TIMEOUT_SEC
    # 候选Provider列表：优先使用传入的candidate_providers，否则用fallback_providers或默认值
    _fb_providers = candidate_providers or fallback_providers or _FALLBACK_PROVIDERS
    _default_providers = default_providers or []

    # 仅默认Provider作为并行线程，不添加额外provider
    routers = dict(llm_routers)
    all_providers = list(routers.keys())

    # 构建共享视觉回退 router：供无视觉能力的 provider 线程借用
    _vision_fallback_router: LLMRouter | None = None
    has_vision_need = any(
        b.agent_type == "kline_vision" and b.image_b64
        for b in base_results.values()
    )
    if has_vision_need:
        fallback_p = os.getenv("VISION_FALLBACK_PROVIDER", _DEFAULT_VISION_FALLBACK)
        # 优先从已有 router 中找到一个支持视觉的
        for p, r in routers.items():
            if r.supports_vision():
                _vision_fallback_router = r
                break
        # 若已有 router 中无视觉能力，创建回退 router
        if not _vision_fallback_router:
            fallback_key = os.getenv(f"{fallback_p.upper()}_API_KEY", "").strip()
            if fallback_key and _supports_vision(fallback_p):
                ref = next(iter(routers.values()))
                _vision_fallback_router = LLMRouter(
                    provider=fallback_p,
                    temperature=ref.temperature,
                    max_tokens=ref.max_tokens,
                    request_timeout_sec=ref.request_timeout_sec,
                )
        if _vision_fallback_router:
            fb_name = getattr(_vision_fallback_router, "provider", "?")
            print(f"[并行] 视觉回退: 无视觉能力的Provider将借用 {fb_name} 视觉模型", flush=True)

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

    print(f"[并行] Provider超时上限: {max_wait:.0f}s | 单Agent超时: {_agent_timeout:.0f}s | 重试: 1次+{_MAX_CANDIDATE_ATTEMPTS}候选", flush=True)
    print(f"[并行] 默认Provider: {pending_providers}", flush=True)
    if _fb_providers:
        print(f"[并行] 候选备选池: {_fb_providers} (仅在agent失败时按需激活)", flush=True)

    stop_event = threading.Event()

    progresses = {
        p: ProviderProgress(provider=p, start_time=time.time())
        for p in pending_providers
    }

    # 新进度显示器（表格化：Provider列 × Agent行，仅默认Provider）
    agent_order = list(base_results.keys())
    display = ProviderProgressDisplay(
        progresses,
        agent_names=agent_names,
        total_agents=len(base_results),
        agent_order=agent_order,
        default_providers=_default_providers,
    )

    # 启动进度显示守护线程（1秒间隔）
    display_thread = threading.Thread(
        target=run_display_loop,
        args=(display, stop_event, 1.0),
        daemon=True,
    )
    display_thread.start()

    new_results: dict[str, ProviderResult] = {}

    executor = ThreadPoolExecutor(max_workers=len(pending_providers))
    futures = {
        executor.submit(
            _provider_worker,
            p, routers[p], base_results, analyst_cfg,
            symbol, name, task_summary,
            progresses[p],
            # 仅对无视觉能力的 provider 传入回退 router
            vision_fallback_router=(
                _vision_fallback_router
                if _vision_fallback_router and not routers[p].supports_vision()
                else None
            ),
            agent_call_timeout_sec=_agent_timeout,
            fallback_providers=_fb_providers,
        ): p
        for p in pending_providers
    }

    try:
        for future in as_completed(futures, timeout=max_wait):
            p = futures[future]
            try:
                new_results[p] = future.result()
            except Exception as e:
                logger.error("provider %s future exception: %s", p, e)
                new_results[p] = ProviderResult(provider=p, error=str(e))
            # ── 增量持久化：每个Provider完成后立即存盘 ──
            if run_dir:
                _save_provider_result(run_dir, p, new_results[p])
    except FuturesTimeoutError:
        # 超时：收集未完成的Provider，发送取消信号 + 强制关闭HTTP连接
        timed_out = [p for f, p in futures.items() if p not in new_results]
        print(f"\n[超时] {max_wait:.0f}s 已到，以下Provider未完成将被取消: {timed_out}", flush=True)
        for p in timed_out:
            prog = progresses[p]
            prog.cancel_event.set()  # 通知 worker 线程尽快退出
            # 强制关闭 router 的 HTTP Session，中断正在进行的请求
            try:
                routers[p].close_session()
            except Exception:
                pass
            elapsed = time.time() - prog.start_time
            msg = (
                f"超时取消 ({elapsed:.0f}s), "
                f"研判 {prog.enrich_done}/{prog.enrich_total}, "
                f"评分 {prog.score_done}/{prog.score_total}"
            )
            prog.error = msg
            logger.warning("provider %s timed out: %s", p, msg)

        # 等待被取消的线程退出（给一个宽限期让当前HTTP请求结束）
        grace_deadline = time.time() + 120
        for future, p in futures.items():
            if p not in new_results:
                remaining = max(0.1, grace_deadline - time.time())
                try:
                    new_results[p] = future.result(timeout=remaining)
                except Exception:
                    new_results[p] = ProviderResult(
                        provider=p,
                        error=progresses[p].error or f"超时取消 ({max_wait:.0f}s)",
                        elapsed_sec=time.time() - progresses[p].start_time,
                    )
                if run_dir:
                    _save_provider_result(run_dir, p, new_results[p])

    executor.shutdown(wait=False)

    stop_event.set()
    display_thread.join(timeout=3)

    # 打印最终状态
    display.render_final()

    # 合并缓存结果 + 新结果
    results = {**cached_results, **new_results}

    # 汇总统计
    ok_count = sum(1 for r in results.values() if not r.error)
    fail_count = sum(1 for r in results.values() if r.error)
    total_elapsed = max((r.elapsed_sec for r in results.values()), default=0)
    cached_note = f" 缓存:{len(cached_results)}" if cached_results else ""
    timeout_list = [p for p in new_results if new_results[p].error and "超时" in (new_results[p].error or "")]
    timeout_note = f" 超时:{timeout_list}" if timeout_list else ""
    print(
        f"\n[并行] 完成: {ok_count}成功 {fail_count}失败{cached_note}{timeout_note} | 总耗时 {total_elapsed:.1f}s",
        flush=True,
    )

    return results
