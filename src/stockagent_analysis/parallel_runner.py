# -*- coding: utf-8 -*-
"""并行化执行引擎：每个 LLM Provider 作为独立线程，内部 Agent 并发执行。

优化:
- 研判+评分合并为1次LLM调用 (enrich_and_score)
- Provider内Agent并发 (ThreadPoolExecutor, max_agent_concurrency)
- 权重直接从配置读取，不再调LLM分配
- 视觉Agent保持 enrich → score 两步调用
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ── 从 core 导入通用基础设施 ──
from core.runner import (
    _timed_call, _create_fallback_router, _ProviderCancelled,
    ProviderProgress, ProviderResult,
    _save_provider_result, _load_existing_results,
    _DEFAULT_PROVIDER_TIMEOUT_SEC, _AGENT_CALL_TIMEOUT_SEC,
    _FALLBACK_PROVIDERS, _MAX_CANDIDATE_ATTEMPTS,
)
from core.router import LLMRouter, _supports_vision, _DEFAULT_VISION_FALLBACK

# ── 领域相关导入 ──
from .agents import AgentBaseResult
from .llm_client import (
    assign_agent_weights, score_agent_analysis, enrich_and_score, config_weights,
)

logger = logging.getLogger(__name__)

# 默认 Provider 内 Agent 并发数
_DEFAULT_MAX_AGENT_CONCURRENCY = 6


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
    max_agent_concurrency: int = _DEFAULT_MAX_AGENT_CONCURRENCY,
) -> ProviderResult:
    """单个 Provider 线程：配置权重 → Agent并发(研判+评分合并) → 完成。"""
    t0 = time.time()
    result = ProviderResult(provider=provider)
    agent_count = len(base_results)
    progress.enrich_total = agent_count
    progress.score_total = agent_count

    cancelled = progress.cancel_event
    _fb_providers = fallback_providers or _FALLBACK_PROVIDERS
    # 备选Router缓存（线程安全）
    _fb_router_cache: dict[str, LLMRouter | None] = {}
    _fb_cache_lock = threading.Lock()

    # Provider 级别失败追踪：连续失败过多时跳过重试/跳过主 Provider
    _primary_fail_count = [0]
    _primary_success_count = [0]
    _primary_track_lock = threading.Lock()

    def _get_fb_router(fp: str) -> LLMRouter | None:
        with _fb_cache_lock:
            if fp not in _fb_router_cache:
                _fb_router_cache[fp] = _create_fallback_router(fp, router)
            return _fb_router_cache[fp]

    def _avg_existing_score(agent_id: str) -> float:
        """全部失败时，采用已成功agent的平均分数。"""
        existing = list(progress.agent_scores.values())
        if existing:
            avg = sum(existing) / len(existing)
            logger.info("[平均分] %s agent=%s 全部失败, 使用已成功%d个agent平均分 %.1f",
                       provider, agent_id, len(existing), avg)
            return avg
        return 50.0

    def _enrich_and_score_with_fallback(agent_id: str, base: AgentBaseResult) -> tuple[str | None, float]:
        """合并研判+评分：主Router→重试一次→随机候选Provider(最多4个)。
        自适应：主Provider连续失败>=3次跳过重试，>=6次跳过主Provider直接走候选。"""
        def _do_call(rtr: LLMRouter) -> tuple[str | None, float | None]:
            return enrich_and_score(
                rtr, base.role, agent_id, symbol, name,
                base.reason, base.data_context,
            )

        # 读取失败追踪状态
        with _primary_track_lock:
            skip_primary = _primary_fail_count[0] >= 6 and _primary_success_count[0] == 0
            skip_retry = _primary_fail_count[0] >= 3 and _primary_success_count[0] == 0

        primary_succeeded = False

        if not skip_primary:
            # 第1次：主Router
            try:
                pair, timed_out = _timed_call(lambda: _do_call(router), agent_call_timeout_sec, cancelled)
                if not timed_out and pair:
                    text, score = pair
                    if text and score is not None:
                        primary_succeeded = True
                        with _primary_track_lock:
                            _primary_success_count[0] += 1
                        return text, score
            except Exception as e:
                logger.warning("enrich_and_score primary error provider=%s agent=%s: %s",
                              provider, agent_id, e)

            if not primary_succeeded:
                with _primary_track_lock:
                    _primary_fail_count[0] += 1

            if cancelled.is_set():
                return None, _avg_existing_score(agent_id)

            # 第2次：重试主Router（skip_retry 时跳过）
            if not skip_retry:
                logger.info("[重试] %s agent=%s 分析首次失败, 重试中...", provider, agent_id)
                try:
                    pair, timed_out = _timed_call(lambda: _do_call(router), agent_call_timeout_sec, cancelled)
                    if not timed_out and pair:
                        text, score = pair
                        if text and score is not None:
                            with _primary_track_lock:
                                _primary_success_count[0] += 1
                            return text, score
                except Exception as e:
                    logger.warning("enrich_and_score retry error provider=%s agent=%s: %s",
                                  provider, agent_id, e)

                if cancelled.is_set():
                    return None, _avg_existing_score(agent_id)
        else:
            logger.info("[跳过主Provider] %s agent=%s 已累计%d次失败, 直接使用候选",
                       provider, agent_id, _primary_fail_count[0])
            print(f"[跳过] {provider} agent={agent_id} 累计{_primary_fail_count[0]}次失败→候选", flush=True)

        # 候选Provider（最多4个）
        fb_list = [fp for fp in _fb_providers if fp != provider]
        random.shuffle(fb_list)
        tried = 0
        if not skip_primary:
            logger.info("[切换] %s agent=%s 尝试候选Provider (池: %s)",
                       provider, agent_id, fb_list[:_MAX_CANDIDATE_ATTEMPTS])
        for fp in fb_list:
            if tried >= _MAX_CANDIDATE_ATTEMPTS or cancelled.is_set():
                break
            fb_rtr = _get_fb_router(fp)
            if fb_rtr is None:
                continue
            tried += 1
            try:
                pair, timed_out2 = _timed_call(lambda r=fb_rtr: _do_call(r), agent_call_timeout_sec, cancelled)
                if not timed_out2 and pair:
                    text, score = pair
                    if text and score is not None:
                        progress.agent_fallbacks[agent_id] = fp
                        logger.info("[切换] %s agent=%s 分析已切换到 %s", provider, agent_id, fp)
                        print(f"[切换] {provider}→{fp} agent={agent_id}", flush=True)
                        return text, score
            except Exception:
                logger.warning("[切换] %s agent=%s 候选%s也失败", provider, agent_id, fp)
                continue

        # 全部失败
        attempts = (0 if skip_primary else (1 if skip_retry else 2)) + tried
        logger.warning("[放弃] %s agent=%s 分析全部失败(尝试%d次)", provider, agent_id, attempts)
        print(f"[放弃] {provider} agent={agent_id} {attempts}次全部失败, 使用平均分", flush=True)
        return None, _avg_existing_score(agent_id)

    def _vision_enrich_with_fallback(agent_id: str, base: AgentBaseResult,
                                      vision_rtr: LLMRouter | None) -> str | None:
        """视觉Agent研判增强（保持原两步）：主Router→重试→候选。
        复用 _primary_fail_count 自适应跳过。"""
        def _do_call(rtr: LLMRouter) -> str | None:
            if vision_rtr:
                actual_rtr = vision_rtr
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

        with _primary_track_lock:
            skip_primary = _primary_fail_count[0] >= 6 and _primary_success_count[0] == 0
            skip_retry = _primary_fail_count[0] >= 3 and _primary_success_count[0] == 0

        if not skip_primary:
            # 第1次：主Router
            try:
                text, timed_out = _timed_call(lambda: _do_call(router), agent_call_timeout_sec, cancelled)
                if not timed_out and text:
                    with _primary_track_lock:
                        _primary_success_count[0] += 1
                    return text
            except Exception as e:
                logger.warning("vision enrich primary error provider=%s agent=%s: %s",
                              provider, agent_id, e)

            with _primary_track_lock:
                _primary_fail_count[0] += 1

            if cancelled.is_set():
                return None

            # 第2次：重试（skip_retry 时跳过）
            if not skip_retry:
                try:
                    text, timed_out = _timed_call(lambda: _do_call(router), agent_call_timeout_sec, cancelled)
                    if not timed_out and text:
                        with _primary_track_lock:
                            _primary_success_count[0] += 1
                        return text
                except Exception as e:
                    logger.warning("vision enrich retry error provider=%s agent=%s: %s",
                                  provider, agent_id, e)

                if cancelled.is_set():
                    return None

        # 候选Provider
        fb_list = [fp for fp in _fb_providers if fp != provider]
        random.shuffle(fb_list)
        tried = 0
        for fp in fb_list:
            if tried >= _MAX_CANDIDATE_ATTEMPTS or cancelled.is_set():
                break
            fb_rtr = _get_fb_router(fp)
            if fb_rtr is None:
                continue
            tried += 1
            try:
                text, timed_out2 = _timed_call(lambda r=fb_rtr: _do_call(r), agent_call_timeout_sec, cancelled)
                if not timed_out2 and text:
                    progress.agent_fallbacks[agent_id] = fp
                    return text
            except Exception:
                continue

        return None

    def _vision_score_with_fallback(agent_id: str, base: AgentBaseResult,
                                     reason_text: str) -> float:
        """视觉Agent评分（保持原两步）：主Router→重试→候选→平均分。
        复用 _primary_fail_count 自适应跳过。"""
        def _do_score(rtr: LLMRouter) -> float:
            return score_agent_analysis(
                rtr, base.role, agent_id, symbol, name,
                reason_text, base.data_context,
            )

        with _primary_track_lock:
            skip_primary = _primary_fail_count[0] >= 6 and _primary_success_count[0] == 0
            skip_retry = _primary_fail_count[0] >= 3 and _primary_success_count[0] == 0

        if not skip_primary:
            # 第1次
            try:
                score, timed_out = _timed_call(lambda: _do_score(router), agent_call_timeout_sec, cancelled)
                if not timed_out and score is not None:
                    with _primary_track_lock:
                        _primary_success_count[0] += 1
                    return score
            except Exception:
                pass

            with _primary_track_lock:
                _primary_fail_count[0] += 1

            if cancelled.is_set():
                return _avg_existing_score(agent_id)

            # 第2次（skip_retry 时跳过）
            if not skip_retry:
                try:
                    score, timed_out = _timed_call(lambda: _do_score(router), agent_call_timeout_sec, cancelled)
                    if not timed_out and score is not None:
                        with _primary_track_lock:
                            _primary_success_count[0] += 1
                        return score
                except Exception:
                    pass

                if cancelled.is_set():
                    return _avg_existing_score(agent_id)

        # 候选
        preferred_fb = progress.agent_fallbacks.get(agent_id)
        fb_order = [fp for fp in _fb_providers if fp != provider]
        random.shuffle(fb_order)
        if preferred_fb and preferred_fb in fb_order:
            fb_order.remove(preferred_fb)
            fb_order.insert(0, preferred_fb)

        tried = 0
        for fp in fb_order:
            if tried >= _MAX_CANDIDATE_ATTEMPTS or cancelled.is_set():
                break
            fb_rtr = _get_fb_router(fp)
            if fb_rtr is None:
                continue
            tried += 1
            try:
                score, timed_out2 = _timed_call(lambda r=fb_rtr: _do_score(r), agent_call_timeout_sec, cancelled)
                if not timed_out2 and score is not None:
                    if agent_id not in progress.agent_fallbacks:
                        progress.agent_fallbacks[agent_id] = fp
                    return score
            except Exception:
                continue

        return _avg_existing_score(agent_id)

    def _process_single_agent(agent_id: str, base: AgentBaseResult) -> tuple[str, str | None, float]:
        """处理单个Agent：视觉保持两步，普通合并一步。返回 (agent_id, text, score)。"""
        is_vision_agent = (
            base.agent_type == "kline_vision"
            and base.image_b64
            and base.vision_prompt
        )

        # 标记进行中
        with progress._agents_lock:
            progress.agents_in_progress.add(agent_id)

        try:
            if is_vision_agent:
                # 视觉Agent：两步走
                has_vision = router.supports_vision()
                vision_router = None
                if has_vision:
                    vision_router = router
                elif vision_fallback_router:
                    vision_router = vision_fallback_router

                text = _vision_enrich_with_fallback(agent_id, base, vision_router)
                if text:
                    result.enrichments[agent_id] = text
                reason_for_score = text or base.reason
                score = _vision_score_with_fallback(agent_id, base, reason_for_score)
            else:
                # 普通Agent：合并一步
                text, score = _enrich_and_score_with_fallback(agent_id, base)
                if text:
                    result.enrichments[agent_id] = text
        except Exception as e:
            logger.warning("agent processing failed provider=%s agent=%s: %s", provider, agent_id, e)
            text = None
            score = _avg_existing_score(agent_id)
        finally:
            with progress._agents_lock:
                progress.agents_in_progress.discard(agent_id)

        return agent_id, text, score

    try:
        # ── Step 1: 配置权重（无LLM调用） ──
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
        result.weights = config_weights(agent_list)
        progress.weight_done = True
        progress.agent_weights = dict(result.weights)

        # ── Step 2: Agent并发分析（研判+评分合并） ──
        progress.current_stage = "分析"
        max_concurrent = min(max_agent_concurrency, agent_count)

        with ThreadPoolExecutor(max_workers=max_concurrent) as agent_exec:
            futures = {
                agent_exec.submit(_process_single_agent, aid, b): aid
                for aid, b in base_results.items()
            }
            for future in as_completed(futures):
                if cancelled.is_set():
                    # 取消剩余任务
                    for f in futures:
                        f.cancel()
                    raise _ProviderCancelled(provider)
                try:
                    aid, text, score = future.result()
                    result.scores[aid] = score
                    progress.agent_scores[aid] = score
                    progress.enrich_done += 1
                    progress.score_done += 1
                except _ProviderCancelled:
                    raise
                except Exception as e:
                    aid = futures[future]
                    logger.warning("agent future failed provider=%s agent=%s: %s", provider, aid, e)
                    result.scores[aid] = 50.0
                    progress.agent_scores[aid] = 50.0
                    progress.enrich_done += 1
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
    max_agent_concurrency: int = _DEFAULT_MAX_AGENT_CONCURRENCY,
) -> dict[str, ProviderResult]:
    """主调度：为每个 Provider 启动线程并行执行，收集结果。支持增量持久化、断点续传与超时取消。"""
    from concurrent.futures import TimeoutError as FuturesTimeoutError
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

    print(f"[并行] Provider超时: {max_wait:.0f}s | Agent超时: {_agent_timeout:.0f}s | Agent并发: {max_agent_concurrency} | 重试: 1次+{_MAX_CANDIDATE_ATTEMPTS}候选", flush=True)
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
            max_agent_concurrency=max_agent_concurrency,
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
                f"分析 {prog.enrich_done}/{prog.enrich_total}"
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
