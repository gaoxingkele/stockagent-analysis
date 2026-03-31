# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .agent_data_mapping import build_agent_data_table, format_agent_data_table
from .agents import AnalystAgent, AgentResult, AgentBaseResult, write_message, create_agent, _parse_vision_response
from .config_loader import load_agent_configs, load_project_config, split_agents
from .data_backend import DataBackend
from .io_utils import dump_json, get_agent_logger
from core.router import LLMRouter
from core.progress import PipelineTracker, AgentNameRegistry
from .llm_client import assign_agent_weights, score_agent_analysis, generate_scenario_and_position
from .report_pdf import build_investor_pdf


def _print_data_progress(step: str, detail: str) -> None:
    print(f"[数据] {step}: {detail}", flush=True)


def _build_agent_task_summary(analyst_cfg: list[dict[str, Any]]) -> str:
    """构建多智能体任务分工综述，供总协调在调用各模型分配权重时传入。"""
    lines = []
    for c in analyst_cfg:
        agent_id = c.get("agent_id", "")
        role = c.get("role", "")
        name = c.get("name", role)
        points = c.get("data_sources", {}).get("required_data_points", [])
        task_desc = "、".join(points) if isinstance(points, list) else str(points)
        lines.append(f"- {agent_id}（{name} / {role}）：负责 {task_desc or '综合研判'}")
    return "【多智能体任务分工综述】\n" + "\n".join(lines)


# ── Provider 别名自动识别 ──────────────────────────────

_KNOWN_PROVIDERS = [
    "doubao", "minmax", "claude", "openai", "grok", "kimi",
    "deepseek", "glm", "qwen", "gemini", "perplexity",
]


def _collect_known_models(provider: str) -> list[str]:
    """收集某 provider 的所有已知模型名（env vars + Cloubic chain）。"""
    from core.router import _get_cloubic_model_chain, _get_direct_model_chain
    models: list[str] = []

    # 直连模型链
    p_upper = provider.upper()
    key_map = {"claude": "ANTHROPIC"}
    env_prefix = key_map.get(provider.lower(), p_upper)
    primary = os.getenv(f"{env_prefix}_MODEL", "").strip()
    if primary:
        models.extend(_get_direct_model_chain(provider, primary))

    # Cloubic 模型链
    models.extend(_get_cloubic_model_chain(provider))

    # 去重保序
    seen: set[str] = set()
    return [m for m in models if m and not (m.lower() in seen or seen.add(m.lower()))]  # type: ignore[func-returns-value]


def _normalize(s: str) -> str:
    """去除 -_. 并小写，用于模糊匹配。"""
    import re
    return re.sub(r"[-_.\s]", "", s).lower()


def _find_model_by_hint(provider: str, hint: str) -> str | None:
    """在 provider 的已知模型列表中，用 hint 模糊匹配。"""
    models = _collect_known_models(provider)
    hint_n = _normalize(hint)
    if not hint_n:
        return None
    # 精确子串匹配（归一化后）
    for m in models:
        if hint_n in _normalize(m):
            return m
    return None


def _resolve_provider_arg(arg: str) -> tuple[str, str | None, str | None, bool]:
    """解析 --providers 中的单个参数。

    Returns: (provider, model_override_or_None, display_name_or_None, no_fallback)

    - "doubao"          → ("doubao", None, None, False)         # 默认模式
    - "doubao-seed1.6"  → ("doubao", "doubao-seed-1-6-251015", "doubao-seed1.6", True)
    """
    arg_lower = arg.strip().lower()
    # 精确匹配已知 provider → 默认模式
    if arg_lower in _KNOWN_PROVIDERS:
        return (arg_lower, None, None, False)

    # 前缀匹配：按长度倒序避免 "glm" 截断 "glm-xxx"
    for p in sorted(_KNOWN_PROVIDERS, key=len, reverse=True):
        if arg_lower.startswith(p + "-"):
            hint = arg[len(p) + 1:]  # 保留原始大小写
            model = _find_model_by_hint(p, hint)
            if model:
                return (p, model, arg.strip(), True)
            # hint 无法匹配到已知模型 → 当作精确模型名使用
            return (p, hint, arg.strip(), True)

    # 无法解析 → 当普通 provider 传入
    return (arg, None, None, False)


def _verify_data_readiness(
    analysis_context: dict[str, Any],
    analyst_cfg: list[dict[str, Any]],
    run_dir: Path,
    manager_logger: logging.Logger,
) -> bool:
    """进入大模型调用前的数据就绪门控：校验所有必需数据文件已落地。"""
    missing: list[str] = []

    # 1. is_complete（month 缺失属于可降级，不阻断）
    integrity = analysis_context.get("data_integrity", {})
    failed_items_raw = integrity.get("failed_items", [])
    only_month_missing = set(failed_items_raw) <= {"month"}
    if not integrity.get("is_complete", False) and not only_month_missing:
        missing.append("data_integrity.is_complete=False")

    # 2. analysis_context.json 文件存在
    ctx_path = run_dir / "data" / "analysis_context.json"
    if not ctx_path.exists():
        missing.append(f"analysis_context.json ({ctx_path})")

    # 3. K线CSV文件（day/week 必须，month 可选降级）
    data_dir = run_dir / "data"
    failed_tfs = set(integrity.get("failed_items", []))
    for tf in ("day", "week"):
        csv_path = data_dir / "kline" / f"{tf}.csv"
        if not csv_path.exists():
            missing.append(f"kline/{tf}.csv")
    month_csv = data_dir / "kline" / "month.csv"
    if not month_csv.exists():
        manager_logger.info("month kline CSV missing — degraded mode (non-fatal)")

    # 4. kline_vision agent 依赖的 chart PNG 文件（仅校验有数据的周期）
    chart_files = analysis_context.get("chart_files", {})
    for cfg in analyst_cfg:
        if cfg.get("agent_type") != "kline_vision":
            continue
        tf = cfg.get("timeframe", "")
        if tf in failed_tfs:
            manager_logger.info("kline_vision chart %s skipped — data unavailable (non-fatal)", tf)
            continue
        chart_path_str = chart_files.get(tf, "")
        if chart_path_str:
            if not Path(chart_path_str).exists():
                missing.append(f"K线图表 {tf}: {chart_path_str}")
        else:
            # chart_files 中没有此 timeframe — 数据行数太少无法生成图表，降级跳过
            manager_logger.info("kline_vision chart %s not generated (data rows too few for chart) — degraded (non-fatal)", tf)
            continue

    if missing:
        manager_logger.warning("data readiness check FAILED: missing=%s", missing)
        print("[数据门控] 数据就绪校验失败，缺失项：", flush=True)
        for item in missing:
            print(f"  - {item}", flush=True)
        return False

    total_items = 3 + len([c for c in analyst_cfg if c.get("agent_type") == "kline_vision"]) + 1
    print(f"[数据就绪] 全部 {total_items} 项数据文件已确认落地，允许进入大模型调用阶段", flush=True)
    manager_logger.info("data readiness check PASSED: %d items verified", total_items)
    return True


def _merge_provider_results(
    analysts: list[AnalystAgent],
    base_results: dict[str, AgentBaseResult],
    provider_results: dict,
    analysis_context: dict[str, Any],
) -> tuple[list[AgentResult], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """合并所有 Provider 的并行执行结果为统一的 submissions / model_weights / model_scores。"""
    from core.runner import ProviderResult

    model_weights: dict[str, dict[str, float]] = {}
    model_scores: dict[str, dict[str, float]] = {}

    for p, pr in provider_results.items():
        if pr.error:
            continue
        model_weights[p] = pr.weights
        model_scores[p] = pr.scores

    submissions: list[AgentResult] = []
    for a in analysts:
        base = base_results[a.agent_id]
        llm_evals: dict[str, str] = {}

        # 收集各 Provider 对此 Agent 的增强文本
        for p, pr in provider_results.items():
            if pr.error:
                continue
            text = pr.enrichments.get(a.agent_id)
            if text:
                llm_evals[p] = text

        # K线视觉智能体：收集视觉结果并聚合
        vision_texts: dict[str, str] = {}
        for p, pr in provider_results.items():
            if pr.error:
                continue
            vt = pr.vision_results.get(a.agent_id)
            if vt:
                vision_texts[p] = vt

        if base.agent_type == "kline_vision" and vision_texts:
            all_votes, all_scores = [], []
            parts = []
            for p, vt in vision_texts.items():
                v, s, _ = _parse_vision_response(vt)
                all_votes.append(v)
                all_scores.append(s)
                short = vt.replace("\n", " ").strip()[:120]
                parts.append(f"{p}: {short}")

            avg_score = max(0.0, min(100.0, sum(all_scores) / len(all_scores)))
            if avg_score >= 70:
                final_vote = "buy"
            elif avg_score < 50:
                final_vote = "sell"
            else:
                final_vote = "hold"

            from .chart_generator import TIMEFRAME_LABEL
            tf_label = TIMEFRAME_LABEL.get(base.timeframe or "", base.timeframe or "")
            reason = f"【{tf_label}K线视觉研判】" + "；".join(parts)

            # 视觉模式使用视觉专用置信度
            data_quality = float(analysis_context.get("features", {}).get("data_quality_score", 0.7))
            confidence = max(0.1, min(0.98, 0.5 + 0.4 * data_quality))

            result = AgentResult(
                agent_id=a.agent_id,
                dim_code=base.dim_code,
                vote=final_vote,
                score_0_100=avg_score,
                confidence_0_1=confidence,
                reason=reason,
                llm_multi_evaluations=llm_evals,
            )
        else:
            # 普通智能体：合并多模型评价
            if llm_evals:
                reason = AnalystAgent._merge_multi_eval(llm_evals, fallback=base.reason)
            else:
                reason = base.reason

            result = AgentResult(
                agent_id=a.agent_id,
                dim_code=base.dim_code,
                vote=base.vote,
                score_0_100=base.score_0_100,
                confidence_0_1=base.confidence_0_1,
                reason=reason,
                llm_multi_evaluations=llm_evals,
            )

        submissions.append(result)
        # 写入提交文件
        a._submit(result, base.snap, analysis_context)

    return submissions, model_weights, model_scores


def run_analysis(
    root: Path,
    symbol: str,
    name: str,
    run_dir: Path,
    llm_provider_override: str | None = None,
    multi_eval_providers_override: str | None = None,
) -> dict[str, Any]:
    project_cfg = load_project_config(root)
    agents_cfg = load_agent_configs(root)
    manager_cfg, analyst_cfg = split_agents(agents_cfg)
    backend_cfg = project_cfg["data_backend"]
    backend = DataBackend(mode=backend_cfg["mode"], default_sources=backend_cfg["default_sources"])
    llm_cfg = project_cfg.get("llm", {})
    llm_enabled = bool(llm_cfg.get("enabled", True))
    multi_turn = bool(llm_cfg.get("multi_turn_default", True))

    # ── 自动发现所有Provider：默认 + 候选 ──
    default_providers_cfg = llm_cfg.get("default_providers", [])
    candidate_providers_cfg = llm_cfg.get("candidate_providers", [])

    # ── 解析 --providers 参数（支持别名模式） ──
    # alias_map: key(显示名) → (provider, model_override, no_fallback)
    alias_map: dict[str, tuple[str, str | None, bool]] = {}

    def _parse_provider_list(raw_args: list[str]) -> list[str]:
        """解析 provider 列表，支持别名模式，填充 alias_map。"""
        result = []
        for arg in raw_args:
            prov, model_ov, disp_name, no_fb = _resolve_provider_arg(arg)
            key = disp_name or prov
            result.append(key)
            if model_ov:
                alias_map[key] = (prov, model_ov, no_fb)
        return result

    if multi_eval_providers_override:
        raw_args = [p.strip() for p in multi_eval_providers_override.split(",") if p.strip()]
        default_providers = _parse_provider_list(raw_args)
    elif llm_provider_override:
        default_providers = _parse_provider_list([llm_provider_override])
    else:
        raw_cfg = list(default_providers_cfg) if default_providers_cfg else llm_cfg.get("multi_eval_providers", ["kimi"])
        default_providers = _parse_provider_list(raw_cfg)

    provider = default_providers[0] if default_providers else "kimi"
    # 别名 key → 实际 provider 名（用于 API key 检查）
    def _real_provider(key: str) -> str:
        return alias_map[key][0] if key in alias_map else key

    # 过滤：只保留有API_KEY的Provider（Cloubic桥接的provider用CLOUBIC_API_KEY即可）
    from core.router import _should_route_via_cloubic
    cloubic_key = os.getenv("CLOUBIC_API_KEY", "").strip()

    def _has_api_key(p: str) -> bool:
        """检查provider是否有可用的API Key。"""
        real_p = _real_provider(p)
        # Claude 的环境变量是 ANTHROPIC_API_KEY，特殊映射
        key_map = {"claude": "ANTHROPIC_API_KEY"}
        env_key = key_map.get(real_p.lower(), f"{real_p.upper()}_API_KEY")
        if os.getenv(env_key, "").strip():
            return True
        # Cloubic 桥接的 provider，有 CLOUBIC_API_KEY 即可
        if _should_route_via_cloubic(real_p) and cloubic_key:
            return True
        return False

    default_providers = [p for p in default_providers if _has_api_key(p)]
    candidate_providers = [
        p for p in candidate_providers_cfg
        if p not in default_providers and _has_api_key(p)
    ]

    # 全部Provider = 默认 + 候选，都启动子任务
    all_providers = default_providers + candidate_providers

    llm_routers: dict[str, LLMRouter] = {}
    if llm_enabled:
        for key in all_providers:
            if key in alias_map:
                real_p, model_ov, no_fb = alias_map[key]
                llm_routers[key] = LLMRouter(
                    provider=real_p,
                    temperature=float(llm_cfg.get("temperature", 0.3)),
                    max_tokens=int(llm_cfg.get("max_tokens", 600)),
                    request_timeout_sec=float(llm_cfg.get("request_timeout_sec", 45.0)),
                    multi_turn=multi_turn,
                    model_override=model_ov,
                    no_fallback=no_fb,
                )
            else:
                llm_routers[key] = LLMRouter(
                    provider=key,
                    temperature=float(llm_cfg.get("temperature", 0.3)),
                    max_tokens=int(llm_cfg.get("max_tokens", 600)),
                    request_timeout_sec=float(llm_cfg.get("request_timeout_sec", 45.0)),
                    multi_turn=multi_turn,
                )
    manager_logger = get_agent_logger(run_dir, manager_cfg["agent_id"])

    manager_logger.info("start symbol=%s name=%s analysts=%s", symbol, name, [a["agent_id"] for a in analyst_cfg])
    manager_logger.info("llm_enabled=%s default=%s candidate=%s", llm_enabled, default_providers, candidate_providers)

    multi_model_weight_mode = bool(project_cfg.get("multi_model_weight_mode", False))
    weight_providers = list(llm_routers.keys())[:10]
    use_multi_model_weights = multi_model_weight_mode and len(weight_providers) >= 2

    # 初始化流水线追踪与Agent名称注册
    pipeline = PipelineTracker()
    agent_registry = AgentNameRegistry(analyst_cfg)

    print(
        f"[启动] 股票 {symbol} {name} | 智能体数量: {len(analyst_cfg)} | "
        f"默认: {', '.join(default_providers)} | 候选: {', '.join(candidate_providers)}"
        + (f" | 多模型权重模式" if use_multi_model_weights else ""),
        flush=True
    )

    # ── 阶段1: 数据采集 ──
    pipeline.advance("数据采集")
    existing_ctx_path = run_dir / "data" / "analysis_context.json"
    if existing_ctx_path.exists():
        print("[断点续传] 检测到已有数据，跳过数据采集，直接进入分析阶段", flush=True)
        analysis_context = json.loads(existing_ctx_path.read_text(encoding="utf-8"))
        manager_logger.info("resume: loaded existing analysis_context from %s", existing_ctx_path)
    else:
        analysis_context = backend.collect_and_save_context(
            symbol=symbol,
            name=name,
            run_dir=str(run_dir),
            preferred_sources=backend_cfg.get("default_sources", []),
            progress_cb=_print_data_progress,
        )
    integrity = analysis_context.get("data_integrity", {})
    is_complete = bool(integrity.get("is_complete", False))
    failed_items = integrity.get("failed_items", [])
    # day 必须；week/month 缺失可降级继续（新股/北交所数据源不全时）
    only_optional_missing = set(failed_items) <= {"month", "week"}
    if not is_complete and not only_optional_missing:
        err_msg = f"数据获取失败，缺失项: {', '.join(failed_items) if failed_items else '基础数据'}"
        manager_logger.warning("data_integrity_incomplete=%s; terminate_no_llm_no_report", integrity)
        print(f"[错误] {err_msg}", flush=True)
        print("[终止] K线数据重试3次仍失败，已终止大模型调用，不输出研判报告。请检查 TUSHARE_TOKEN、网络连接后重试。", flush=True)
        return {
            "error": True,
            "message": err_msg,
            "failed_items": failed_items,
            "data_integrity": integrity,
            "symbol": symbol,
            "name": name,
        }
    if not is_complete and only_optional_missing:
        print(f"[警告] 缺失 {', '.join(failed_items)} 数据，月线相关Agent将降级为文本分析模式", flush=True)

    # ── v2: 新闻/舆情增强 (改进计划#5) ──
    _news_enhance = bool(project_cfg.get("news_enhance", True))
    if _news_enhance and llm_routers:
        from .news_search import enrich_news_data
        _news_router = next(iter(llm_routers.values()), None)
        if _news_router:
            print("[新闻增强] 启动LLM新闻分析...", flush=True)
            _existing_news = analysis_context.get("news", [])
            try:
                _news_result = enrich_news_data(
                    _news_router, symbol, name, _existing_news,
                    use_perplexity=bool(os.getenv("PERPLEXITY_API_KEY", "").strip()),
                )
                # 写入 analysis_context 供 SENTIMENT_FLOW Agent 消费
                analysis_context["news_analysis"] = _news_result.get("sentiment", {})
                if _news_result.get("perplexity_used"):
                    analysis_context["news"] = _news_result.get("news_items", _existing_news)
                # 更新 features.news_sentiment 为LLM分析的情绪分
                _llm_sent = _news_result.get("sentiment", {}).get("sentiment_score", 0)
                if _llm_sent != 0 and "features" in analysis_context:
                    analysis_context["features"]["news_sentiment"] = float(_llm_sent)
                    analysis_context["features"]["news_sentiment_source"] = "llm"
                dump_json(run_dir / "data" / "news_analysis.json", _news_result)
                print(f"[新闻增强] 完成: 情绪={_llm_sent} 事件={len(_news_result.get('sentiment', {}).get('key_events', []))}个", flush=True)
            except Exception as e:
                manager_logger.warning("news_enhance failed: %s", e)
                print(f"[新闻增强] 失败: {e}, 使用原始新闻数据", flush=True)

    analysts = [create_agent(cfg, run_dir, backend, llm_routers=llm_routers) for cfg in analyst_cfg]

    # 调用大模型前检查各 Agent 所需数据，列出本地/云端对照
    agent_mappings = build_agent_data_table(analyst_cfg, analysis_context)
    print(format_agent_data_table(agent_mappings), flush=True)
    dump_json(run_dir / "data" / "agent_data_mapping.json", agent_mappings)

    debate_rounds = 0 if use_multi_model_weights else int(project_cfg.get("debate_rounds", 2))
    model_weights: dict[str, dict[str, float]] = {}
    model_scores: dict[str, dict[str, float]] = {}
    submissions: list[AgentResult] = []

    if use_multi_model_weights:
        # ══════════════════════════════════════════════════════════════
        # 并行路径：本地策略分析 → 多Provider并行 → 合并结果
        # ══════════════════════════════════════════════════════════════
        from .parallel_runner import run_providers_parallel

        # 数据就绪门控
        if not _verify_data_readiness(analysis_context, analyst_cfg, run_dir, manager_logger):
            err_msg = "数据就绪校验失败，无法进入大模型调用阶段"
            print(f"[错误] {err_msg}", flush=True)
            return {
                "error": True,
                "message": err_msg,
                "symbol": symbol,
                "name": name,
            }

        # ── 阶段2: 本地策略分析 ──
        pipeline.advance("本地分析")
        base_results: dict[str, AgentBaseResult] = {}
        for a in analysts:
            base_results[a.agent_id] = a.analyze_local(symbol, name, analysis_context)
        print(f"[本地分析] 完成 ({len(base_results)}个智能体)", flush=True)

        # ── 阶段3: 并行分析 ──
        pipeline.advance("并行分析")
        task_summary = _build_agent_task_summary(analyst_cfg)
        (run_dir / "data").mkdir(parents=True, exist_ok=True)
        (run_dir / "data" / "task_summary.txt").write_text(task_summary, encoding="utf-8")

        # 仅默认Provider启动并行子任务，候选Provider作为共享备选池（按需激活）
        parallel_routers = {p: llm_routers[p] for p in default_providers if p in llm_routers}
        provider_timeout = float(llm_cfg.get("provider_timeout_sec", 600))
        agent_call_timeout = float(llm_cfg.get("agent_call_timeout_sec", 120))
        max_agent_concurrency = int(llm_cfg.get("max_agent_concurrency", 6))
        provider_results = run_providers_parallel(
            parallel_routers, base_results, analyst_cfg, symbol, name, task_summary,
            run_dir=run_dir,
            provider_timeout_sec=provider_timeout,
            agent_names=agent_registry.as_dict(),
            agent_call_timeout_sec=agent_call_timeout,
            default_providers=default_providers,
            candidate_providers=candidate_providers,
            max_agent_concurrency=max_agent_concurrency,
        )

        # ── 阶段4: 合并评分 ──
        pipeline.advance("合并评分")
        submissions, model_weights, model_scores = _merge_provider_results(
            analysts, base_results, provider_results, analysis_context,
        )
        ok_providers = [p for p, pr in provider_results.items() if not pr.error]
        fail_providers = [p for p, pr in provider_results.items() if pr.error]
        manager_logger.info(
            "parallel merge done: ok=%s fail=%s weights_keys=%s scores_keys=%s",
            ok_providers, fail_providers, list(model_weights.keys()), list(model_scores.keys()),
        )
        print(f"[合并] 完成 (成功:{len(ok_providers)} 失败:{len(fail_providers)})", flush=True)

        # 如果全部Provider失败，使用本地base_results作为后备
        if not submissions:
            print("[警告] 全部Provider失败，使用本地策略分析结果", flush=True)
            for a in analysts:
                base = base_results[a.agent_id]
                submissions.append(AgentResult(
                    agent_id=base.agent_id, dim_code=base.dim_code,
                    vote=base.vote, score_0_100=base.score_0_100,
                    confidence_0_1=base.confidence_0_1,
                    reason=base.reason, llm_multi_evaluations={},
                ))

        done = len(analysts)
        total_steps = len(analysts) + 2  # debate_rounds=0 in parallel mode
    else:
        # ══════════════════════════════════════════════════════════════
        # 串行路径：保持原有逻辑不变
        # ══════════════════════════════════════════════════════════════
        pipeline.advance("本地分析")
        pipeline.advance("并行分析")  # 串行模式也走这个阶段标记
        total_steps = len(analysts) + debate_rounds + 2
        done = 0

        def agent_progress_cb(s: str, d: str) -> None:
            print(f"[智能体] {s}: {d}", flush=True)

        for a in analysts:
            cn = agent_registry.get(a.agent_id)
            print(f"[分析] {done}/{total_steps} 正在分析 {cn}", flush=True)
            submissions.append(
                a.analyze(
                    symbol, name,
                    analysis_context=analysis_context,
                    progress_cb=agent_progress_cb,
                )
            )
            done += 1
            print(f"[分析] {done}/{total_steps} 完成 {cn} vote={submissions[-1].vote} score={submissions[-1].score_0_100:.1f}", flush=True)

    # ── v2 结构化辩论 (改进计划#3) ──
    debate_bull_bear: dict[str, Any] = {}
    debate_result_data: dict[str, Any] = {}
    _enable_structured_debate = bool(project_cfg.get("structured_debate", True))
    if _enable_structured_debate and llm_routers and submissions:
        from .debate import run_structured_debate
        # 选择辩论用路由器 (优先用 deep_model 配置的 provider)
        _debate_provider = project_cfg.get("llm", {}).get("debate_provider") or next(iter(llm_routers), None)
        _debate_router = llm_routers.get(_debate_provider) if _debate_provider else None
        if _debate_router:
            _use_multi_agent = bool(project_cfg.get("debate_multi_agent", False))
            _mode_tag = "multi-agent" if _use_multi_agent else "普通"
            print(f"[辩论] 启动结构化辩论 (provider={_debate_provider}, 仲裁={_mode_tag})", flush=True)
            pipeline.advance("结构化辩论")
            _current_price = float(analysis_context.get("snapshot", {}).get("close", 0) or 0)
            _debate_subs = [
                {
                    "agent_id": r.agent_id, "dim_code": r.dim_code,
                    "role": next((a.role for a in analysts if a.agent_id == r.agent_id), r.agent_id),
                    "score": r.score_0_100, "reason": r.reason,
                }
                for r in submissions
            ]
            # 构建备用router列表: 排除主辩论provider, 优先用gemini/minmax等
            _fallback_routers = [
                r for p, r in llm_routers.items()
                if p != _debate_provider
            ]
            # 提取日线斐波那契和ATR供仲裁参考
            _kli = analysis_context.get("features", {}).get("kline_indicators", {})
            _day_kli = _kli.get("day", {}) if isinstance(_kli, dict) else {}
            _fib = _day_kli.get("fibonacci") if isinstance(_day_kli, dict) else None
            _atr = _day_kli.get("atr") if isinstance(_day_kli, dict) else None
            try:
                _dr = run_structured_debate(
                    _debate_router, _debate_subs, symbol, name,
                    _current_price, debate_rounds=1,
                    fallback_routers=_fallback_routers,
                    use_multi_agent=_use_multi_agent,
                    fibonacci=_fib,
                    atr=float(_atr) if _atr else None,
                )
                debate_result_data = {
                    "decision": _dr.decision,
                    "score_override": _dr.score_override,
                    "target_price": _dr.target_price,
                    "stop_loss": _dr.stop_loss,
                    "confidence": _dr.confidence,
                    "risk_score": _dr.risk_score,
                    "reasoning": _dr.reasoning,
                    "plans": _dr.plans,
                    "team_reports": _dr.team_reports,
                    "debate_transcript": _dr.debate_transcript,
                    "risk_assessment": _dr.risk_assessment,
                }
                # 辩论结果保存
                dump_json(run_dir / "data" / "debate_result.json", debate_result_data)
                print(
                    f"[辩论] 完成: decision={_dr.decision} score={_dr.score_override} "
                    f"target={_dr.target_price} stop={_dr.stop_loss}",
                    flush=True,
                )
            except Exception as e:
                manager_logger.warning("structured debate failed: %s", e)
                print(f"[辩论] 结构化辩论失败: {e}, 使用加权评分", flush=True)
    elif debate_rounds > 0:
        # v1 旧辩论逻辑 (串行模式降级)
        for r in range(1, debate_rounds + 1):
            bull = max(submissions, key=lambda x: x.score_0_100)
            bear = min(submissions, key=lambda x: x.score_0_100)
            bull_msg = f"Bull观点：{bull.reason}。请反驳并给出风险点。"
            bear_msg = f"Bear观点：{bear.reason}。请回应并给出触发条件。"
            write_message(run_dir, bull.agent_id, bear.agent_id, r, bull_msg)
            write_message(run_dir, bear.agent_id, bull.agent_id, r, bear_msg)
            judge_msg = (
                f"Judge仲裁：Bull={bull.score_0_100:.1f}, Bear={bear.score_0_100:.1f}。"
                f"结论以证据完整性与数据时效优先。"
            )
            write_message(run_dir, manager_cfg["agent_id"], "all_agents", r, judge_msg)
            debate_bull_bear = {
                "bull_agent_id": bull.agent_id,
                "bull_role": next((a.role for a in analysts if a.agent_id == bull.agent_id), bull.agent_id),
                "bull_reason": bull.reason,
                "bull_score": round(bull.score_0_100, 2),
                "bear_agent_id": bear.agent_id,
                "bear_role": next((a.role for a in analysts if a.agent_id == bear.agent_id), bear.agent_id),
                "bear_reason": bear.reason,
                "bear_score": round(bear.score_0_100, 2),
                "judge_msg": judge_msg,
            }
            done += 1

    detail = []
    for a, result in zip(analysts, submissions):
        detail.append(
            {
                "agent_id": a.agent_id,
                "dim_code": result.dim_code,
                "role": a.role,
                "weight": a.weight,
                "vote": result.vote,
                "score_0_100": result.score_0_100,
                "confidence_0_1": result.confidence_0_1,
                "reason": result.reason,
                "llm_multi_evaluations": result.llm_multi_evaluations,
                "llm_scores": {p: sc.get(a.agent_id) for p, sc in model_scores.items() if sc.get(a.agent_id) is not None},
            }
        )

    model_totals: dict[str, float] = {}
    # v2: 不再需要 _INVERT_DIMS — PATTERN agent 内部已处理顶底结构反转
    _INVERT_DIMS: set[str] = set()

    # ── 逐Agent动态LLM权重计算 ──
    # 每个Agent有 llm_base_weight (config, 0.20/0.35/0.45)
    # 根据跨Provider一致性(σ)动态调整: consensus_factor = clamp(1 - (σ-3)/15, 0.5, 1.2)
    # 最终 llm_w = clamp(base * factor, 0.15, 0.50)
    def _calc_agent_llm_weight(agent_id: str, base_w: float, m_scores: dict) -> float:
        """计算单个Agent的动态LLM权重。"""
        scores_for_agent = []
        for p, sc in m_scores.items():
            s = sc.get(agent_id)
            if s is not None and 0 <= s <= 100:
                scores_for_agent.append(float(s))
        if len(scores_for_agent) < 2:
            return base_w  # 不足2个Provider，无法算一致性，用base
        mean = sum(scores_for_agent) / len(scores_for_agent)
        variance = sum((x - mean) ** 2 for x in scores_for_agent) / len(scores_for_agent)
        sigma = variance ** 0.5
        factor = max(0.5, min(1.2, 1.0 - (sigma - 3.0) / 15.0))
        return max(0.15, min(0.50, base_w * factor))

    # 构建 agent_id → llm_base_weight 映射
    _agent_llm_base = {a.agent_id: float(a.cfg.get("llm_base_weight", 0.35)) for a in analysts}

    if use_multi_model_weights and model_weights:
        # 构建 agent_id → 本地评分 映射
        local_scores = {d["agent_id"]: d["score_0_100"] for d in detail}
        _dim_map = {a.agent_id: a.dim_code for a in analysts}
        for p, w_map in model_weights.items():
            total = 0.0
            for a, res in zip(analysts, submissions):
                w = w_map.get(a.agent_id, 1.0 / len(analysts))
                # 逐Agent动态融合权重
                _llm_w = _calc_agent_llm_weight(a.agent_id, _agent_llm_base[a.agent_id], model_scores)
                _local_w = 1.0 - _llm_w
                # 融合本地评分与LLM评分
                local_s = local_scores.get(a.agent_id, res.score_0_100)
                llm_s = model_scores.get(p, {}).get(a.agent_id)
                if llm_s is not None and 0 <= llm_s <= 100:
                    score = local_s * _local_w + llm_s * _llm_w
                else:
                    score = local_s
                # v2: PATTERN agent内部已处理顶底反转, 无需外部反转
                if _dim_map.get(a.agent_id) in _INVERT_DIMS:
                    score = 100.0 - score
                total += score * w
            model_totals[p] = round(total, 4)
        if model_totals:
            final_score = sum(model_totals.values()) / len(model_totals)
        else:
            total_weight = sum(a.weight for a in analysts) or 1.0
            weighted_score = sum(
                (100.0 - r.score_0_100 if a.dim_code in _INVERT_DIMS else r.score_0_100) * a.weight
                for a, r in zip(analysts, submissions)
            )
            final_score = weighted_score / total_weight
    else:
        total_weight = sum(a.weight for a in analysts) or 1.0
        weighted_score = sum(
            (100.0 - r.score_0_100 if a.dim_code in _INVERT_DIMS else r.score_0_100) * a.weight
            for a, r in zip(analysts, submissions)
        )
        final_score = weighted_score / total_weight
    # ── v2: 辩论评分融合 ──
    # 辩论返回的 score 是"对 decision 的置信度"，需要对齐到 0-100 看多量表：
    #   decision=buy  → score 直接使用（高=看多）
    #   decision=sell → 100 - score（高置信卖出 = 低看多分）
    #   decision=hold → 固定 50（中性）
    _debate_score_raw = debate_result_data.get("score_override")
    _debate_decision = debate_result_data.get("decision", "").lower()
    if _debate_score_raw is not None and 0 <= float(_debate_score_raw) <= 100:
        _ds = float(_debate_score_raw)
        if _debate_decision == "sell":
            _debate_score_aligned = 100.0 - _ds  # sell+87 → 13（强烈看空）
        elif _debate_decision == "buy":
            _debate_score_aligned = _ds           # buy+80 → 80（看多）
        else:
            _debate_score_aligned = _ds            # hold → 用辩论原始分
        _debate_w = 0.40
        _weighted_score_before = final_score
        final_score = final_score * (1.0 - _debate_w) + _debate_score_aligned * _debate_w
        manager_logger.info(
            "debate score fusion: weighted=%.2f debate_raw=%s(%s) aligned=%.2f → final=%.2f",
            _weighted_score_before, _debate_score_raw, _debate_decision, _debate_score_aligned, final_score,
        )
        print(f"[辩论融合] 加权评分={_weighted_score_before:.1f} × 60% + 辩论评分={_ds:.0f}({_debate_decision})→对齐={_debate_score_aligned:.0f} × 40% = {final_score:.1f}", flush=True)

    # 基于市场状态动态调整阈值
    _regime = analysis_context.get("features", {}).get("market_regime", {}).get("regime", "unknown")
    if _regime == "bull":
        buy_th, sell_th = 65.0, 45.0
    elif _regime == "bear":
        buy_th, sell_th = 78.0, 55.0
    else:
        buy_th = float(project_cfg.get("decision_threshold_buy", 70.0))
        sell_th = float(project_cfg.get("decision_threshold_sell", 50.0))
    if final_score >= buy_th:
        final_decision = "buy"
    elif final_score < sell_th:
        final_decision = "sell"
    else:
        final_decision = "hold"
    # 五级决策：strong_sell(<40), weak_sell(40-50), hold(50-70), weak_buy(70-85), strong_buy(>=85)
    if final_score >= 85:
        decision_level = "strong_buy"
    elif final_score >= 70:
        decision_level = "weak_buy"
    elif final_score >= 50:
        decision_level = "hold"
    elif final_score >= 40:
        decision_level = "weak_sell"
    else:
        decision_level = "strong_sell"
    # ── 乖离率后置过滤 ──
    bias_warning = ""
    _bias_cfg = project_cfg.get("bias_filter", {})
    if _bias_cfg.get("enabled", False):
        _bias_pct = float(
            analysis_context.get("features", {}).get("kline_indicators", {})
            .get("day", {}).get("ma_system", {}).get("ma5", {}).get("pct_above", 0)
        )
        _bias_th = float(_bias_cfg.get("bias_threshold_pct", 5.0))
        if _bias_pct > _bias_th:
            if decision_level in ("strong_buy", "weak_buy"):
                decision_level = "hold"
                final_decision = "hold"
                bias_warning = f"⚠️ 乖离率警告: MA5乖离{_bias_pct:.1f}%>{_bias_th}%，建议等待回踩后介入"
            elif decision_level == "hold":
                bias_warning = f"⚠️ 乖离率偏高({_bias_pct:.1f}%)，谨慎追高"

    done += 1
    print(
        f"[评分] 最终={final_decision} score={final_score:.2f} (阈值: buy>={buy_th} sell<{sell_th}) 市场={_regime}",
        flush=True,
    )
    if bias_warning:
        print(f"[乖离率] {bias_warning}", flush=True)

    # 短线/中长线建议：基于周线、月线趋势
    kli = analysis_context.get("features", {}).get("kline_indicators", {})
    week_ind = kli.get("week", {}) if isinstance(kli, dict) else {}
    month_ind = kli.get("month", {}) if isinstance(kli, dict) else {}
    mom_week = float(week_ind.get("momentum_10", 0)) if isinstance(week_ind, dict) else 0.0
    mom_month = float(month_ind.get("momentum_10", 0)) if isinstance(month_ind, dict) else 0.0
    has_month = isinstance(month_ind, dict) and month_ind.get("ok")
    short_term_hold = "建议持有" if final_score >= 50 else "建议观望或减仓"
    if has_month and mom_week > 0 and mom_month > 0:
        medium_long_term_hold = "建议中长线持有"
    elif has_month:
        medium_long_term_hold = "暂不建议中长线持有"
    else:
        medium_long_term_hold = "月线数据不足，以周线为准；周线向好可考虑中长线"

    decision_level_cn = {
        "strong_buy": "强烈买入", "weak_buy": "弱买入", "hold": "观望",
        "weak_sell": "弱卖出", "strong_sell": "强烈卖出",
    }.get(decision_level, decision_level)
    scenario_analysis, position_strategy = "", ""
    scenarios_data: dict = {}
    sniper_points: dict = {}
    position_advice: dict = {}
    if llm_routers:
        key_levels = analysis_context.get("features", {}).get("key_levels", {}) or {}
        kl_summary = ""
        if isinstance(key_levels, dict) and key_levels.get("ok"):
            kl_summary = f"高点{key_levels.get('band_high')} 低点{key_levels.get('band_low')} 当前{key_levels.get('current')} 38.2%回撤{key_levels.get('retrace_382')}"
        _snap_close = analysis_context.get("features", {}).get("close")
        _current_price = float(_snap_close) if _snap_close else None
        first_router = next(iter(llm_routers.values()), None)
        if first_router:
            print("[情景] 生成情景分析与建仓/止损建议", flush=True)
            # 情景分析响应较长，临时增大 max_tokens
            _orig_max_tokens = first_router.max_tokens
            first_router.max_tokens = 32768
            try:
                _sr = generate_scenario_and_position(
                    first_router, symbol, name, final_score, decision_level_cn, kl_summary,
                    current_price=_current_price,
                )
            finally:
                first_router.max_tokens = _orig_max_tokens
            scenarios_data = _sr.get("scenarios", {})
            sniper_points = _sr.get("sniper_points", {})
            position_strategy = _sr.get("position_strategy", "")
            position_advice = _sr.get("position_advice", {})
            # 新增结构化字段
            rating = _sr.get("rating", "")
            executive_summary = _sr.get("executive_summary", "")
            investment_thesis = _sr.get("investment_thesis", "")
            # 兼容旧格式scenario_analysis
            if scenarios_data:
                parts = []
                for k in ("optimistic", "neutral", "pessimistic"):
                    s = scenarios_data.get(k, {})
                    if isinstance(s, dict) and s.get("reason"):
                        parts.append(f"{k}: {s['reason']} (概率{s.get('probability', '?')}%)")
                scenario_analysis = "；".join(parts) if parts else ""

    # 提取斐波那契和入场方案供报告使用
    _kli_day_fib = kli.get("day", {}).get("fibonacci") if isinstance(kli, dict) else None
    _entry_plans = debate_result_data.get("plans", []) if debate_result_data else []

    output = {
        "symbol": symbol,
        "name": name,
        "final_score": round(final_score, 4),
        "final_decision": final_decision,
        "decision_level": decision_level,
        "thresholds": {"buy": buy_th, "sell": sell_th},
        "agent_votes": detail,
        "debate_rounds": debate_rounds,
        "provider": provider,
        "multi_eval_providers": list(llm_routers.keys()) if llm_routers else [],
        "multi_model_weight_mode": use_multi_model_weights,
        "model_weights": model_weights if use_multi_model_weights else {},
        "model_scores": model_scores if use_multi_model_weights else {},
        "model_totals": model_totals if use_multi_model_weights else {},
        "llm_fusion_weights": {
            a.agent_id: round(_calc_agent_llm_weight(a.agent_id, _agent_llm_base[a.agent_id], model_scores), 4)
            for a in analysts
        } if use_multi_model_weights else {},
        "analysis_data_files": analysis_context.get("data_files", {}),
        "analysis_features": analysis_context.get("features", {}),
        "score_mapping": {
            "strong_buy": "S >= 85",
            "weak_buy": "70 <= S < 85",
            "hold": "50 <= S < 70",
            "weak_sell": "40 <= S < 50",
            "strong_sell": "S < 40",
        },
        "short_term_hold": short_term_hold,
        "medium_long_term_hold": medium_long_term_hold,
        "debate_bull_bear": debate_bull_bear if debate_rounds else {},
        "structured_debate": debate_result_data if debate_result_data else {},
        "scenario_analysis": scenario_analysis,
        "scenarios": scenarios_data,
        "sniper_points": sniper_points,
        "position_strategy": position_strategy,
        "position_advice": position_advice,
        "rating": rating,
        "executive_summary": executive_summary,
        "investment_thesis": investment_thesis,
        "warnings": [w for w in [bias_warning] if w],
        "fibonacci": _kli_day_fib,
        "entry_plans": _entry_plans,
    }
    # ── 阶段5: 输出报告 ──
    pipeline.advance("输出报告")
    json_path = run_dir / "final_decision.json"
    dump_json(json_path, output)
    pdf_path = build_investor_pdf(run_dir, output)
    output["final_pdf_path"] = str(pdf_path)
    output["final_json_path"] = str(json_path)
    dump_json(json_path, output)
    manager_logger.info("final_decision=%s score=%.4f", final_decision, final_score)
    pipeline.finish()
    print(f"[完成] PDF={pdf_path.name}", flush=True)

    # ── 记录信号到回测数据库 ──
    try:
        from .backtest import record_signal
        _features = analysis_context.get("features", {})
        _snap = {
            "symbol": symbol,
            "name": name,
            "close": _features.get("close"),
        }
        _fd = {"score": final_score, "decision": final_decision, "sniper_points": sniper_points}
        record_signal(_fd, detail, _snap, _features)
    except Exception:
        pass

    # ── 保存评分快照到历史目录（支持逐日对比） ──
    try:
        from .score_history import save_score_snapshot
        save_score_snapshot(run_dir, output)
    except Exception:
        pass

    # ── BM25 记忆存入 ─────────────────────────────────────
    try:
        from .memory import BM25Memory
        _feat = analysis_context.get("features", {})
        _mem = BM25Memory.get_instance()
        _mem.add_decision(output, _feat, _feat)
    except Exception:
        pass

    return output
