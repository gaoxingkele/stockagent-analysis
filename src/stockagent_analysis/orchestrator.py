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

    if multi_eval_providers_override:
        # --providers 显式指定默认Provider
        default_providers = [p.strip().lower() for p in multi_eval_providers_override.split(",") if p.strip()]
    elif llm_provider_override:
        default_providers = [llm_provider_override.lower()]
    else:
        default_providers = list(default_providers_cfg) if default_providers_cfg else llm_cfg.get("multi_eval_providers", ["kimi"])

    provider = default_providers[0] if default_providers else "kimi"

    # 过滤：只保留有API_KEY的Provider
    default_providers = [p for p in default_providers if os.getenv(f"{p.upper()}_API_KEY", "").strip()]
    candidate_providers = [
        p for p in candidate_providers_cfg
        if p not in default_providers and os.getenv(f"{p.upper()}_API_KEY", "").strip()
    ]

    # 全部Provider = 默认 + 候选，都启动子任务
    all_providers = default_providers + candidate_providers

    llm_routers: dict[str, LLMRouter] = {}
    if llm_enabled:
        for p in all_providers:
            llm_routers[p] = LLMRouter(
                provider=p,
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
    # day/week 必须；month 缺失可降级继续
    only_optional_missing = set(failed_items) <= {"month"}
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

    debate_bull_bear: dict[str, Any] = {}
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
        manager_logger.info("debate_round=%s bull=%s bear=%s", r, bull.agent_id, bear.agent_id)
        done += 1
        print(
            f"[辩论] {done}/{total_steps} 第{r}轮 Bull={agent_registry.get(bull.agent_id)}({bull.score_0_100:.1f}) vs Bear={agent_registry.get(bear.agent_id)}({bear.score_0_100:.1f})",
            flush=True,
        )

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
    # TOP_STRUCTURE 评分语义: 高分=顶部信号强(卖出)，在加权时需反转(100-score)使其拉低总分
    _INVERT_DIMS = {"TOP_STRUCTURE"}

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
                # TOP_STRUCTURE: 高分=顶部强→反转后拉低总分
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
    done += 1
    print(
        f"[评分] 最终={final_decision} score={final_score:.2f} (阈值: buy>={buy_th} sell<{sell_th}) 市场={_regime}",
        flush=True,
    )

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
    if llm_routers:
        key_levels = analysis_context.get("features", {}).get("key_levels", {}) or {}
        kl_summary = ""
        if isinstance(key_levels, dict) and key_levels.get("ok"):
            kl_summary = f"高点{key_levels.get('band_high')} 低点{key_levels.get('band_low')} 当前{key_levels.get('current')} 38.2%回撤{key_levels.get('retrace_382')}"
        first_router = next(iter(llm_routers.values()), None)
        if first_router:
            print("[情景] 生成情景分析与建仓/止损建议", flush=True)
            scenario_analysis, position_strategy = generate_scenario_and_position(
                first_router, symbol, name, final_score, decision_level_cn, kl_summary
            )

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
        "scenario_analysis": scenario_analysis,
        "position_strategy": position_strategy,
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
        _fd = {"score": final_score, "decision": final_decision}
        record_signal(_fd, detail, _snap, _features)
    except Exception:
        pass

    # ── 保存评分快照到历史目录（支持逐日对比） ──
    try:
        from .score_history import save_score_snapshot
        save_score_snapshot(run_dir, output)
    except Exception:
        pass

    return output
