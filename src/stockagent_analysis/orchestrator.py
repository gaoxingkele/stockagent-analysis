# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .agent_data_mapping import build_agent_data_table, format_agent_data_table
from .agents import AnalystAgent, AgentResult, AgentBaseResult, write_message, create_agent, _parse_vision_response
from .config_loader import load_agent_configs, load_project_config, split_agents
from .data_backend import DataBackend
from .io_utils import dump_json, get_agent_logger
from .llm_client import LLMRouter, assign_agent_weights, score_agent_analysis, generate_scenario_and_position
from .report_pdf import build_investor_pdf


def _print_progress(stage: str, done: int, total: int, detail: str = "") -> None:
    pct = (done / total * 100) if total else 100
    msg = f"[进度] {stage}: {done}/{total} ({pct:.1f}%)"
    if detail:
        msg += f" | {detail}"
    print(msg, flush=True)


def _print_data_progress(step: str, detail: str) -> None:
    print(f"[数据] {step}: {detail}", flush=True)


def _print_agent_progress(stage: str, detail: str) -> None:
    print(f"[智能体] {stage}: {detail}", flush=True)


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

    # 1. is_complete
    integrity = analysis_context.get("data_integrity", {})
    if not integrity.get("is_complete", False):
        missing.append("data_integrity.is_complete=False")

    # 2. analysis_context.json 文件存在
    ctx_path = run_dir / "data" / "analysis_context.json"
    if not ctx_path.exists():
        missing.append(f"analysis_context.json ({ctx_path})")

    # 3. K线CSV文件
    data_dir = run_dir / "data"
    for tf in ("day", "week", "month"):
        csv_path = data_dir / "kline" / f"{tf}.csv"
        if not csv_path.exists():
            missing.append(f"kline/{tf}.csv")

    # 4. kline_vision agent 依赖的 chart PNG 文件
    chart_files = analysis_context.get("chart_files", {})
    for cfg in analyst_cfg:
        if cfg.get("agent_type") != "kline_vision":
            continue
        tf = cfg.get("timeframe", "")
        chart_path_str = chart_files.get(tf, "")
        if chart_path_str:
            if not Path(chart_path_str).exists():
                missing.append(f"K线图表 {tf}: {chart_path_str}")
        else:
            # chart_files 中没有此 timeframe 的条目
            missing.append(f"K线图表 {tf}: 未生成")

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
    from .parallel_runner import ProviderResult

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
    provider = (llm_provider_override or llm_cfg.get("default_provider", "kimi")).lower()
    multi_eval_providers = llm_cfg.get("multi_eval_providers", ["kimi"])
    if multi_eval_providers_override:
        multi_eval_providers = [p.strip().lower() for p in multi_eval_providers_override.split(",") if p.strip()]
        if multi_eval_providers:
            provider = multi_eval_providers[0]
    elif llm_provider_override:
        multi_eval_providers = [provider] + [p for p in multi_eval_providers if p != provider]
    llm_routers: dict[str, LLMRouter] = {}
    multi_turn = bool(llm_cfg.get("multi_turn_default", True))
    if llm_enabled:
        for p in multi_eval_providers:
            llm_routers[p] = LLMRouter(
                provider=p,
                temperature=float(llm_cfg.get("temperature", 0.3)),
                max_tokens=int(llm_cfg.get("max_tokens", 600)),
                request_timeout_sec=float(llm_cfg.get("request_timeout_sec", 25.0)),
                multi_turn=multi_turn,
            )
    manager_logger = get_agent_logger(run_dir, manager_cfg["agent_id"])

    manager_logger.info("start symbol=%s name=%s analysts=%s", symbol, name, [a["agent_id"] for a in analyst_cfg])
    manager_logger.info("llm_enabled=%s providers=%s", llm_enabled, list(llm_routers.keys()))

    multi_model_weight_mode = bool(project_cfg.get("multi_model_weight_mode", False))
    explicit = llm_cfg.get("multi_model_weight_providers", [])
    if multi_eval_providers_override:
        weight_providers = [p for p in list(llm_routers.keys()) if p in llm_routers]
    else:
        weight_providers = [p for p in (explicit or list(llm_routers.keys())) if p in llm_routers]
    weight_providers = weight_providers[:10]
    use_multi_model_weights = multi_model_weight_mode and len(weight_providers) >= 2

    print(
        f"[启动] 股票 {symbol} {name} | 智能体数量: {len(analyst_cfg)} | "
        f"综合评估API: {', '.join(llm_routers.keys()) if llm_routers else 'disabled'}"
        + (f" | 多模型权重模式: {weight_providers}" if use_multi_model_weights else ""),
        flush=True
    )
    # ── 断点续传：检查是否已有 analysis_context.json ──
    existing_ctx_path = run_dir / "data" / "analysis_context.json"
    if existing_ctx_path.exists():
        print("[断点续传] 检测到已有数据，跳过数据采集，直接进入分析阶段", flush=True)
        analysis_context = json.loads(existing_ctx_path.read_text(encoding="utf-8"))
        manager_logger.info("resume: loaded existing analysis_context from %s", existing_ctx_path)
    else:
        print("[进度] 数据采集与落地: 开始", flush=True)
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
    # 仅当所有数据获取成功才继续；否则终止 LLM 调用，不输出研判报告
    if not is_complete:
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
    analysts = [create_agent(cfg, run_dir, backend, llm_routers=llm_routers) for cfg in analyst_cfg]
    print("[进度] 数据采集与落地: 完成", flush=True)

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

        # 阶段2: 本地策略分析（无LLM调用，秒级完成）
        print("[进度] 本地策略分析: 开始", flush=True)
        base_results: dict[str, AgentBaseResult] = {}
        for a in analysts:
            base_results[a.agent_id] = a.analyze_local(symbol, name, analysis_context)
        print(f"[进度] 本地策略分析: 完成 ({len(base_results)}个智能体)", flush=True)

        # 阶段3: 多Provider并行执行（权重分配+研判增强+独立评分）
        task_summary = _build_agent_task_summary(analyst_cfg)
        (run_dir / "data").mkdir(parents=True, exist_ok=True)
        (run_dir / "data" / "task_summary.txt").write_text(task_summary, encoding="utf-8")

        parallel_routers = {p: llm_routers[p] for p in weight_providers}
        provider_results = run_providers_parallel(
            parallel_routers, base_results, analyst_cfg, symbol, name, task_summary,
            run_dir=run_dir,
        )

        # 阶段4: 合并结果
        print("[进度] 合并多Provider结果: 开始", flush=True)
        submissions, model_weights, model_scores = _merge_provider_results(
            analysts, base_results, provider_results, analysis_context,
        )
        ok_providers = [p for p, pr in provider_results.items() if not pr.error]
        fail_providers = [p for p, pr in provider_results.items() if pr.error]
        manager_logger.info(
            "parallel merge done: ok=%s fail=%s weights_keys=%s scores_keys=%s",
            ok_providers, fail_providers, list(model_weights.keys()), list(model_scores.keys()),
        )
        print(f"[进度] 合并多Provider结果: 完成 (成功:{len(ok_providers)} 失败:{len(fail_providers)})", flush=True)

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
        total_steps = len(analysts) + debate_rounds + 2
        done = 0

        def agent_progress_cb(s: str, d: str) -> None:
            _print_agent_progress(s, d)

        for a in analysts:
            _print_progress("多智能体分析", done, total_steps, detail=f"正在分析 {a.agent_id} ({a.role})")
            submissions.append(
                a.analyze(
                    symbol, name,
                    analysis_context=analysis_context,
                    progress_cb=agent_progress_cb,
                )
            )
            done += 1
            _print_progress("多智能体分析", done, total_steps, detail=f"完成 {a.agent_id} vote={submissions[-1].vote} score={submissions[-1].score_0_100:.1f}")

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
        _print_progress(
            "辩论仲裁", done, total_steps,
            detail=f"第{r}轮 Bull={bull.agent_id}({bull.score_0_100:.1f}) vs Bear={bear.agent_id}({bear.score_0_100:.1f})",
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
            }
        )

    model_totals: dict[str, float] = {}
    if use_multi_model_weights and model_weights:
        for p, w_map in model_weights.items():
            total = 0.0
            s_map = model_scores.get(p, {})
            for a, res in zip(analysts, submissions):
                w = w_map.get(a.agent_id, 1.0 / len(analysts))
                score = s_map.get(a.agent_id, res.score_0_100)
                total += score * w
            model_totals[p] = round(total, 4)
        if model_totals:
            final_score = sum(model_totals.values()) / len(model_totals)
        else:
            total_weight = sum(a.weight for a in analysts) or 1.0
            weighted_score = sum(r.score_0_100 * a.weight for a, r in zip(analysts, submissions))
            final_score = weighted_score / total_weight
    else:
        total_weight = sum(a.weight for a in analysts) or 1.0
        weighted_score = sum(r.score_0_100 * a.weight for a, r in zip(analysts, submissions))
        final_score = weighted_score / total_weight
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
    _print_progress(
        "加权评分", done, total_steps,
        detail=f"最终={final_decision} score={final_score:.2f} (阈值: buy>={buy_th} sell<{sell_th})",
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
            _print_agent_progress("情景与策略", "生成情景分析与建仓/止损建议")
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
    json_path = run_dir / "final_decision.json"
    dump_json(json_path, output)
    pdf_path = build_investor_pdf(run_dir, output)
    output["final_pdf_path"] = str(pdf_path)
    output["final_json_path"] = str(json_path)
    dump_json(json_path, output)
    manager_logger.info("final_decision=%s score=%.4f", final_decision, final_score)
    done += 1
    _print_progress("输出结果", done, total_steps, detail=f"PDF={pdf_path.name}")
    return output
