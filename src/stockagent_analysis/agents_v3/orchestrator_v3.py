"""v3 流水线主入口 - 串联 Phase 0-5。

流程:
  Phase 0: 量化事实层(纯 Python, 6 份报告)
  Phase 1: 专业视角分析师(4 个 LLM 角色并行)
  Phase 2: 多空辩论(Bull/Bear 3 轮 + Judge)
  Phase 3: 首席交易员(产出交易方案)
  Phase 4: 风控三辩论 + Portfolio Manager
  Phase 5: 综合评分 + 报告生成
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..data_backend import DataBackend
from .phase0_data import build_all_reports, ReportBundle
from .phase1_experts import run_all_experts, ExpertResult
from .phase2_debate import run_investment_debate, InvestmentPlan
from .phase3_trader import run_head_trader, TradingPlan
from .phase4_risk import run_risk_debate, RiskPolicy
from .memory_v3 import get_memory
from .memory_v3.role_memory import MemoryRecord

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "runs_v3"


def _normalize_symbol(symbol: str) -> str:
    s = (symbol or "").upper().strip()
    if "." in s:
        s = s.split(".")[0]
    return s


def _make_run_dir(symbol: str, name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = _OUTPUT_ROOT / f"{ts}_{symbol}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "data").mkdir(exist_ok=True)
    (run / "messages").mkdir(exist_ok=True)
    return run


def _compose_final_score(
    experts: dict[str, ExpertResult],
    investment_plan: InvestmentPlan,
    trading_plan: TradingPlan,
    risk_policy: RiskPolicy,
) -> tuple[float, dict[str, float]]:
    """融合专家+Judge+Trader+PM 产出最终评分(v3 融合公式)。

    改进点(v3, 2026-04-19):
    1. Risk 映射拉伸: [38,73] → [20,85] (仓位跨度更大)
    2. Judge 分重构 judge_adj 为辩论综合质量分, 避免 LLM 心理挡位扎堆:
       - overall_score 基底 × 0.65
       - expert_avg 融合 × 0.15 (借 Expert 分散度冲击 Judge 集中性)
       - confidence 偏移 ±方向性(BUY 加/SELL 减)
       - winning_points 条数 × 1.5 (胜方论据数量)
       - key_reasons 文本量加成 (论据详尽度)
    3. 三层方向一致性奖励: Judge/Trader/PM 共识时 +5 或 -5
    4. 冲突惩罚: Judge vs Trader 方向相反时 -4(降低置信)

    融合系数: α=0.50·expert + β=0.32·judge_adj + γ=0.18·risk + bonus
    """
    # FOMC 点阵图模式: 收集 4 专家详情, 计算均值/中位数/分歧度/异议
    expert_details = []
    for role_id, er in experts.items():
        if isinstance(er, ExpertResult):
            s = er.score
            cn = er.role_cn
        else:
            # dict-like (反序列化场景)
            try:
                s = float(er.get("score", 50))
                cn = er.get("role_cn", role_id)
            except Exception:
                continue
        if 0 <= s <= 100:
            expert_details.append({"role": role_id, "role_cn": cn, "score": s})

    scores = [d["score"] for d in expert_details]
    n = len(scores)
    expert_avg = sum(scores) / max(n, 1) if scores else 50.0

    # 中位数(偶数个取中间两个均值)
    if scores:
        s_sorted = sorted(scores)
        if n % 2 == 1:
            expert_median = s_sorted[n // 2]
        else:
            expert_median = (s_sorted[n // 2 - 1] + s_sorted[n // 2]) / 2
    else:
        expert_median = 50.0

    # 标准差(分歧度)
    if n > 1:
        _mean = expert_avg
        expert_std = (sum((s - _mean) ** 2 for s in scores) / (n - 1)) ** 0.5
    else:
        expert_std = 0.0

    # 识别异议专家(偏离中位数超过 15 分)
    dissent = []
    for d in expert_details:
        dev = d["score"] - expert_median
        if abs(dev) >= 15:
            dissent.append({
                "role_cn": d["role_cn"],
                "score": d["score"],
                "deviation": round(dev, 1),
                "direction": "偏多" if dev > 0 else "偏空",
            })

    # 共识分: 用中位数为主, 叠加分歧惩罚(均值与中位数偏离越大 → 共识越低)
    gap = abs(expert_avg - expert_median)
    consensus_penalty = min(gap * 0.3, 3.0)   # 最多扣 3 分
    expert_consensus = max(0.0, min(100.0, expert_median - consensus_penalty))

    judge_score_raw = float(investment_plan.overall_score) if investment_plan.overall_score else 50.0
    j_dir = (investment_plan.direction or "HOLD").upper()
    j_conf = float(investment_plan.confidence or 0.5)

    # 1) Judge 分重构 — 融合多维度避免 LLM 整齐挡位扎堆
    # 基底: Judge overall_score 占 65%, Expert 共识分(FOMC 中位数)冲击 15%
    base = judge_score_raw * 0.65 + expert_consensus * 0.15

    # 方向性置信偏移: conf=0.5→0, conf=0.75→+5, conf=0.85→+7
    conf_offset = (j_conf - 0.5) * 20
    if j_dir == "BUY":
        direction_shift = conf_offset    # BUY 高置信 → +
    elif j_dir == "SELL":
        direction_shift = -conf_offset   # SELL 高置信 → -
    else:
        direction_shift = conf_offset * 0.3  # HOLD 权重弱化

    # 论据质量因子: 胜方论据数量(0-3) + 理由详尽度(字符数)
    winning_points = investment_plan.winning_points or []
    key_reasons = investment_plan.key_reasons or []
    argument_bonus = 0.0
    argument_bonus += min(len(winning_points), 3) * 1.5  # 0-4.5
    reason_chars = sum(len(str(r)) for r in key_reasons)
    argument_bonus += min(reason_chars / 80.0, 6.0)   # 0-6

    judge_adj = base + direction_shift + argument_bonus
    judge_adj = max(0.0, min(100.0, judge_adj))

    # 2) Risk 映射拉伸: 仓位 [0,1] → 分数 [20,80], 叠加 rating_bonus
    rating = (risk_policy.final_risk_rating or "中").strip()
    rating_bonus = {"低": +8.0, "中": 0.0, "高": -10.0}.get(rating, 0.0)
    try:
        maxpos = float(risk_policy.max_position_ratio or 0.3)
    except (ValueError, TypeError):
        maxpos = 0.3
    risk_base = 20 + maxpos * 60  # 0仓位→20, 0.5仓位→50, 1仓位→80
    risk_mapped = max(0.0, min(100.0, risk_base + rating_bonus))

    # 3) 三层方向一致性奖励/冲突惩罚
    t_dir = (trading_plan.final_decision or "HOLD").upper()
    pm_bias = (risk_policy.alignment_with or "neutral").lower()
    bonus = 0.0
    bonus_reason = "无"
    if j_dir == "BUY" and t_dir == "BUY":
        bonus = 5.0 if pm_bias in ("aggressive", "neutral") else 2.0
        bonus_reason = f"Judge+Trader 共识 BUY | PM={pm_bias}"
    elif j_dir == "SELL" and t_dir == "SELL":
        bonus = -5.0 if pm_bias in ("conservative", "neutral") else -2.0
        bonus_reason = f"Judge+Trader 共识 SELL | PM={pm_bias}"
    elif (j_dir == "BUY" and t_dir == "SELL") or (j_dir == "SELL" and t_dir == "BUY"):
        bonus = -4.0
        bonus_reason = f"Judge({j_dir}) vs Trader({t_dir}) 冲突"

    # 4) 最终融合(用 FOMC 共识分替代均值)
    final_raw = 0.50 * expert_consensus + 0.32 * judge_adj + 0.18 * risk_mapped + bonus
    final = round(max(0.0, min(100.0, final_raw)), 2)

    return final, {
        "expert_avg": round(expert_avg, 2),
        "expert_median": round(expert_median, 2),
        "expert_std": round(expert_std, 2),
        "expert_consensus": round(expert_consensus, 2),
        "expert_details": expert_details,
        "dissent": dissent,
        "judge_score": round(judge_score_raw, 2),
        "judge_adj": round(judge_adj, 2),
        "risk_mapped": round(risk_mapped, 2),
        "consensus_bonus": round(bonus, 2),
        "bonus_reason": bonus_reason,
        "final_score": final,
    }


def _decide_level(final_score: float, trader_decision: str) -> tuple[str, str]:
    """根据融合分与 Trader 决策确定最终等级。

    阈值重新校准 (去极端化后分布拉开):
    - ≥80     → strong_buy
    - 72-79.9 → weak_buy (若 Trader=SELL 则 hold)
    - 62-71.9 → hold
    - 52-61.9 → watch (hold 偏悲观,持有者可开始减)
    - 42-51.9 → weak_sell
    - <42     → strong_sell

    Returns: (final_decision, decision_level)
    """
    td = (trader_decision or "HOLD").upper()
    if final_score >= 80:
        return "buy", "strong_buy"
    if final_score >= 72:
        return ("buy", "weak_buy") if td != "SELL" else ("hold", "hold")
    if final_score >= 62:
        return "hold", "hold"
    if final_score >= 52:
        return "hold", "watch_sell"
    if final_score >= 42:
        return "sell", "weak_sell"
    return "sell", "strong_sell"


def run_analysis_v3(
    symbol: str,
    name: str = "",
    *,
    bull_provider: str = "grok",
    bear_provider: str = "grok",
    judge_provider: str = "grok",
    trader_provider: str = "grok",
    pm_provider: str = "grok",
    expert_providers: dict[str, str] | None = None,
    debate_rounds: int = 3,
    risk_rounds: int = 2,
    reuse_context_from: Path | None = None,
    run_dir: Path | None = None,
    save_memory: bool = True,
) -> dict[str, Any]:
    """v3 主流水线。

    Args:
        symbol: 股票代码(6 位)
        name: 股票名称
        bull/bear/judge/trader/pm_provider: 各角色 LLM provider
        expert_providers: Phase1 4 专家的 provider 覆盖
        debate_rounds: Phase 2 辩论轮数(默认 3)
        risk_rounds: Phase 4 辩论轮数(默认 2)
        reuse_context_from: 复用旧 run_dir 的 analysis_context.json(调试用)
        run_dir: 自定义输出目录(默认按时间戳建)
        save_memory: 是否把本次决策写入角色记忆库

    Returns: final_result dict
    """
    symbol = _normalize_symbol(symbol)
    t0 = time.time()

    run_dir = run_dir or _make_run_dir(symbol, name)
    logger.info("[v3] 启动 %s %s → %s", symbol, name, run_dir)

    # ── Phase -1: 数据采集 ──
    ctx_path = run_dir / "data" / "analysis_context.json"
    if reuse_context_from:
        src_dir = Path(reuse_context_from)
        src = src_dir / "data" / "analysis_context.json"
        if src.exists():
            ctx = json.loads(src.read_text(encoding="utf-8"))
            ctx_path.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[v3] 复用已有 analysis_context from %s", reuse_context_from)
            # 复用 kline 目录(StructureExpert 绘图依赖)
            src_kline = src_dir / "data" / "kline"
            dst_kline = run_dir / "data" / "kline"
            if src_kline.exists() and not dst_kline.exists():
                import shutil
                try:
                    shutil.copytree(src_kline, dst_kline)
                    logger.info("[v3] 复用 kline 目录")
                except Exception as e:
                    logger.warning("[v3] kline 复制失败(非致命): %s", e)
        else:
            raise FileNotFoundError(f"复用路径不存在: {src}")
    else:
        backend = DataBackend()
        logger.info("[v3] Phase -1: 数据采集中...")
        ctx = backend.collect_and_save_context(symbol, name, str(run_dir))

    if not name:
        name = (ctx.get("snapshot", {}) or {}).get("name", "")

    # ── Tushare 高级数据增强 (真实筹码/主力资金/技术因子) ──
    try:
        from ..tushare_enrich import enrich_with_tushare
        logger.info("[v3] Tushare 增强: 拉取 stk_factor_pro/cyq_perf/moneyflow...")
        ts_enrich = enrich_with_tushare(symbol, run_dir=run_dir, use_cache=True)
        # 并入 ctx.features 供 Phase 0 报告使用
        if ts_enrich:
            ctx.setdefault("features", {})
            for key in ("tushare_factors", "tushare_cyq", "tushare_cyq_chips",
                         "tushare_moneyflow", "tushare_holders"):
                if key in ts_enrich:
                    ctx["features"][key] = ts_enrich[key]
            logger.info("[v3] Tushare 增强完成: %s",
                        [k for k in ("tushare_factors","tushare_cyq","tushare_moneyflow","tushare_holders") if k in ts_enrich])
            # 回写 analysis_context.json(含增强数据, 方便下游复用)
            ctx_path.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("[v3] Tushare 增强失败(非致命, 继续): %s", e)

    # ── Phase 0: 量化事实层 ──
    logger.info("[v3] Phase 0: 生成 6 份客观报告")
    bundle = build_all_reports(symbol, name, ctx)
    (run_dir / "data" / "phase0_bundle.md").write_text(bundle.as_markdown(), encoding="utf-8")
    (run_dir / "data" / "phase0_bundle.json").write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Phase 1: 专业视角分析师(并行) ──
    logger.info("[v3] Phase 1: 4 专家并行分析")
    experts = run_all_experts(bundle, providers=expert_providers, parallel=True, run_dir=run_dir)
    (run_dir / "data" / "phase1_experts.json").write_text(
        json.dumps({r: e.to_dict() for r, e in experts.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ── 记忆召回 ──
    situation_summary = f"{symbol} {name} | " + bundle.technical[:500] + " | " + bundle.sentiment[:300]
    bull_mem = get_memory("bull").retrieve(situation_summary)
    bear_mem = get_memory("bear").retrieve(situation_summary)
    judge_mem = get_memory("judge").retrieve(situation_summary)
    trader_mem = get_memory("trader").retrieve(situation_summary)
    pm_mem = get_memory("pm").retrieve(situation_summary)

    past_mems_inv = {
        "bull": get_memory("bull").format_for_prompt(bull_mem),
        "bear": get_memory("bear").format_for_prompt(bear_mem),
        "judge": get_memory("judge").format_for_prompt(judge_mem),
    }
    past_mems_risk = {
        "aggressive": "",
        "conservative": "",
        "neutral": "",
        "pm": get_memory("pm").format_for_prompt(pm_mem),
    }

    # ── Phase 2: 多空辩论 + Judge 仲裁 ──
    logger.info("[v3] Phase 2: %d 轮多空辩论 + Judge", debate_rounds)
    debate_out = run_investment_debate(
        bundle, experts,
        rounds=debate_rounds,
        bull_provider=bull_provider, bear_provider=bear_provider,
        judge_provider=judge_provider,
        past_memories=past_mems_inv,
    )
    investment_plan: InvestmentPlan = debate_out["investment_plan"]
    (run_dir / "data" / "phase2_debate.md").write_text(
        debate_out["transcript"], encoding="utf-8")
    (run_dir / "data" / "phase2_investment_plan.json").write_text(
        json.dumps(investment_plan.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ── Phase 3: 首席交易员 ──
    logger.info("[v3] Phase 3: 首席交易员产出交易方案")
    trader_mem_text = get_memory("trader").format_for_prompt(trader_mem)
    trading_plan: TradingPlan = run_head_trader(
        bundle, experts, investment_plan,
        past_memories=trader_mem_text,
        provider=trader_provider,
    )
    (run_dir / "data" / "phase3_trading_plan.json").write_text(
        json.dumps(trading_plan.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ── Phase 4: 风控三辩论 + Portfolio Manager ──
    logger.info("[v3] Phase 4: %d 轮风控辩论 + PM 拍板", risk_rounds)
    risk_out = run_risk_debate(
        bundle, investment_plan, trading_plan,
        rounds=risk_rounds,
        aggressive_provider=pm_provider, conservative_provider=pm_provider, neutral_provider=pm_provider,
        pm_provider=pm_provider,
        past_memories=past_mems_risk,
    )
    risk_policy: RiskPolicy = risk_out["risk_policy"]
    (run_dir / "data" / "phase4_risk.md").write_text(
        risk_out["transcript"], encoding="utf-8")
    (run_dir / "data" / "phase4_risk_policy.json").write_text(
        json.dumps(risk_policy.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")

    # ── Phase 5: 综合评分 + 最终决策 ──
    final_score, score_components = _compose_final_score(
        experts, investment_plan, trading_plan, risk_policy)
    final_decision, decision_level = _decide_level(final_score, trading_plan.final_decision)

    result = {
        "version": "v3",
        "symbol": symbol,
        "name": name,
        "generated_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "duration_sec": round(time.time() - t0, 2),
        # 融合结果
        "final_score": final_score,
        "final_decision": final_decision,
        "decision_level": decision_level,
        "score_components": score_components,
        "trader_decision": trading_plan.final_decision,
        # Phase 产出
        "meta": bundle.meta,
        "experts": {r: e.to_dict() for r, e in experts.items()},
        "investment_plan": investment_plan.to_dict(),
        "trading_plan": trading_plan.to_dict(),
        "risk_policy": risk_policy.to_dict(),
        "bull_rounds": debate_out["bull_rounds"],
        "bear_rounds": debate_out["bear_rounds"],
        "risk_aggressive_rounds": risk_out["aggressive_rounds"],
        "risk_conservative_rounds": risk_out["conservative_rounds"],
        "risk_neutral_rounds": risk_out["neutral_rounds"],
        # 提示词与流水线配置
        "config": {
            "bull_provider": bull_provider,
            "bear_provider": bear_provider,
            "judge_provider": judge_provider,
            "trader_provider": trader_provider,
            "pm_provider": pm_provider,
            "debate_rounds": debate_rounds,
            "risk_rounds": risk_rounds,
        },
    }

    (run_dir / "final_decision_v3.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成 markdown 总报告(原始, 供调试阅读)
    md_report = _build_markdown_report(result, bundle, debate_out, risk_out)
    (run_dir / f"{symbol}_{name}_v3_报告.md").write_text(md_report, encoding="utf-8")

    # 生成机构级 PDF 报告
    try:
        from .report_pdf_v3 import build_investor_pdf_v3
        pdf_path = build_investor_pdf_v3(run_dir, result)
        result["pdf_path"] = str(pdf_path)
        logger.info("[v3] PDF 报告生成: %s", pdf_path)
    except Exception as e:
        logger.warning("[v3] PDF 生成失败(不影响流水线): %s", e)

    # ── 保存记忆 ──
    if save_memory:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        situation_rec = situation_summary[:500]
        get_memory("bull").add(MemoryRecord(
            ts=ts, symbol=symbol, situation=situation_rec,
            decision=(debate_out["bull_rounds"][-1] if debate_out["bull_rounds"] else "")[:500],
        ))
        get_memory("bear").add(MemoryRecord(
            ts=ts, symbol=symbol, situation=situation_rec,
            decision=(debate_out["bear_rounds"][-1] if debate_out["bear_rounds"] else "")[:500],
        ))
        get_memory("judge").add(MemoryRecord(
            ts=ts, symbol=symbol, situation=situation_rec,
            decision=f"{investment_plan.direction} conf={investment_plan.confidence} winner={investment_plan.winner}",
        ))
        get_memory("trader").add(MemoryRecord(
            ts=ts, symbol=symbol, situation=situation_rec,
            decision=f"{trading_plan.final_decision} pos={trading_plan.initial_position_ratio} strat={trading_plan.primary_strategy}",
        ))
        get_memory("pm").add(MemoryRecord(
            ts=ts, symbol=symbol, situation=situation_rec,
            decision=f"rating={risk_policy.final_risk_rating} maxpos={risk_policy.max_position_ratio} align={risk_policy.alignment_with}",
        ))

    logger.info("[v3] 完成 %s final=%.2f decision=%s dur=%.1fs",
                symbol, final_score, final_decision, result["duration_sec"])
    return result


def _build_markdown_report(result: dict[str, Any], bundle: ReportBundle,
                            debate_out: dict[str, Any], risk_out: dict[str, Any]) -> str:
    symbol = result["symbol"]
    name = result["name"]
    ip = result["investment_plan"]
    tp = result["trading_plan"]
    rp = result["risk_policy"]
    sc = result["score_components"]

    lines = [
        f"# v3 多智能体投研报告 · {symbol} {name}",
        f"",
        f"**生成时间**: {result['generated_at']}  |  **耗时**: {result['duration_sec']}s",
        f"",
        f"## 最终决策",
        f"- **综合评分**: {result['final_score']}",
        f"- **最终决策**: {result['final_decision'].upper()} ({result['decision_level']})",
        f"- **评分拆解**: 专家均值 {sc['expert_avg']} × 0.5 + Judge {sc['judge_score']} × 0.3 + 风控 {sc['risk_mapped']} × 0.2",
        f"",
        f"## Phase 2 研究主管 investment_plan",
        f"- 方向: **{ip.get('direction')}**  置信度: {ip.get('confidence')}  胜方: {ip.get('winner')}",
        f"- 核心理由:",
    ]
    for r in ip.get("key_reasons", []) or []:
        lines.append(f"  - {r}")
    lines += [
        f"- 时间维度: {ip.get('time_horizon')}",
        f"- 支撑位: {ip.get('key_support')} | 阻力位: {ip.get('key_resistance')}",
        f"- 仲裁点评: {ip.get('judge_comment')}",
        f"",
        f"## Phase 3 首席交易员 trading_plan",
        f"- 最终决策: **{tp.get('final_decision')}**",
        f"- 首次建仓比例: {tp.get('initial_position_ratio')}",
        f"- 主策略: {tp.get('primary_strategy')}  |  周期: {tp.get('time_horizon')}",
        f"- 入场方案:",
    ]
    for ep in tp.get("entry_plans", []) or []:
        lines.append(
            f"  - {ep.get('strategy')}: 入场={ep.get('entry')} T1={ep.get('target_1')} T2={ep.get('target_2')} 止损={ep.get('stop')} RR={ep.get('rr')}")
    lines += [
        f"- 持有条件: {tp.get('hold_conditions')}",
        f"- 离场条件: {tp.get('exit_conditions')}",
        f"- 交易逻辑: {tp.get('reasoning')}",
        f"- {tp.get('final_line', '')}",
        f"",
        f"## Phase 4 风控 Portfolio Manager",
        f"- 最终风险评级: **{rp.get('final_risk_rating')}**  |  认同方: {rp.get('alignment_with')}",
        f"- 最大仓位: {rp.get('max_position_ratio')}  |  首次建仓: {rp.get('initial_position_ratio')}",
        f"- 止损纪律: {rp.get('stop_loss_discipline')}",
        f"- 止盈规则: {rp.get('take_profit_rule')}",
        f"- 加仓条件: {rp.get('add_position_condition')}",
        f"- 减仓条件: {rp.get('reduce_position_condition')}",
        f"- 黑天鹅应对: {rp.get('black_swan_response')}",
        f"- 风控总结: {rp.get('pm_summary')}",
        f"",
        f"## Phase 1 专家研判",
    ]
    for role, er in result["experts"].items():
        lines.append(f"- **{er.get('role_cn')}** ({er.get('provider')}) score={er.get('score')}")
        lines.append(f"  - 研判: {er.get('analysis')}")
        lines.append(f"  - 风险: {er.get('risk')}")

    lines += [f"", f"## Phase 0 量化事实报告(摘要)", bundle.technical, "", bundle.structure]
    lines += [f"", f"## Phase 2 多空辩论全文", debate_out["transcript"]]
    lines += [f"", f"## Phase 4 风控辩论全文", risk_out["transcript"]]
    return "\n".join(lines)
