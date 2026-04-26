"""分析运行器: 双模式调度 + 缓存命中检测 + 失败自动退款。

核心入口:
- determine_analysis_type: 智能预判 full / quant_only / cache_hit
- submit_analysis: 创建 job + 扣款 + 异步执行
- run_full_async: 调用 v3.1 完整流水线 (7min)
- run_quant_async: 调用 quant_only 模块 (10s)
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.db import get_session_factory
from ..models import (
    AnalysisJob, AnalysisResult, AnalysisType, JobStatus, ResultStatus,
    TransactionReason, User,
)
from .points_service import deduct_points, refund_points
from .progress_service import emit_done, emit_failed, emit_progress

logger = logging.getLogger(__name__)


# 把主项目 src/ 加入 sys.path 让我们能 import stockagent_analysis 模块
_PROJECT_SRC = settings.project_root / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))


# ─────────────── 交易日工具 ───────────────

def _cache_window_start() -> datetime:
    """缓存窗口起点: 当前时间往回推 24 小时 (近似 "当日缓存")。

    简化逻辑: A 股一个交易日内 Tushare 数据基本不变, 24h 窗口足够判定.
    用 UTC 存储, 跨时区无歧义.
    """
    return datetime.now(timezone.utc) - timedelta(hours=24)


# ─────────────── 类型预判 ───────────────

async def has_any_full_score(db: AsyncSession, symbol: str) -> bool:
    """系统中任何用户曾对此股做过 full 评分?"""
    res = await db.execute(
        select(AnalysisResult.id).where(
            AnalysisResult.symbol == symbol,
            AnalysisResult.analysis_type == AnalysisType.full,
            AnalysisResult.status == ResultStatus.done,
        ).limit(1)
    )
    return res.scalar_one_or_none() is not None


async def find_full_cache_today(db: AsyncSession, symbol: str) -> AnalysisResult | None:
    """查当日是否有 full 评分可复用 (任何用户)。"""
    today_start = _cache_window_start()
    res = await db.execute(
        select(AnalysisResult).where(
            and_(
                AnalysisResult.symbol == symbol,
                AnalysisResult.analysis_type == AnalysisType.full,
                AnalysisResult.status == ResultStatus.done,
                AnalysisResult.created_at >= today_start,
            )
        ).order_by(desc(AnalysisResult.created_at)).limit(1)
    )
    return res.scalar_one_or_none()


async def find_quant_cache_today(db: AsyncSession, symbol: str) -> AnalysisResult | None:
    """查当日是否有 quant 结果可复用。"""
    today_start = _cache_window_start()
    res = await db.execute(
        select(AnalysisResult).where(
            and_(
                AnalysisResult.symbol == symbol,
                AnalysisResult.analysis_type == AnalysisType.quant_only,
                AnalysisResult.status == ResultStatus.done,
                AnalysisResult.created_at >= today_start,
            )
        ).order_by(desc(AnalysisResult.created_at)).limit(1)
    )
    return res.scalar_one_or_none()


async def find_latest_full_score(db: AsyncSession, symbol: str) -> AnalysisResult | None:
    """找最近一次 full 评分(任何时间, 任何用户)。"""
    res = await db.execute(
        select(AnalysisResult).where(
            AnalysisResult.symbol == symbol,
            AnalysisResult.analysis_type == AnalysisType.full,
            AnalysisResult.status == ResultStatus.done,
        ).order_by(desc(AnalysisResult.created_at)).limit(1)
    )
    return res.scalar_one_or_none()


async def determine_analysis_type(
    db: AsyncSession, symbol: str, force_full: bool = False,
) -> tuple[AnalysisType, int, AnalysisResult | None]:
    """返回 (analysis_type, points_to_charge, source_cache_record)。

    - 无 LLM 历史 → full (20pt)
    - force_full + 当日已有缓存 → full + 10pt(命中)
    - force_full + 当日无缓存 → full (20pt)
    - 默认: 已有 LLM 历史 → quant_only (1pt)
    """
    has_llm = await has_any_full_score(db, symbol)

    if not has_llm:
        return AnalysisType.full, settings.points_analyze_full_cost, None

    if force_full:
        cache = await find_full_cache_today(db, symbol)
        if cache:
            return AnalysisType.full, settings.points_analyze_full_cache_hit, cache
        return AnalysisType.full, settings.points_analyze_full_cost, None

    return AnalysisType.quant_only, settings.points_analyze_quant_cost, None


# ─────────────── 提交 ───────────────

async def submit_analysis(
    db: AsyncSession, user: User, symbols: list[str], force_full_set: set[str] | None = None,
) -> AnalysisJob:
    """创建任务 + 扣款 + 异步执行。"""
    if not symbols:
        raise ValueError("至少提交一只股票")
    force_full_set = force_full_set or set()

    # 预判每只股
    breakdown: list[dict] = []
    total_points = 0
    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        a_type, pts, cache_src = await determine_analysis_type(
            db, sym, force_full=(sym in force_full_set))
        breakdown.append({
            "symbol": sym, "type": a_type, "points": pts, "cache_src": cache_src,
        })
        total_points += pts

    if user.points < total_points:
        from .points_service import InsufficientPointsError
        raise InsufficientPointsError(need=total_points, have=user.points)

    # 创建 job
    job = AnalysisJob(
        user_id=user.id, symbols_count=len(breakdown),
        total_points_charged=total_points, status=JobStatus.pending,
    )
    db.add(job)
    await db.flush()

    # 创建每只股的 result 占位
    results: list[AnalysisResult] = []
    for item in breakdown:
        a_type = item["type"]
        cache_src = item["cache_src"]
        pts = item["points"]

        if a_type == AnalysisType.full and cache_src is not None:
            # full 命中缓存: 直接复制源记录的核心字段, 状态 done
            r = AnalysisResult(
                job_id=job.id, user_id=user.id, symbol=item["symbol"],
                name=cache_src.name, run_dir=cache_src.run_dir,
                analysis_type=AnalysisType.full,
                is_cache_hit=True, source_result_id=cache_src.id,
                points_charged=pts,
                status=ResultStatus.done,
                progress_pct=100, current_phase="cache_hit",
                final_score=cache_src.final_score,
                decision_level=cache_src.decision_level,
                quant_score=cache_src.quant_score,
                trader_decision=cache_src.trader_decision,
                expert_scores_json=cache_src.expert_scores_json,
                score_components_json=cache_src.score_components_json,
                quant_components_json=cache_src.quant_components_json,
                duration_sec=0,
                finished_at=datetime.now(timezone.utc),
            )
        else:
            r = AnalysisResult(
                job_id=job.id, user_id=user.id, symbol=item["symbol"],
                analysis_type=a_type,
                points_charged=pts,
                status=ResultStatus.queued,
                progress_pct=0, current_phase="queued",
            )
        db.add(r)
        results.append(r)
    await db.flush()

    # 扣款 (一次性整笔)
    reason_map = {
        AnalysisType.full: TransactionReason.analyze_full,
        AnalysisType.quant_only: TransactionReason.analyze_quant,
    }
    # 简化: 按总额扣一次, related_result_id 记第一条
    if total_points > 0:
        # 关联到第一只股的 result_id (流水可读性)
        primary_result_id = results[0].id if results else None
        await deduct_points(
            db, user, total_points,
            reason=reason_map.get(breakdown[0]["type"], TransactionReason.analyze_full),
            related_result_id=primary_result_id,
            note=f"分析 {len(breakdown)} 只股票",
            auto_commit=False,
        )

    if any(r.status == ResultStatus.queued for r in results):
        job.status = JobStatus.running
    else:
        job.status = JobStatus.done
        job.finished_at = datetime.now(timezone.utc)
    await db.commit()

    # 异步触发未完成的(非缓存)
    factory = get_session_factory()
    for r in results:
        if r.status == ResultStatus.queued:
            asyncio.create_task(_run_one(factory, r.id, r.analysis_type, r.symbol))

    return job


# ─────────────── 异步执行 ───────────────

async def _aggregate_job_status(db: AsyncSession, job_id: int) -> None:
    """根据子 result 状态聚合 job.status; 全完成时回填 finished_at。"""
    job_res = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = job_res.scalar_one_or_none()
    if job is None:
        return
    rs = await db.execute(select(AnalysisResult).where(AnalysisResult.job_id == job_id))
    rows = list(rs.scalars().all())
    if not rows:
        return
    pending = sum(1 for r in rows if r.status in (ResultStatus.queued, ResultStatus.running))
    done = sum(1 for r in rows if r.status == ResultStatus.done)
    bad = sum(1 for r in rows if r.status in (ResultStatus.failed, ResultStatus.refunded))
    if pending > 0:
        job.status = JobStatus.running
    elif done == len(rows):
        job.status = JobStatus.done
        job.finished_at = datetime.now(timezone.utc)
    elif done > 0 and bad > 0:
        job.status = JobStatus.partial_done
        job.finished_at = datetime.now(timezone.utc)
    else:
        job.status = JobStatus.failed
        job.finished_at = datetime.now(timezone.utc)
    await db.commit()


async def _run_one(factory, result_id: int, a_type: AnalysisType, symbol: str):
    """单只股票异步执行 (在新 session 里)。"""
    async with factory() as db:
        # 取出最新对象
        res = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
        rec = res.scalar_one_or_none()
        if rec is None:
            return
        job_id = rec.job_id
        rec.status = ResultStatus.running
        rec.current_phase = "starting"
        await db.commit()

        try:
            if a_type == AnalysisType.quant_only:
                await _do_quant_only(db, rec)
            else:
                await _do_full(db, rec)
        except Exception as e:
            logger.exception("[analysis] %s failed: %s", symbol, e)
            await _mark_failed_and_refund(db, rec, str(e))
        finally:
            try:
                await _aggregate_job_status(db, job_id)
            except Exception as e:
                logger.error("[analysis] aggregate job status failed: %s", e)


async def _do_quant_only(db: AsyncSession, rec: AnalysisResult):
    """quant_only: 拉 Tushare + 算 quant_score + 入库。10s 内完成。"""
    from stockagent_analysis.tushare_enrich import compute_quant_score, enrich_with_tushare

    await emit_progress(db, rec.id, phase_id="fetching_tushare", percent=20,
                         message=f"正在拉取 {rec.symbol} 最新数据...")

    # 阻塞 IO 放到线程池
    ts_enrich = await asyncio.to_thread(
        enrich_with_tushare, rec.symbol, run_dir=None, use_cache=True,
    )

    await emit_progress(db, rec.id, phase_id="computing_quant", percent=70,
                         message="正在计算 4 维量化指标...",
                         data={
                             "adx": (ts_enrich or {}).get("tushare_factors", {}).get("dmi_adx"),
                             "winner_rate": (ts_enrich or {}).get("tushare_cyq", {}).get("winner_rate"),
                         } if ts_enrich else None)

    quant_info = compute_quant_score(ts_enrich or {})
    score = float(quant_info.get("quant_score", 50.0))

    # 决策等级映射 (沿用 weak_buy/hold/weak_sell 与 LLM 一致)
    if score >= 80: level = "strong_buy"
    elif score >= 72: level = "weak_buy"
    elif score >= 62: level = "hold"
    elif score >= 52: level = "watch_sell"
    elif score >= 42: level = "weak_sell"
    else: level = "strong_sell"

    parent = await find_latest_full_score(db, rec.symbol)

    rec.final_score = score
    rec.decision_level = level
    rec.quant_score = score
    rec.quant_components_json = quant_info
    rec.parent_full_result_id = parent.id if parent else None
    rec.status = ResultStatus.done
    rec.progress_pct = 100
    rec.current_phase = "done"
    rec.finished_at = datetime.now(timezone.utc)
    await db.commit()

    await emit_progress(db, rec.id, phase_id="done", percent=100,
                         message=f"量化评分完成: {score:.1f} ({level})",
                         data={"quant_score": score, "level": level,
                               "adjustments": quant_info.get("adjustments")})
    await emit_done(db, rec.id, final_score=score, decision_level=level,
                    extra={"quant_score": score, "type": "quant_only"})

    logger.info("[quant_only] %s done: score=%.1f level=%s", rec.symbol, score, level)


async def _do_full(db: AsyncSession, rec: AnalysisResult):
    """full: 调用现有 v3.1 流水线 + 粗粒度 SSE 进度。"""
    from stockagent_analysis.agents_v3.orchestrator_v3 import run_analysis_v3

    await emit_progress(db, rec.id, phase_id="phase_-1_data", percent=5,
                         message=f"正在采集 {rec.symbol} 数据...")

    # 在线程池跑 (无法实时回调 SSE; 用主进程定时刷新进度)
    # 简化: 启动一个后台 polling 任务定时往 SSE 推 "still running"
    loop = asyncio.get_event_loop()
    cancel_event = asyncio.Event()

    async def _heartbeat():
        phases = [
            ("phase_0_data", 10, "数据采集完成, 生成 6 份报告"),
            ("phase_1_experts", 30, "4 专家并行分析中..."),
            ("phase_2_debate", 55, "多空辩论 4 轮 + Judge 仲裁..."),
            ("phase_3_trader", 75, "首席交易员产出方案..."),
            ("phase_4_risk", 90, "风控三辩论 + PM 拍板..."),
        ]
        idx = 0
        while not cancel_event.is_set() and idx < len(phases):
            await asyncio.sleep(60)   # 大约每分钟推一次
            if cancel_event.is_set(): break
            phase, pct, msg = phases[idx]
            await emit_progress(db, rec.id, phase_id=phase, percent=pct, message=msg)
            idx += 1

    hb_task = asyncio.create_task(_heartbeat())

    try:
        result = await asyncio.to_thread(
            run_analysis_v3, rec.symbol, name="", debate_rounds=4, risk_rounds=3,
        )
    finally:
        cancel_event.set()
        hb_task.cancel()

    sc = result.get("score_components", {})
    rec.run_dir = str(result.get("run_dir", ""))
    rec.name = result.get("name", "")
    rec.final_score = float(result.get("final_score", 0))
    rec.decision_level = result.get("decision_level")
    rec.quant_score = sc.get("quant_score")
    rec.trader_decision = result.get("trader_decision")
    rec.expert_scores_json = {k: v for k, v in (sc.get("expert_details") or {}).items()} \
        if isinstance(sc.get("expert_details"), dict) else {"details": sc.get("expert_details")}
    rec.score_components_json = sc
    rec.quant_components_json = sc.get("quant_components")
    rec.duration_sec = int(result.get("duration_sec", 0))
    rec.status = ResultStatus.done
    rec.progress_pct = 100
    rec.current_phase = "done"
    rec.finished_at = datetime.now(timezone.utc)
    await db.commit()

    await emit_progress(db, rec.id, phase_id="done", percent=100,
                         message=f"分析完成: {rec.final_score:.1f} ({rec.decision_level})")
    await emit_done(db, rec.id, final_score=rec.final_score,
                    decision_level=rec.decision_level,
                    extra={"quant_score": rec.quant_score, "type": "full"})

    logger.info("[full] %s done: score=%.1f level=%s",
                rec.symbol, rec.final_score, rec.decision_level)


async def _mark_failed_and_refund(db: AsyncSession, rec: AnalysisResult, err: str):
    """失败标记 + 自动退款。"""
    rec.status = ResultStatus.failed
    rec.error_message = err[:500]
    rec.finished_at = datetime.now(timezone.utc)
    await db.commit()

    # 退款给用户
    user_res = await db.execute(select(User).where(User.id == rec.user_id))
    user = user_res.scalar_one_or_none()
    if user is None:
        return
    refunded_amount = 0
    try:
        await refund_points(
            db, user, rec.points_charged,
            related_result_id=rec.id,
            note=f"分析失败自动退: {err[:80]}",
        )
        rec.status = ResultStatus.refunded
        await db.commit()
        refunded_amount = rec.points_charged
        logger.warning("[analysis] 已退款: user=%d result=%d amount=%d",
                       user.id, rec.id, rec.points_charged)
    except Exception as e:
        logger.error("[analysis] 退款失败 user=%d: %s", user.id, e)

    # 推 SSE 失败事件 + 退款金额
    try:
        await emit_failed(db, rec.id, error=err, refunded=refunded_amount)
    except Exception as e:
        logger.error("[analysis] 推送失败事件失败: %s", e)
