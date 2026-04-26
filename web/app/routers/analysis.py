"""分析 API: 提交 / 预览 / 一键跟踪 / 结果详情。"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import get_db
from ..core.deps import get_current_user
from ..models import AnalysisJob, AnalysisResult, AnalysisType, ResultStatus, User
from ..schemas.analysis import (
    AnalyzePreviewItem, AnalyzePreviewResponse, AnalyzeRequest,
    JobBriefResponse, ResultDetailResponse,
)
from ..core.redis import subscribe_progress
from ..services.analysis_runner import (
    determine_analysis_type, find_latest_full_score, submit_analysis,
)
from ..services.points_service import InsufficientPointsError
from ..services.progress_service import list_progress_events

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/analyze/preview", response_model=AnalyzePreviewResponse)
async def preview_analysis(
    body: AnalyzeRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """预判每只股票的类型和扣分(不真扣)。"""
    force_set = {s.strip().upper() for s in body.force_full}
    items: list[AnalyzePreviewItem] = []
    total = 0
    for sym in body.symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        a_type, pts, cache = await determine_analysis_type(db, sym, force_full=(sym in force_set))
        # 取最近一次 full 评分供前端展示
        last_full = await find_latest_full_score(db, sym)
        items.append(AnalyzePreviewItem(
            symbol=sym,
            type="cache_hit" if cache else a_type.value,
            points=pts,
            last_full_score=last_full.final_score if last_full else None,
            last_full_at=last_full.created_at if last_full else None,
        ))
        total += pts

    return AnalyzePreviewResponse(
        items=items, total_points=total,
        user_points=user.points, enough=user.points >= total,
    )


@router.post("/analyze")
async def create_analysis(
    body: AnalyzeRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    force_set = {s.strip().upper() for s in body.force_full}
    try:
        job = await submit_analysis(db, user, body.symbols, force_full_set=force_set)
    except InsufficientPointsError as e:
        raise HTTPException(402, f"积分不足, 需要 {e.need} 你有 {e.have}")
    except ValueError as e:
        raise HTTPException(400, str(e))

    # 拿出 results 供返回
    res = await db.execute(
        select(AnalysisResult).where(AnalysisResult.job_id == job.id)
    )
    results = list(res.scalars().all())

    return {
        "job_id": job.id,
        "total_points_charged": job.total_points_charged,
        "status": job.status.value,
        "results": [
            {
                "id": r.id, "symbol": r.symbol,
                "type": r.analysis_type.value, "status": r.status.value,
                "is_cache_hit": r.is_cache_hit,
                "points_charged": r.points_charged,
            }
            for r in results
        ],
    }


@router.post("/analyze/track-all")
async def track_all(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """一键跟踪: 对当前用户所有曾跑过 full 评分的股票, 批量做量化。"""
    res = await db.execute(
        select(AnalysisResult.symbol).where(
            AnalysisResult.user_id == user.id,
            AnalysisResult.analysis_type == AnalysisType.full,
            AnalysisResult.status == ResultStatus.done,
        ).distinct()
    )
    symbols = sorted({r[0] for r in res.all()})
    if not symbols:
        raise HTTPException(400, "你还没有 LLM 全量评分历史")

    body = AnalyzeRequest(symbols=symbols, force_full=[])
    try:
        job = await submit_analysis(db, user, body.symbols)
    except InsufficientPointsError as e:
        raise HTTPException(402, f"积分不足, 需要 {e.need} 你有 {e.have}")
    return {"job_id": job.id, "symbols": symbols, "total_points": job.total_points_charged}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    res = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = res.scalar_one_or_none()
    if job is None:
        raise HTTPException(404, "任务不存在")
    if job.user_id != user.id and not user.is_admin:
        raise HTTPException(403, "无权查看")

    res2 = await db.execute(
        select(AnalysisResult).where(AnalysisResult.job_id == job_id).order_by(AnalysisResult.id)
    )
    results = list(res2.scalars().all())
    return {
        "id": job.id, "user_id": job.user_id,
        "symbols_count": job.symbols_count,
        "total_points_charged": job.total_points_charged,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "breakdown": [
            {
                "id": r.id, "symbol": r.symbol, "type": r.analysis_type.value,
                "status": r.status.value, "progress_pct": r.progress_pct,
                "current_phase": r.current_phase,
                "is_cache_hit": r.is_cache_hit,
                "final_score": r.final_score,
                "decision_level": r.decision_level,
                "quant_score": r.quant_score,
                "error_message": r.error_message,
                "duration_sec": r.duration_sec,
            }
            for r in results
        ],
    }


@router.post("/jobs/repair-status")
async def repair_job_status(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """修复历史卡住状态的 job: 子 result 全完成但 job 仍是 running 的, 重算状态。"""
    from datetime import datetime, timezone
    from ..models import JobStatus
    res = await db.execute(
        select(AnalysisJob).where(
            AnalysisJob.user_id == user.id,
            AnalysisJob.status.in_((JobStatus.pending, JobStatus.running)),
        )
    )
    fixed = 0
    for job in res.scalars().all():
        rs = await db.execute(select(AnalysisResult).where(AnalysisResult.job_id == job.id))
        rows = list(rs.scalars().all())
        if not rows:
            continue
        pending = sum(1 for r in rows if r.status in (ResultStatus.queued, ResultStatus.running))
        done = sum(1 for r in rows if r.status == ResultStatus.done)
        bad = sum(1 for r in rows if r.status in (ResultStatus.failed, ResultStatus.refunded))
        if pending > 0:
            continue
        if done == len(rows):
            job.status = JobStatus.done
        elif done > 0 and bad > 0:
            job.status = JobStatus.partial_done
        else:
            job.status = JobStatus.failed
        job.finished_at = datetime.now(timezone.utc)
        fixed += 1
    await db.commit()
    return {"repaired": fixed}


@router.get("/jobs")
async def my_jobs(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    total = await db.scalar(
        select(func.count()).select_from(AnalysisJob).where(AnalysisJob.user_id == user.id)
    )
    res = await db.execute(
        select(AnalysisJob).where(AnalysisJob.user_id == user.id)
        .order_by(desc(AnalysisJob.created_at)).limit(limit).offset(offset)
    )
    jobs = list(res.scalars().all())
    job_ids = [j.id for j in jobs]

    # 一次拉所有 job 的子 result, 按 job 分组聚合 done/failed 计数 + 股票列表
    breakdowns: dict[int, list[AnalysisResult]] = {jid: [] for jid in job_ids}
    if job_ids:
        rs = await db.execute(
            select(AnalysisResult).where(AnalysisResult.job_id.in_(job_ids))
            .order_by(AnalysisResult.id)
        )
        for r in rs.scalars().all():
            breakdowns.setdefault(r.job_id, []).append(r)

    items = []
    for j in jobs:
        rows = breakdowns.get(j.id, [])
        done = sum(1 for r in rows if r.status == ResultStatus.done)
        failed = sum(1 for r in rows if r.status in (ResultStatus.failed, ResultStatus.refunded))
        symbols = [
            {"symbol": r.symbol, "name": r.name or "",
             "status": r.status.value, "final_score": r.final_score}
            for r in rows
        ]
        items.append({
            "id": j.id, "symbols_count": j.symbols_count,
            "total_points_charged": j.total_points_charged,
            "status": j.status.value,
            "created_at": j.created_at.isoformat(),
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            "done_count": done, "failed_count": failed,
            "symbols": symbols,
        })

    return {
        "items": items,
        "total": int(total or 0), "limit": limit, "offset": offset,
    }


@router.get("/results/{result_id}", response_model=ResultDetailResponse)
async def get_result_detail(
    result_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    res = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
    rec = res.scalar_one_or_none()
    if rec is None:
        raise HTTPException(404, "结果不存在")
    if rec.user_id != user.id and not user.is_admin:
        raise HTTPException(403, "无权查看")

    return ResultDetailResponse(
        id=rec.id, symbol=rec.symbol, name=rec.name,
        analysis_type=rec.analysis_type.value,
        is_cache_hit=rec.is_cache_hit,
        parent_full_result_id=rec.parent_full_result_id,
        status=rec.status.value, current_phase=rec.current_phase,
        progress_pct=rec.progress_pct, points_charged=rec.points_charged,
        final_score=rec.final_score, decision_level=rec.decision_level,
        quant_score=rec.quant_score, trader_decision=rec.trader_decision,
        expert_scores_json=rec.expert_scores_json,
        score_components_json=rec.score_components_json,
        quant_components_json=rec.quant_components_json,
        error_message=rec.error_message, duration_sec=rec.duration_sec,
        created_at=rec.created_at, finished_at=rec.finished_at,
        run_dir=rec.run_dir,
    )


@router.get("/results/{result_id}/stream")
async def stream_progress(
    result_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """SSE 实时进度推送。先回放历史事件, 再订阅新事件。"""
    from sse_starlette.sse import EventSourceResponse

    res = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
    rec = res.scalar_one_or_none()
    if rec is None:
        raise HTTPException(404, "结果不存在")
    if rec.user_id != user.id and not user.is_admin:
        raise HTTPException(403, "无权查看")

    async def event_generator():
        # 1. 回放历史
        history = await list_progress_events(db, result_id)
        for h in history:
            yield {"event": "progress", "data": __import__("json").dumps(h, ensure_ascii=False)}

        # 2. 已完成 → 直接发 done 事件并结束
        if rec.status.value in ("done", "failed", "refunded"):
            yield {
                "event": rec.status.value,
                "data": __import__("json").dumps({
                    "result_id": rec.id,
                    "final_score": rec.final_score,
                    "decision_level": rec.decision_level,
                    "error": rec.error_message,
                }, ensure_ascii=False),
            }
            return

        # 3. 实时订阅
        async for evt in subscribe_progress(result_id):
            yield {
                "event": evt.get("type", "message"),
                "data": __import__("json").dumps(evt, ensure_ascii=False),
            }
            if evt.get("type") in ("done", "failed"):
                break

    return EventSourceResponse(event_generator())


@router.get("/stocks/{symbol}/kline")
async def stock_kline(
    symbol: str,
    user: Annotated[User, Depends(get_current_user)],
    days: int = Query(180, ge=20, le=720),
):
    """OHLC + 成交量, 供 ECharts 蜡烛图。"""
    import asyncio
    def _fetch():
        from stockagent_analysis.tushare_enrich import _get_pro
        pro = _get_pro()
        if pro is None:
            raise HTTPException(503, "Tushare 未初始化")
        from datetime import date, timedelta
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=days * 2)).strftime("%Y%m%d")
        df = pro.daily(ts_code=symbol.upper(), start_date=start, end_date=end)
        if df is None or df.empty:
            return []
        df = df.sort_values("trade_date").tail(days)
        return [
            {
                "date": str(r.trade_date),
                "open": float(r.open), "close": float(r.close),
                "low": float(r.low),   "high": float(r.high),
                "volume": float(r.vol or 0),
                "pct_chg": float(r.pct_chg or 0),
            }
            for r in df.itertuples()
        ]
    rows = await asyncio.to_thread(_fetch)
    return {"symbol": symbol.upper(), "rows": rows}


@router.get("/stocks/{symbol}/history")
async def stock_history(
    symbol: str,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    type_filter: str | None = Query(None, alias="type"),
    limit: int = Query(50, ge=1, le=200),
):
    """同股时序所有分析(走势图数据)。type=all/full/quant_only。"""
    q = select(AnalysisResult).where(
        AnalysisResult.symbol == symbol.upper(),
        AnalysisResult.user_id == user.id,
        AnalysisResult.status == ResultStatus.done,
    )
    if type_filter == "full":
        q = q.where(AnalysisResult.analysis_type == AnalysisType.full)
    elif type_filter == "quant_only":
        q = q.where(AnalysisResult.analysis_type == AnalysisType.quant_only)

    res = await db.execute(q.order_by(AnalysisResult.created_at).limit(limit))
    items = list(res.scalars().all())
    return {
        "symbol": symbol,
        "items": [
            {
                "id": r.id, "type": r.analysis_type.value,
                "final_score": r.final_score, "decision_level": r.decision_level,
                "quant_score": r.quant_score,
                "created_at": r.created_at.isoformat(),
            } for r in items
        ],
    }
