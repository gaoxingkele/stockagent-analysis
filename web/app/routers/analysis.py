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
from ..services.analysis_runner import (
    determine_analysis_type, find_latest_full_score, submit_analysis,
)
from ..services.points_service import InsufficientPointsError

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
    return {
        "items": [
            {
                "id": j.id, "symbols_count": j.symbols_count,
                "total_points_charged": j.total_points_charged,
                "status": j.status.value,
                "created_at": j.created_at.isoformat(),
                "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            } for j in jobs
        ],
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
