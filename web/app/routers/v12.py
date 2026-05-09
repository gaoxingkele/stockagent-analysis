"""V12 路由: 单股实时评分 + 全市场异步推理 + LLM 视觉过滤 + SSE 进度.

SSE 进度复用 /api/results/{id}/stream (analysis.py).
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import get_db
from ..core.deps import get_current_user
from ..models import User
from ..schemas.v12 import (
    RunLlmFilterRequest, RunMarketRequest, V12ContradictionItem,
    V12DatesResponse, V12JobResponse, V12RecommendResponse, V12ScoreResponse,
)
from ..services import v12_service
from ..services.points_service import InsufficientPointsError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v12", tags=["v12"])


@router.get("/dates", response_model=V12DatesResponse)
async def list_dates(user: Annotated[User, Depends(get_current_user)]):
    dates = v12_service.list_available_dates()
    return V12DatesResponse(dates=dates, latest=dates[-1] if dates else None)


@router.get("/recommend", response_model=V12RecommendResponse)
async def get_recommend(
    user: Annotated[User, Depends(get_current_user)],
    date: str = Query(..., pattern=r"^\d{8}$"),
):
    data = v12_service.read_recommend(date)
    if data["total"] == 0:
        raise HTTPException(404, f"{date} 无 V12 推荐数据 - 请先跑全市场推理")
    return V12RecommendResponse(**{k: v for k, v in data.items() if k != "source_file"})


@router.get("/contradiction")
async def get_contradiction(
    user: Annotated[User, Depends(get_current_user)],
    date: str = Query(..., pattern=r"^\d{8}$"),
):
    items = v12_service.read_contradiction(date)
    return {"date": date, "total": len(items), "items": items}


@router.get("/score/{symbol}", response_model=V12ScoreResponse)
async def get_score(
    symbol: str,
    user: Annotated[User, Depends(get_current_user)],
    date: str = Query(..., pattern=r"^\d{8}$"),
):
    """单股实时 V12 评分 (同步 ~5s, 不入 job, 不扣分)."""
    try:
        result = await v12_service.score_stock_now(symbol, date)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, f"数据文件缺失: {e}")
    return V12ScoreResponse(**result)


@router.post("/jobs/market", response_model=V12JobResponse)
async def run_market(
    body: RunMarketRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """触发 V12 全市场推理 (异步, 5pt). 返回 result_id 用于 SSE."""
    try:
        rec = await v12_service.submit_v12_market(db, user, body.date)
    except InsufficientPointsError as e:
        raise HTTPException(402, f"积分不足, 需要 {e.need} 你有 {e.have}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return V12JobResponse(
        job_id=rec.job_id, result_id=rec.id, status=rec.status.value,
        points_charged=rec.points_charged,
        message=f"V12 全市场推理已启动, 通过 /api/results/{rec.id}/stream 监听进度",
    )


@router.post("/jobs/llm-filter", response_model=V12JobResponse)
async def run_llm_filter(
    body: RunLlmFilterRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """触发 V11 LLM 视觉过滤 (异步, 1pt/股 + LLM 成本). 默认跑当日全部矛盾段."""
    try:
        rec = await v12_service.submit_v12_llm_filter(
            db, user, body.date, symbols=body.symbols, limit=body.limit,
        )
    except InsufficientPointsError as e:
        raise HTTPException(402, f"积分不足, 需要 {e.need} 你有 {e.have}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return V12JobResponse(
        job_id=rec.job_id, result_id=rec.id, status=rec.status.value,
        points_charged=rec.points_charged,
        message=f"V11 LLM 视觉过滤已启动 ({rec.symbols_count if hasattr(rec,'symbols_count') else '...'} 股), "
                 f"通过 /api/results/{rec.id}/stream 监听进度",
    )
