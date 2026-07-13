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
    RunLlmFilterRequest, RunMarketRequest, RunUpdateRequest, V12ContradictionItem,
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


@router.get("/pool-dates")
async def list_pool_dates(user: Annotated[User, Depends(get_current_user)]):
    """有四池看板数据的日期 (output/daily_pick/dashboard_*)."""
    dates = v12_service.list_pool_dates()
    return {"dates": dates, "latest": dates[-1] if dates else None}


@router.get("/pools")
async def get_pools(
    user: Annotated[User, Depends(get_current_user)],
    date: str = Query(..., pattern=r"^\d{8}$"),
):
    """四池看板: A系统推荐 / B自选 / C基金重仓 / D追高动量 (读 daily_dashboard 产出)."""
    data = v12_service.read_four_pools(date)
    if not data["exists"]:
        raise HTTPException(404, f"{date} 无四池数据 - 请先跑全市场推理生成四池看板")
    return data


@router.get("/semas-stocks")
async def get_semas_stocks(
    user: Annotated[User, Depends(get_current_user)],
    date: str = Query(..., pattern=r"^\d{8}$"),
):
    """SEMAS 最优组合推荐 (stock_benchmark 外部推荐 + r20/ratio 评分)."""
    data = v12_service.read_semas_stocks(date)
    if not data["exists"]:
        raise HTTPException(404, f"{date} 无 SEMAS 推荐数据")
    return data


@router.get("/index-metrics")
async def get_index_metrics(
    user: Annotated[User, Depends(get_current_user)],
    date: str = Query("", pattern=r"^(\d{8})?$"),
):
    """大盘指数量化指标 (r20/ratio 代理 + RSI + MA偏离)."""
    indices = v12_service.compute_index_metrics(date)
    return {"date": date or indices[0]["trade_date"] if indices else date, "indices": indices}


@router.get("/watchlist-b")
async def get_watchlist_b(user: Annotated[User, Depends(get_current_user)]):
    """池B 自选清单 (JSON 持久化, config/watchlist_b.json)."""
    return {"codes": v12_service.get_watchlist_b()}


@router.post("/watchlist-b")
async def add_watchlist_b(
    user: Annotated[User, Depends(get_current_user)],
    code: str = Query(..., description="6位.SZ/.SH 或纯6位"),
):
    """加一只到池B自选 (持久化; 次日四池重生成后入池打分)."""
    try:
        codes = v12_service.add_watchlist_b(code)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"codes": codes, "added": code}


@router.delete("/watchlist-b/{code}")
async def remove_watchlist_b(
    user: Annotated[User, Depends(get_current_user)],
    code: str,
):
    """从池B自选删一只 (持久化)."""
    return {"codes": v12_service.remove_watchlist_b(code), "removed": code}


@router.get("/pool-item")
async def get_pool_item(
    user: Annotated[User, Depends(get_current_user)],
    date: str = Query(..., pattern=r"^\d{8}$"),
    code: str = Query(...),
):
    """单 code 的池-item (加自选后即时注入当前视图; 读 scores_<date>.parquet)."""
    item = v12_service.score_pool_item(date, code)
    if not item:
        raise HTTPException(404, f"{date} 无 {code} 市场评分 (下次跑全市场后入池)")
    return item


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


@router.post("/jobs/update", response_model=V12JobResponse)
async def run_update(
    body: RunUpdateRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """更新数据: 拉最新交易日 daily/moneyflow + 重算特征 + 重跑五池 (异步). date 缺省=自动最新."""
    try:
        rec = await v12_service.submit_v12_update(db, user, body.date)
    except InsufficientPointsError as e:
        raise HTTPException(402, f"积分不足, 需要 {e.need} 你有 {e.have}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return V12JobResponse(
        job_id=rec.job_id, result_id=rec.id, status=rec.status.value,
        points_charged=rec.points_charged,
        message=f"更新数据+重跑五池已启动, 通过 /api/results/{rec.id}/stream 监听进度",
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
